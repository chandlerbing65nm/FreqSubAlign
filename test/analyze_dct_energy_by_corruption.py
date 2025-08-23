#!/usr/bin/env python3
"""
Analyze 2D DCT energy by radial frequency bands per corruption category on the test set.

For each corruption subfolder under --data_root, this script:
- Iterates over videos (recursively, by extension)
- Samples frames (configurable stride)
- Computes 2D DCT-II per frame/channel over (H,W) with 'ortho' norm
- Bins the coefficient grid into K radial bands (from DC at (0,0) outward) and measures energy in each band
- Aggregates per-video averages and then averages across videos in each corruption category

Outputs a readable table to stdout and optionally saves CSV/JSON.

Example:
    python -m test.analyze_dct_energy_by_corruption \
        --data_root /scratch/project_465001897/datasets/xxx/val_corruptions \
        --extensions mp4 avi \
        --frame_stride 4 \
        --k_bands 8 \
        --max_videos_per_corruption 200 \
        --save_csv dct_energy_summary.csv
"""

import argparse
import os
import sys
import json
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# DCT backends
try:
    from scipy.fft import dctn as _scipy_dctn
    _HAS_SCIPY_DCTN = True
except Exception:
    _HAS_SCIPY_DCTN = False
    try:
        from scipy.fftpack import dct as _scipy_dct1
    except Exception:
        _scipy_dct1 = None

# Wavelet backend (optional, only needed for wavelet entropy mode)
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

# Video IO
try:
    import decord
    from decord import VideoReader
    try:
        decord.bridge.set_bridge('native')
    except Exception:
        pass
except Exception:
    print("[ERROR] Decord is required. Install via `pip install decord`.", file=sys.stderr)
    raise


# ----------------------------- Utilities -----------------------------

def find_video_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith('.'):
                continue
            if os.path.splitext(fn)[1].lower().lstrip('.') in exts:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def frame_indices(num_frames: int, stride: int, max_frames: int = 0) -> List[int]:
    idx = list(range(0, num_frames, max(1, stride)))
    if max_frames and max_frames > 0:
        idx = idx[:max_frames]
    return idx


def _dct2(frame_hw: np.ndarray) -> np.ndarray:
    """2D DCT-II with orthonormalization."""
    if _HAS_SCIPY_DCTN:
        return _scipy_dctn(frame_hw, type=2, norm='ortho')
    if _scipy_dct1 is not None:
        return _scipy_dct1(_scipy_dct1(frame_hw, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')
    raise ImportError("SciPy is required for DCT (scipy.fft.dctn or scipy.fftpack.dct)")


def _compute_dct_radial_band_indices(h: int, w: int, k_bands: int) -> np.ndarray:
    """Band indices for 2D DCT frequency indices (u,v) with DC at (0,0).
    Radius is sqrt(u^2+v^2) normalized by max radius to (H-1,W-1)."""
    uu, vv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    r = np.sqrt(uu * uu + vv * vv)
    r_max = float(np.sqrt((h - 1) ** 2 + (w - 1) ** 2))
    r_norm = r / max(1e-12, r_max)
    edges = np.linspace(0.0, 1.0, num=k_bands + 1, dtype=np.float64)
    band = np.digitize(r_norm.ravel(), edges, right=True)
    band = np.clip(band, 1, k_bands) - 1
    return band.reshape(h, w)


# ------------------------- Feature primitives -------------------------

def _normalize_frame_to_float01(frame_hw3: np.ndarray) -> np.ndarray:
    """Return float32 image in [0,1]. Handles uint8 and float inputs robustly."""
    if frame_hw3.dtype == np.uint8:
        return frame_hw3.astype(np.float32) / 255.0
    fr = frame_hw3.astype(np.float32, copy=False)
    # If values look like [0,255] range but dtype is float, scale down
    m = float(np.nanmax(fr)) if fr.size else 0.0
    if m > 1.5:
        fr = fr / 255.0
    return fr

def _shannon_entropy_hist(vals: np.ndarray, bins: int = 128, eps: float = 1e-12) -> float:
    """Compute Shannon entropy (bits) using a histogram-based discrete estimate."""
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    # Change of base to bits (log2)
    ent = -np.sum(p * (np.log(p + eps) / np.log(2.0)))
    if not np.isfinite(ent):
        return 0.0
    return float(ent)

def _wavelet_subband_entropies_per_frame(frame_hw3: np.ndarray,
                                         wavelet: str = 'haar',
                                         level: int = 1,
                                         bins: int = 128,
                                         value_mode: str = 'abs',
                                         eps: float = 1e-12) -> np.ndarray:
    """Compute per-frame wavelet subband entropies over RGB channels and average.

    Returns a feature vector of length K = 1 + 3*level in order:
      [LL_L, LH_1, HL_1, HH_1, ..., LH_L, HL_L, HH_L]
    """
    if frame_hw3.ndim != 3 or frame_hw3.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H,W,3), got {frame_hw3.shape}")

    fr = _normalize_frame_to_float01(frame_hw3)

    # Prepare per-channel feature accumulation
    K = 1 + 3 * int(level)
    feats = np.zeros((3, K), dtype=np.float64)

    if not _HAS_PYWT:
        raise ImportError("PyWavelets is required for --feature_mode dwt_entropy (pip install PyWavelets)")

    for c in range(3):
        coeffs = pywt.wavedec2(fr[:, :, c], wavelet=wavelet, level=level, mode='periodization')
        # coeffs: [cA_L, (cH_L,cV_L,cD_L), ..., (cH_1,cV_1,cD_1)]
        idx = 0
        cA_L = coeffs[0]
        vals = cA_L
        if value_mode == 'abs':
            vals = np.abs(vals)
        elif value_mode == 'sq':
            vals = np.square(vals)
        feats[c, idx] = _shannon_entropy_hist(vals.ravel(), bins=bins, eps=eps)
        idx += 1
        for lev in range(1, level + 1):
            cH, cV, cD = coeffs[lev]
            for band in (cH, cV, cD):
                vals = band
                if value_mode == 'abs':
                    vals = np.abs(vals)
                elif value_mode == 'sq':
                    vals = np.square(vals)
                feats[c, idx] = _shannon_entropy_hist(vals.ravel(), bins=bins, eps=eps)
                idx += 1

    return feats.mean(axis=0)


# --------------------------- Core computation ---------------------------

def dct2_energy_bands_per_frame(frame_hw3: np.ndarray, band_idx_hw: np.ndarray, k_bands: int, eps: float = 1e-12) -> np.ndarray:
    """Compute mean energy per radial DCT band for a single RGB frame.

    - frame_hw3: HxWx3 uint8/float in [0,255] or [0,1]
    - band_idx_hw: precomputed integer indices [0..K-1] for each (y,x)
    - Returns: (K,) array with mean energy per band, averaged over 3 channels

    Energy per band is computed as the mean of DCT coefficient squared magnitude within that band.
    """
    if frame_hw3.ndim != 3 or frame_hw3.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H,W,3), got {frame_hw3.shape}")
    H, W, _ = frame_hw3.shape

    # Normalize to float32 [0,1]
    fr = _normalize_frame_to_float01(frame_hw3)

    bands_energy = np.zeros((3, k_bands), dtype=np.float64)
    band_counts = np.bincount(band_idx_hw.ravel(), minlength=k_bands).astype(np.float64)
    band_counts = np.maximum(band_counts, eps)

    for c in range(3):
        C = _dct2(fr[:, :, c])
        S = np.abs(C) ** 2
        sums = np.bincount(band_idx_hw.ravel(), weights=S.ravel(), minlength=k_bands).astype(np.float64)
        bands_energy[c] = sums / band_counts

    return bands_energy.mean(axis=0)


def dct2_energy_bands_per_video(vpath: str, k_bands: int, stride: int, max_frames: int) -> Optional[np.ndarray]:
    try:
        vr = VideoReader(vpath)
    except Exception:
        print(f"[WARN] Failed to open video: {vpath}")
        return None

    T = len(vr)
    idx = frame_indices(T, stride=stride, max_frames=max_frames)
    if not idx:
        return None

    try:
        batch = vr.get_batch(idx).asnumpy()  # (N,H,W,3)
    except Exception:
        print(f"[WARN] Failed to read frames: {vpath}")
        return None

    N, H, W, C = batch.shape
    band_idx_hw = _compute_dct_radial_band_indices(H, W, k_bands)

    acc = np.zeros(k_bands, dtype=np.float64)
    for n in range(N):
        acc += dct2_energy_bands_per_frame(batch[n], band_idx_hw, k_bands)
    return acc / float(N)


def wavelet_entropy_subbands_per_video(vpath: str,
                                       wavelet: str,
                                       level: int,
                                       bins: int,
                                       value_mode: str,
                                       stride: int,
                                       max_frames: int) -> Optional[np.ndarray]:
    try:
        vr = VideoReader(vpath)
    except Exception:
        print(f"[WARN] Failed to open video: {vpath}")
        return None

    T = len(vr)
    idx = frame_indices(T, stride=stride, max_frames=max_frames)
    if not idx:
        return None

    try:
        batch = vr.get_batch(idx).asnumpy()  # (N,H,W,3)
    except Exception:
        print(f"[WARN] Failed to read frames: {vpath}")
        return None

    N = batch.shape[0]
    K = 1 + 3 * int(level)
    acc = np.zeros(K, dtype=np.float64)
    for n in range(N):
        acc += _wavelet_subband_entropies_per_frame(batch[n], wavelet=wavelet, level=level, bins=bins, value_mode=value_mode)
    return acc / float(N)


@dataclass
class CategoryDCTBands:
    name: str
    n_videos: int
    mean_energy: List[float]
    std_energy: List[float]
    mean_proportions: List[float]


def aggregate_category_dct(category: str,
                           vpaths: List[str],
                           k_bands: int,
                           stride: int,
                           max_frames: int,
                           max_videos: int,
                           feature_mode: str = 'dct_energy',
                           wavelet: str = 'haar',
                           dwt_level: int = 1,
                           entropy_bins: int = 128,
                           entropy_value_mode: str = 'abs') -> CategoryDCTBands:
    if max_videos and max_videos > 0:
        vpaths = vpaths[:max_videos]

    per_video: List[np.ndarray] = []
    for i, vp in enumerate(vpaths):
        try:
            if feature_mode == 'dct_energy':
                e = dct2_energy_bands_per_video(vp, k_bands=k_bands, stride=stride, max_frames=max_frames)
            elif feature_mode == 'dwt_entropy':
                e = wavelet_entropy_subbands_per_video(vp,
                                                       wavelet=wavelet,
                                                       level=dwt_level,
                                                       bins=entropy_bins,
                                                       value_mode=entropy_value_mode,
                                                       stride=stride,
                                                       max_frames=max_frames)
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode}")
            if e is None:
                continue
            if not np.isfinite(e).all():
                continue
            per_video.append(e)
        except KeyboardInterrupt:
            raise
        except Exception:
            print(f"[WARN] Error processing {vp}")
            continue
        if (i + 1) % 20 == 0:
            print(f"  [{category}] processed {i+1}/{len(vpaths)} videos…")

    if not per_video:
        return CategoryDCTBands(name=category, n_videos=0,
                                mean_energy=[0.0]*k_bands, std_energy=[0.0]*k_bands,
                                mean_proportions=[0.0]*k_bands)

    M = np.stack(per_video, axis=0)  # (V,K)
    mean = M.mean(axis=0)
    std = M.std(axis=0, ddof=1) if len(per_video) > 1 else np.zeros(k_bands, dtype=np.float64)

    # Mean of per-video proportions
    props = []
    for v in per_video:
        s = float(np.sum(v))
        if s <= 0:
            props.append(np.zeros(k_bands, dtype=np.float64))
        else:
            props.append((v / s).astype(np.float64))
    mean_props = np.mean(np.stack(props, axis=0), axis=0)

    return CategoryDCTBands(name=category,
                            n_videos=len(per_video),
                            mean_energy=[float(x) for x in mean],
                            std_energy=[float(x) for x in std],
                            mean_proportions=[float(x) for x in mean_props])


# ------------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze 2D DCT band energies per corruption category.")
    parser.add_argument('--data_root', '--data-root', dest='data_root', type=str, required=True,
                        help='Root directory containing corruption subfolders (e.g., .../val_corruptions)')
    parser.add_argument('--extensions', type=str, nargs='+', default=['mp4'],
                        help='Video file extensions to include (case-insensitive, without dot).')
    parser.add_argument('--k_bands', '--bands', dest='k_bands', type=int, default=8, help='Number of radial frequency bands (>=1) [used when --feature_mode dct_energy]')
    parser.add_argument('--frame_stride', '--frame-stride', dest='frame_stride', type=int, default=4, help='Sample every Nth frame for efficiency')
    parser.add_argument('--max_frames_per_video', '--max-frames', dest='max_frames_per_video', type=int, default=0, help='Cap frames per video after stride; 0 means no cap')
    parser.add_argument('--max_videos_per_corruption', '--max-videos', dest='max_videos_per_corruption', type=int, default=0, help='Limit videos per corruption; 0 means all')
    parser.add_argument('--num_samples_per_corruption', '--num-samples-per-corruption', dest='num_samples_per_corruption', type=int, default=0,
                        help='If >0, overrides max_videos_per_corruption to process exactly this many videos per corruption')
    parser.add_argument('--total_samples', '--total-samples', dest='total_samples', type=int, default=0,
                        help='If >0, process at most this many videos total, distributed across corruptions')
    parser.add_argument('--save_csv', '--out-csv', dest='save_csv', type=str, default='', help='Path to save CSV summary')
    parser.add_argument('--save_json', '--out-json', dest='save_json', type=str, default='', help='Path to save JSON summary')
    parser.add_argument('--feature_mode', '--feature-mode', dest='feature_mode', type=str, default='dct_energy', choices=['dct_energy', 'dwt_entropy'],
                        help='Feature to compute per frame/video: DCT radial band energy or wavelet subband entropy')
    parser.add_argument('--wavelet', dest='wavelet', type=str, default='haar', help='Wavelet name for dwt_entropy (e.g., haar, db2, sym4)')
    parser.add_argument('--dwt_level', '--dwt-level', dest='dwt_level', type=int, default=1, help='Decomposition level for dwt_entropy (>=1)')
    parser.add_argument('--entropy_bins', '--entropy-bins', dest='entropy_bins', type=int, default=128, help='Histogram bins for entropy estimation in dwt_entropy')
    parser.add_argument('--entropy_value_mode', '--entropy-value-mode', dest='entropy_value_mode', type=str, default='abs', choices=['abs', 'raw', 'sq'],
                        help='Apply function to wavelet coefficients before histogram: abs, raw, or squared')

    args = parser.parse_args()

    # Check backends based on feature mode
    if args.feature_mode == 'dct_energy':
        if not _HAS_SCIPY_DCTN and _scipy_dct1 is None:
            print("[ERROR] SciPy is required for DCT (install scipy).", file=sys.stderr)
            sys.exit(1)
    elif args.feature_mode == 'dwt_entropy':
        if not _HAS_PYWT:
            print("[ERROR] PyWavelets is required for --feature_mode dwt_entropy (install PyWavelets).", file=sys.stderr)
            sys.exit(1)

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    corruptions = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d))]
    if not corruptions:
        print(f"[ERROR] No corruption subfolders found in: {data_root}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Corruption categories:")
    for c in corruptions:
        print("  -", c)

    exts = tuple(e.lower().lstrip('.') for e in args.extensions)

    results: List[CategoryDCTBands] = []

    # Determine output feature dimension K_out
    if args.feature_mode == 'dct_energy':
        K_out = int(args.k_bands)
    else:  # dwt_entropy
        K_out = 1 + 3 * int(max(1, args.dwt_level))

    # Determine per-category quotas if total_samples is specified
    quotas: Optional[Dict[str, int]] = None
    if args.total_samples and args.total_samples > 0:
        remaining = int(args.total_samples)
        remaining_cats = len(corruptions)
        quotas = {}
        for i, c in enumerate(corruptions):
            q = max(0, remaining // remaining_cats)
            if i < (args.total_samples % len(corruptions)):
                q += 1
            quotas[c] = q
            remaining -= q
            remaining_cats -= 1

    for c in corruptions:
        c_root = os.path.join(data_root, c)
        vpaths = find_video_files(c_root, exts)
        if not vpaths:
            print(f"[WARN] No videos found in {c_root} (extensions={exts})")
            results.append(CategoryDCTBands(name=c, n_videos=0,
                                            mean_energy=[0.0]*K_out,
                                            std_energy=[0.0]*K_out,
                                            mean_proportions=[0.0]*K_out))
            continue

        print(f"[INFO] {c}: found {len(vpaths)} videos")
        cap = args.max_videos_per_corruption
        if args.num_samples_per_corruption and args.num_samples_per_corruption > 0:
            cap = args.num_samples_per_corruption
        if quotas is not None:
            cap = quotas.get(c, 0)

        stats = aggregate_category_dct(
            category=c,
            vpaths=vpaths,
            k_bands=K_out,
            stride=args.frame_stride,
            max_frames=args.max_frames_per_video,
            max_videos=cap,
            feature_mode=args.feature_mode,
            wavelet=args.wavelet,
            dwt_level=args.dwt_level,
            entropy_bins=args.entropy_bins,
            entropy_value_mode=args.entropy_value_mode,
        )
        results.append(stats)

    # Print summary table
    K = K_out
    band_cols = ' '.join([f"B{i:02d}".rjust(12) for i in range(K)])
    prop_cols = ' '.join([f"pB{i:02d}".rjust(7) for i in range(K)])
    header = f"{'Corruption':<20} {'N':>5}  {band_cols}   {prop_cols}"
    if args.feature_mode == 'dct_energy':
        print("\n===== 2D DCT Band Energy Summary (mean ± std; proportions) =====")
    else:
        print("\n===== Wavelet Subband Entropy Summary (mean ± std; proportions) =====")
    print(header)
    print('-' * len(header))
    for s in results:
        m = s.mean_energy
        d = s.std_energy
        p = s.mean_proportions
        bands_str = ' '.join([f"{m[i]:>6.4f}±{d[i]:>5.4f}" for i in range(K)])
        props_str = ' '.join([f"{p[i]:>6.3f}" for i in range(K)])
        line = f"{s.name:<20} {s.n_videos:>5}  {bands_str}   {props_str}"
        print(line)

    # Save CSV if requested
    if args.save_csv:
        fieldnames = ['corruption', 'n_videos'] + \
                     [f'mean_B{i}' for i in range(K)] + [f'std_B{i}' for i in range(K)] + \
                     [f'prop_B{i}' for i in range(K)]
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in results:
                row = {'corruption': s.name, 'n_videos': s.n_videos}
                for i in range(K):
                    row[f'mean_B{i}'] = s.mean_energy[i]
                    row[f'std_B{i}'] = s.std_energy[i]
                for i in range(K):
                    row[f'prop_B{i}'] = s.mean_proportions[i]
                writer.writerow(row)
        print(f"[INFO] Saved CSV to {args.save_csv}")

    # Save JSON if requested
    if args.save_json:
        payload = []
        for s in results:
            payload.append({
                'corruption': s.name,
                'n_videos': s.n_videos,
                'mean': s.mean_energy,
                'std': s.std_energy,
                'proportions': s.mean_proportions,
            })
        with open(args.save_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved JSON to {args.save_json}")


if __name__ == '__main__':
    main()
