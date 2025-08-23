#!/usr/bin/env python3
"""
Analyze DWT subband energy by corruption category on the test set.

For each corruption subfolder under --data_root, this script:
- Iterates over videos (recursively, by extension)
- Samples frames (configurable stride)
- Applies 2D DWT (pywt.dwt2, default 'haar') to each frame/channel
- Computes mean squared energy per subband (LL, LH, HL, HH), normalized by subband size
- Aggregates per-video averages and then averages across videos in each corruption category

Outputs a readable table to stdout and optionally saves CSV/JSON.

Notes:
- This is a standalone analysis utility; it does NOT modify or depend on ViTTA model code paths.
- It uses Decord to read videos, consistent with data loading used elsewhere in the repo.

Example:
    python -m test.analyze_dwt_energy_by_corruption \
        --data_root /scratch/project_465001897/datasets/xxx/val_corruptions \
        --extensions mp4 avi \
        --frame_stride 4 \
        --max_videos_per_corruption 200 \
        --save_csv dwt_energy_summary.csv
"""

import argparse
import os
import sys
import json
import csv
import math
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pywt
import torch
import torch.nn.functional as F

try:
    import decord
    from decord import VideoReader
    # Use a valid bridge; 'native' works with .asnumpy()
    try:
        decord.bridge.set_bridge('native')
    except Exception:
        # If setting bridge fails, continue with default
        pass
except Exception as e:
    print("[ERROR] Decord is required for this script. Install via `pip install decord`.", file=sys.stderr)
    raise


@dataclass
class SubbandEnergy:
    LL: float = 0.0
    LH: float = 0.0
    HL: float = 0.0
    HH: float = 0.0

    def __iadd__(self, other: 'SubbandEnergy'):
        self.LL += other.LL
        self.LH += other.LH
        self.HL += other.HL
        self.HH += other.HH
        return self

    def __truediv__(self, scalar: float) -> 'SubbandEnergy':
        return SubbandEnergy(self.LL / scalar, self.LH / scalar, self.HL / scalar, self.HH / scalar)

    def total(self) -> float:
        return self.LL + self.LH + self.HL + self.HH

    def proportions(self) -> Dict[str, float]:
        t = self.total()
        if t <= 0:
            return {"LL": 0.0, "LH": 0.0, "HL": 0.0, "HH": 0.0}
        return {"LL": self.LL / t, "LH": self.LH / t, "HL": self.HL / t, "HH": self.HH / t}


@dataclass
class CategoryStats:
    name: str
    n_videos: int
    mean_energy: SubbandEnergy
    std_energy: SubbandEnergy
    mean_proportions: Dict[str, float]


@dataclass
class SubbandEnergy3D:
    """Energies for 3D DWT subbands (time-height-width): 8 combinations.

    Keys follow 'L'/'H' notation per axis order (t,h,w):
    {'LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH'}
    """
    LLL: float = 0.0
    LLH: float = 0.0
    LHL: float = 0.0
    LHH: float = 0.0
    HLL: float = 0.0
    HLH: float = 0.0
    HHL: float = 0.0
    HHH: float = 0.0

    def __iadd__(self, other: 'SubbandEnergy3D') -> 'SubbandEnergy3D':
        self.LLL += other.LLL; self.LLH += other.LLH; self.LHL += other.LHL; self.LHH += other.LHH
        self.HLL += other.HLL; self.HLH += other.HLH; self.HHL += other.HHL; self.HHH += other.HHH
        return self

    def __truediv__(self, scalar: float) -> 'SubbandEnergy3D':
        return SubbandEnergy3D(
            self.LLL / scalar, self.LLH / scalar, self.LHL / scalar, self.LHH / scalar,
            self.HLL / scalar, self.HLH / scalar, self.HHL / scalar, self.HHH / scalar,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            'LLL': self.LLL, 'LLH': self.LLH, 'LHL': self.LHL, 'LHH': self.LHH,
            'HLL': self.HLL, 'HLH': self.HLH, 'HHL': self.HHL, 'HHH': self.HHH,
        }

    def total(self) -> float:
        v = self.as_dict()
        return float(sum(v.values()))

    def proportions(self) -> Dict[str, float]:
        s = self.total()
        if s <= 0:
            return {k: 0.0 for k in self.as_dict().keys()}
        return {k: v / s for k, v in self.as_dict().items()}


@dataclass
class CategoryStats3D:
    name: str
    n_videos: int
    mean_energy: SubbandEnergy3D
    std_energy: SubbandEnergy3D
    mean_proportions: Dict[str, float]


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


def dwt2_energy_per_frame(frame: np.ndarray, wavelet: str, mode: str) -> SubbandEnergy:
    """
    Compute mean squared energy per subband for a single RGB frame.
    - frame: HxWx3 numpy array (uint8 or float)
    - Returns SubbandEnergy with each component being the mean of squared coeffs
      normalized by the number of coefficients in that subband.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H,W,3), got {frame.shape}")

    # Convert to float32 [0,1]
    if frame.dtype != np.float32:
        fr = frame.astype(np.float32) / 255.0
    else:
        fr = frame

    LL_sum = LH_sum = HL_sum = HH_sum = 0.0
    LL_cnt = LH_cnt = HL_cnt = HH_cnt = 0

    for c in range(3):
        coeffs2 = pywt.dwt2(fr[:, :, c], wavelet, mode=mode)
        cA, (cH, cV, cD) = coeffs2
        # Mean squared per subband (normalized by subband size)
        LL_sum += float(np.mean(np.square(cA)))
        LH_sum += float(np.mean(np.square(cH)))
        HL_sum += float(np.mean(np.square(cV)))
        HH_sum += float(np.mean(np.square(cD)))
        LL_cnt += 1
        LH_cnt += 1
        HL_cnt += 1
        HH_cnt += 1

    # Average over channels already baked into the mean calls; counts are equal
    return SubbandEnergy(LL=LL_sum / max(1, LL_cnt),
                         LH=LH_sum / max(1, LH_cnt),
                         HL=HL_sum / max(1, HL_cnt),
                         HH=HH_sum / max(1, HH_cnt))


def dwt2k_energy_per_frame(frame: np.ndarray, wavelet: str, mode: str, levels: int) -> SubbandEnergy:
    """
    K-level (levels>=1) 2D DWT energy per RGB frame using pywt.wavedec2.
    Aggregates detail energies across all levels; approximation is top-level cA.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H,W,3), got {frame.shape}")
    fr = frame.astype(np.float32) / 255.0 if frame.dtype != np.float32 else frame

    E_LL = E_LH = E_HL = E_HH = 0.0
    for c in range(3):
        coeffs = pywt.wavedec2(fr[:, :, c], wavelet=wavelet, mode=mode, level=max(1, int(levels)))
        cA_top, detail_levels = coeffs[0], coeffs[1:]
        # Top-level approximation energy
        E_LL += float(np.mean(np.square(cA_top)))
        # Aggregate details across levels
        for (cH, cV, cD) in detail_levels:
            E_LH += float(np.mean(np.square(cH)))
            E_HL += float(np.mean(np.square(cV)))
            E_HH += float(np.mean(np.square(cD)))
    # Average over 3 channels
    return SubbandEnergy(LL=E_LL / 3.0, LH=E_LH / 3.0, HL=E_HL / 3.0, HH=E_HH / 3.0)


def _build_haar_kernels(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build 4 separable 2x2 Haar kernels (LL, LH, HL, HH) stacked along out_channels with in_channels=1.
    Returns tensor of shape (4, 1, 2, 2).
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    h = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=dtype, device=device)
    g = torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=dtype, device=device)
    # Outer products: rows (H) x cols (W)
    kLL = torch.ger(h, h)  # low H, low W
    kLH = torch.ger(h, g)  # low H, high W  -> cH ("LH")
    kHL = torch.ger(g, h)  # high H, low W  -> cV ("HL")
    kHH = torch.ger(g, g)  # high H, high W
    K = torch.stack([kLL, kLH, kHL, kHH], dim=0).unsqueeze(1)  # (4,1,2,2)
    return K


def torch_batch_dwt2_energy(frames_bhwc: np.ndarray, device: torch.device) -> SubbandEnergy:
    """
    GPU-accelerated 2D Haar DWT for a batch of RGB frames using conv2d stride=2.
    frames_bhwc: numpy array (B, H, W, 3) in uint8 or float.
    Returns average SubbandEnergy over the batch.
    """
    if frames_bhwc.ndim != 4 or frames_bhwc.shape[-1] != 3:
        raise ValueError(f"Expected (B,H,W,3), got {frames_bhwc.shape}")

    B, H, W, C = frames_bhwc.shape
    x = torch.from_numpy(frames_bhwc).to(device)
    if x.dtype != torch.float32:
        x = x.float()
    x = x / 255.0  # (B,H,W,3) in [0,1]
    x = x.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)

    kernels = _build_haar_kernels(device=device, dtype=x.dtype)  # (4,1,2,2)
    C_in = x.shape[1]
    # For groups=C_in, weight must be (4*C_in, 1, 2, 2); repeat kernels per channel group
    weight = kernels.repeat(C_in, 1, 1, 1)  # (4*C_in,1,2,2)

    with torch.no_grad():
        y = F.conv2d(x, weight, bias=None, stride=2, padding=0, groups=C_in)  # (B,4*C,H/2,W/2)
    H2, W2 = y.shape[-2], y.shape[-1]
    y = y.view(B, C_in, 4, H2, W2)  # (B,C,4,H2,W2)

    # Mean squared energy per subband across (C,H2,W2) for each frame, then mean over B
    LL = y[:, :, 0].pow(2).mean(dim=(1, 2, 3))  # (B,)
    LH = y[:, :, 1].pow(2).mean(dim=(1, 2, 3))
    HL = y[:, :, 2].pow(2).mean(dim=(1, 2, 3))
    HH = y[:, :, 3].pow(2).mean(dim=(1, 2, 3))

    return SubbandEnergy(LL=float(LL.mean().item()),
                         LH=float(LH.mean().item()),
                         HL=float(HL.mean().item()),
                         HH=float(HH.mean().item()))


def dwt2_energy_per_video(vpath: str, wavelet: str, mode: str, stride: int, max_frames: int,
                          use_torch: bool, device: torch.device, batch_size_frames: int,
                          levels: int) -> SubbandEnergy:
    try:
        vr = VideoReader(vpath)
    except Exception:
        print(f"[WARN] Failed to open video: {vpath}")
        return SubbandEnergy()

    T = len(vr)
    idx = frame_indices(T, stride=stride, max_frames=max_frames)
    if not idx:
        return SubbandEnergy()

    # Torch path: process frames in chunks on GPU (only single-level Haar supported here)
    if use_torch and device.type == 'cuda' and (levels == 1) and (str(wavelet).lower() == 'haar'):
        total_frames = 0
        acc_LL = acc_LH = acc_HL = acc_HH = 0.0
        i = 0
        N = len(idx)
        while i < N:
            sub_idx = idx[i:i + max(1, batch_size_frames)]
            i += len(sub_idx)
            try:
                frames = vr.get_batch(sub_idx).asnumpy()  # (b,H,W,3)
            except Exception:
                print(f"[WARN] Failed to read frames chunk in {vpath}")
                continue
            e = torch_batch_dwt2_energy(frames, device=device)
            b = frames.shape[0]
            total_frames += b
            acc_LL += e.LL * b
            acc_LH += e.LH * b
            acc_HL += e.HL * b
            acc_HH += e.HH * b
        if total_frames == 0:
            return SubbandEnergy()
        return SubbandEnergy(LL=acc_LL / total_frames,
                              LH=acc_LH / total_frames,
                              HL=acc_HL / total_frames,
                              HH=acc_HH / total_frames)

    # CPU path: per-frame PyWavelets (supports multi-level and arbitrary wavelets)
    try:
        batch = vr.get_batch(idx).asnumpy()  # shape: (N, H, W, 3)
    except Exception:
        print(f"[WARN] Failed to read frames: {vpath}")
        return SubbandEnergy()

    acc = SubbandEnergy()
    for n in range(batch.shape[0]):
        if levels == 1:
            e = dwt2_energy_per_frame(batch[n], wavelet=wavelet, mode=mode)
        else:
            e = dwt2k_energy_per_frame(batch[n], wavelet=wavelet, mode=mode, levels=levels)
        acc += e
    return acc / float(batch.shape[0])


def aggregate_category(category: str, vpaths: List[str], wavelet: str, mode: str, stride: int, max_frames: int, max_videos: int,
                       use_torch: bool, device: torch.device, batch_size_frames: int, levels: int) -> CategoryStats:
    if max_videos and max_videos > 0:
        vpaths = vpaths[:max_videos]

    per_video = []
    for i, vp in enumerate(vpaths):
        try:
            e = dwt2_energy_per_video(vp, wavelet=wavelet, mode=mode, stride=stride, max_frames=max_frames,
                                      use_torch=use_torch, device=device, batch_size_frames=batch_size_frames,
                                      levels=levels)
            if e.total() == 0:
                continue
            per_video.append(e)
        except KeyboardInterrupt:
            raise
        except Exception:
            print(f"[WARN] Error processing {vp}:")
            traceback.print_exc()
            continue
        if (i + 1) % 20 == 0:
            print(f"  [{category}] processed {i+1}/{len(vpaths)} videos…")

    if not per_video:
        return CategoryStats(name=category, n_videos=0,
                             mean_energy=SubbandEnergy(), std_energy=SubbandEnergy(),
                             mean_proportions={"LL": 0.0, "LH": 0.0, "HL": 0.0, "HH": 0.0})

    # Compute mean and std across videos per subband
    LLs = np.array([x.LL for x in per_video], dtype=np.float64)
    LHs = np.array([x.LH for x in per_video], dtype=np.float64)
    HLs = np.array([x.HL for x in per_video], dtype=np.float64)
    HHs = np.array([x.HH for x in per_video], dtype=np.float64)

    mean = SubbandEnergy(LL=float(LLs.mean()), LH=float(LHs.mean()), HL=float(HLs.mean()), HH=float(HHs.mean()))
    std = SubbandEnergy(LL=float(LLs.std(ddof=1) if len(LLs) > 1 else 0.0),
                        LH=float(LHs.std(ddof=1) if len(LHs) > 1 else 0.0),
                        HL=float(HLs.std(ddof=1) if len(HLs) > 1 else 0.0),
                        HH=float(HHs.std(ddof=1) if len(HHs) > 1 else 0.0))

    # Mean of per-video proportions to avoid domination by long videos
    props = []
    for x in per_video:
        props.append(x.proportions())
    mean_props = {k: float(np.mean([p[k] for p in props])) for k in ["LL", "LH", "HL", "HH"]}

    return CategoryStats(name=category, n_videos=len(per_video), mean_energy=mean, std_energy=std, mean_proportions=mean_props)


def _tuple_key_to_str(k) -> str:
    """Convert wavedecn detail key (e.g., ('a','d','h')) to 'LLH' style; accept string keys too."""
    if isinstance(k, tuple):
        # In wavedecn, only 'a'/'d' appear; map 'a'->'L', 'd'->'H'
        return ''.join(('L' if (str(ch).lower() == 'a') else 'H') for ch in k)
    s = str(k)
    # Handle older pywt that may return strings like 'aad'
    return ''.join(('L' if (ch.lower() == 'a') else 'H') for ch in s)


def dwt3_energy_per_video(vpath: str, wavelet: str, mode: str, stride: int, max_frames: int, levels: int) -> SubbandEnergy3D:
    """
    K-level 3D DWT energy over a video using pywt.wavedecn on (T,H,W) volumes per RGB channel.
    Aggregates energies over channels and across all levels for each 3D subband key.
    """
    try:
        vr = VideoReader(vpath)
    except Exception:
        print(f"[WARN] Failed to open video: {vpath}")
        return SubbandEnergy3D()

    T = len(vr)
    idx = frame_indices(T, stride=stride, max_frames=max_frames)
    if not idx:
        return SubbandEnergy3D()
    try:
        batch = vr.get_batch(idx).asnumpy()  # (N,H,W,3)
    except Exception:
        print(f"[WARN] Failed to read frames: {vpath}")
        return SubbandEnergy3D()

    # Convert to float in [0,1]
    if batch.dtype != np.float32:
        batch = batch.astype(np.float32) / 255.0

    # Accumulate per-channel energies and average across channels
    agg = {k: 0.0 for k in ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']}
    for c in range(3):
        vol = batch[..., c]  # (N,H,W)
        coeffs = pywt.wavedecn(vol, wavelet=wavelet, mode=mode, level=max(1, int(levels)))
        cA_top, detail_levels = coeffs[0], coeffs[1:]
        # Approximation at top level is 'LLL'
        agg['LLL'] += float(np.mean(np.square(cA_top)))
        # Details: list from finest to coarsest? wavedecn returns [cA_n, details_n, ..., details_1]
        for d in detail_levels:
            for k, arr in d.items():
                key = _tuple_key_to_str(k)
                if key not in agg:
                    # Only consider L/H combinations (ignore if unexpected)
                    continue
                agg[key] += float(np.mean(np.square(arr)))

    # Average across channels
    for k in agg:
        agg[k] /= 3.0

    return SubbandEnergy3D(**agg)


def aggregate_category_3d(category: str, vpaths: List[str], wavelet: str, mode: str, stride: int, max_frames: int,
                          max_videos: int, levels: int) -> CategoryStats3D:
    if max_videos and max_videos > 0:
        vpaths = vpaths[:max_videos]
    if not vpaths:
        return CategoryStats3D(name=category, n_videos=0,
                               mean_energy=SubbandEnergy3D(), std_energy=SubbandEnergy3D(),
                               mean_proportions={k: 0.0 for k in ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']})

    per_video: List[SubbandEnergy3D] = []
    for vp in vpaths:
        e = dwt3_energy_per_video(vp, wavelet=wavelet, mode=mode, stride=stride, max_frames=max_frames, levels=levels)
        if e.total() == 0:
            continue
        per_video.append(e)

    if not per_video:
        return CategoryStats3D(name=category, n_videos=0,
                               mean_energy=SubbandEnergy3D(), std_energy=SubbandEnergy3D(),
                               mean_proportions={k: 0.0 for k in ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']})

    n = len(per_video)
    # Compute means
    keys = ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']
    means = {k: float(np.mean([getattr(e, k) for e in per_video])) for k in keys}
    stds = {k: float(np.std([getattr(e, k) for e in per_video])) for k in keys}
    mean_props = {k: 0.0 for k in keys}
    for e in per_video:
        p = e.proportions()
        for k in keys:
            mean_props[k] += p.get(k, 0.0)
    for k in keys:
        mean_props[k] /= n

    return CategoryStats3D(
        name=category,
        n_videos=n,
        mean_energy=SubbandEnergy3D(**means),
        std_energy=SubbandEnergy3D(**stds),
        mean_proportions=mean_props,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze DWT subband energies per corruption category.")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing corruption subfolders (e.g., .../val_corruptions)')
    parser.add_argument('--extensions', type=str, nargs='+', default=['mp4'],
                        help='Video file extensions to include (case-insensitive, without dot).')
    parser.add_argument('--wavelet', type=str, default='haar', help='Wavelet name for DWT (e.g., haar, db2, sym4)')
    parser.add_argument('--mode', type=str, default='smooth', help='PyWavelets signal extension mode (e.g., smooth, reflect, periodization)')
    parser.add_argument('--frame_stride', type=int, default=4, help='Sample every Nth frame for efficiency')
    parser.add_argument('--max_frames_per_video', type=int, default=0, help='Cap frames per video after stride; 0 means no cap')
    parser.add_argument('--max_videos_per_corruption', type=int, default=0, help='Limit videos per corruption; 0 means all')
    parser.add_argument('--num_samples_per_corruption', type=int, default=0,
                        help='If >0, overrides max_videos_per_corruption to process exactly this many videos per corruption')
    parser.add_argument('--total_samples', type=int, default=0,
                        help='If >0, process at most this many videos total, distributed across corruptions')
    parser.add_argument('--save_csv', type=str, default='', help='Path to save CSV summary')
    parser.add_argument('--save_json', type=str, default='', help='Path to save JSON summary')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for torch DWT (cuda|cpu)')
    parser.add_argument('--use_torch_dwt', type=lambda v: str(v).lower() in ('1','true','yes','y'),
                        default=torch.cuda.is_available(), help='Use GPU-accelerated torch DWT when available')
    parser.add_argument('--batch_size_frames', type=int, default=64, help='Frames per GPU chunk')
    parser.add_argument('--use_3d_dwt', action='store_true', help='Compute energies on 3D DWT subbands over (T,H,W) volumes')
    parser.add_argument('--levels', type=int, default=1, help='K-level DWT decomposition depth (>=1) for 2D/3D')

    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    # Discover corruption categories as immediate subdirectories
    corruptions = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d))]
    if not corruptions:
        print(f"[ERROR] No corruption subfolders found in: {data_root}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Corruption categories:")
    for c in corruptions:
        print("  -", c)

    exts = tuple(e.lower().lstrip('.') for e in args.extensions)

    results_2d: List[CategoryStats] = []
    results_3d: List[CategoryStats3D] = []
    device = torch.device(args.device)
    use_torch = bool(args.use_torch_dwt) and (device.type == 'cuda') and (not args.use_3d_dwt) and (args.levels == 1)

    # Determine per-category quotas if total_samples is specified
    quotas: Optional[Dict[str, int]] = None
    if args.total_samples and args.total_samples > 0:
        remaining = int(args.total_samples)
        remaining_cats = len(corruptions)
        quotas = {}
        for i, c in enumerate(corruptions):
            # Distribute as evenly as possible
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
            if args.use_3d_dwt:
                results_3d.append(CategoryStats3D(name=c, n_videos=0,
                                                  mean_energy=SubbandEnergy3D(), std_energy=SubbandEnergy3D(),
                                                  mean_proportions={k: 0.0 for k in ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']}))
            else:
                results_2d.append(CategoryStats(name=c, n_videos=0,
                                                mean_energy=SubbandEnergy(), std_energy=SubbandEnergy(),
                                                mean_proportions={"LL": 0.0, "LH": 0.0, "HL": 0.0, "HH": 0.0}))
            continue

        print(f"[INFO] {c}: found {len(vpaths)} videos")
        # Resolve video cap for this category
        cap = args.max_videos_per_corruption
        if args.num_samples_per_corruption and args.num_samples_per_corruption > 0:
            cap = args.num_samples_per_corruption
        if quotas is not None:
            cap = quotas.get(c, 0)

        if args.use_3d_dwt:
            stats3d = aggregate_category_3d(
                category=c,
                vpaths=vpaths,
                wavelet=args.wavelet,
                mode=args.mode,
                stride=args.frame_stride,
                max_frames=args.max_frames_per_video,
                max_videos=cap,
                levels=args.levels,
            )
            results_3d.append(stats3d)
        else:
            stats2d = aggregate_category(
                category=c,
                vpaths=vpaths,
                wavelet=args.wavelet,
                mode=args.mode,
                stride=args.frame_stride,
                max_frames=args.max_frames_per_video,
                max_videos=cap,
                use_torch=use_torch,
                device=device,
                batch_size_frames=args.batch_size_frames,
                levels=args.levels,
            )
            results_2d.append(stats2d)

    # Print summary table
    print("\n===== DWT Subband Energy Summary (mean ± std; proportions) =====")
    if args.use_3d_dwt:
        header = (
            f"{'Corruption':<20} {'N':>5}  "
            f"{'LLL':>12} {'LLH':>12} {'LHL':>12} {'LHH':>12} {'HLL':>12} {'HLH':>12} {'HHL':>12} {'HHH':>12}   "
            f"{'pLLL':>7} {'pLLH':>7} {'pLHL':>7} {'pLHH':>7} {'pHLL':>7} {'pHLH':>7} {'pHHL':>7} {'pHHH':>7}"
        )
        print(header)
        print('-' * len(header))
        for s in results_3d:
            m = s.mean_energy; d = s.std_energy; p = s.mean_proportions
            line = (
                f"{s.name:<20} {s.n_videos:>5}  "
                f"{m.LLL:>6.4f}±{d.LLL:>5.4f} {m.LLH:>6.4f}±{d.LLH:>5.4f} {m.LHL:>6.4f}±{d.LHL:>5.4f} {m.LHH:>6.4f}±{d.LHH:>5.4f} "
                f"{m.HLL:>6.4f}±{d.HLL:>5.4f} {m.HLH:>6.4f}±{d.HLH:>5.4f} {m.HHL:>6.4f}±{d.HHL:>5.4f} {m.HHH:>6.4f}±{d.HHH:>5.4f}   "
                f"{p['LLL']:>6.3f} {p['LLH']:>6.3f} {p['LHL']:>6.3f} {p['LHH']:>6.3f} {p['HLL']:>6.3f} {p['HLH']:>6.3f} {p['HHL']:>6.3f} {p['HHH']:>6.3f}"
            )
            print(line)
    else:
        header = f"{'Corruption':<20} {'N':>5}  {'LL':>12} {'LH':>12} {'HL':>12} {'HH':>12}   {'pLL':>7} {'pLH':>7} {'pHL':>7} {'pHH':>7}"
        print(header)
        print('-' * len(header))
        for s in results_2d:
            m = s.mean_energy; d = s.std_energy; p = s.mean_proportions
            line = (
                f"{s.name:<20} {s.n_videos:>5}  "
                f"{m.LL:>6.4f}±{d.LL:>5.4f} {m.LH:>6.4f}±{d.LH:>5.4f} "
                f"{m.HL:>6.4f}±{d.HL:>5.4f} {m.HH:>6.4f}±{d.HH:>5.4f}   "
                f"{p['LL']:>6.3f} {p['LH']:>6.3f} {p['HL']:>6.3f} {p['HH']:>6.3f}"
            )
            print(line)

    # Save CSV if requested
    if args.save_csv:
        if args.use_3d_dwt:
            fieldnames = [
                'corruption', 'n_videos',
                'mean_LLL','std_LLL','mean_LLH','std_LLH','mean_LHL','std_LHL','mean_LHH','std_LHH',
                'mean_HLL','std_HLL','mean_HLH','std_HLH','mean_HHL','std_HHL','mean_HHH','std_HHH',
                'prop_LLL','prop_LLH','prop_LHL','prop_LHH','prop_HLL','prop_HLH','prop_HHL','prop_HHH'
            ]
            with open(args.save_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in results_3d:
                    row = {
                        'corruption': s.name,
                        'n_videos': s.n_videos,
                        'mean_LLL': s.mean_energy.LLL, 'std_LLL': s.std_energy.LLL,
                        'mean_LLH': s.mean_energy.LLH, 'std_LLH': s.std_energy.LLH,
                        'mean_LHL': s.mean_energy.LHL, 'std_LHL': s.std_energy.LHL,
                        'mean_LHH': s.mean_energy.LHH, 'std_LHH': s.std_energy.LHH,
                        'mean_HLL': s.mean_energy.HLL, 'std_HLL': s.std_energy.HLL,
                        'mean_HLH': s.mean_energy.HLH, 'std_HLH': s.std_energy.HLH,
                        'mean_HHL': s.mean_energy.HHL, 'std_HHL': s.std_energy.HHL,
                        'mean_HHH': s.mean_energy.HHH, 'std_HHH': s.std_energy.HHH,
                        'prop_LLL': s.mean_proportions['LLL'], 'prop_LLH': s.mean_proportions['LLH'],
                        'prop_LHL': s.mean_proportions['LHL'], 'prop_LHH': s.mean_proportions['LHH'],
                        'prop_HLL': s.mean_proportions['HLL'], 'prop_HLH': s.mean_proportions['HLH'],
                        'prop_HHL': s.mean_proportions['HHL'], 'prop_HHH': s.mean_proportions['HHH'],
                    }
                    writer.writerow(row)
        else:
            fieldnames = [
                'corruption', 'n_videos',
                'mean_LL', 'std_LL', 'mean_LH', 'std_LH', 'mean_HL', 'std_HL', 'mean_HH', 'std_HH',
                'prop_LL', 'prop_LH', 'prop_HL', 'prop_HH'
            ]
            with open(args.save_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in results_2d:
                    row = {
                        'corruption': s.name,
                        'n_videos': s.n_videos,
                        'mean_LL': s.mean_energy.LL,
                        'std_LL': s.std_energy.LL,
                        'mean_LH': s.mean_energy.LH,
                        'std_LH': s.std_energy.LH,
                        'mean_HL': s.mean_energy.HL,
                        'std_HL': s.std_energy.HL,
                        'mean_HH': s.mean_energy.HH,
                        'std_HH': s.std_energy.HH,
                        'prop_LL': s.mean_proportions['LL'],
                        'prop_LH': s.mean_proportions['LH'],
                        'prop_HL': s.mean_proportions['HL'],
                        'prop_HH': s.mean_proportions['HH'],
                    }
                    writer.writerow(row)
        print(f"[INFO] Saved CSV to {args.save_csv}")

    # Save JSON if requested
    if args.save_json:
        payload = []
        if args.use_3d_dwt:
            for s in results_3d:
                payload.append({
                    'corruption': s.name,
                    'n_videos': s.n_videos,
                    'mean': asdict(s.mean_energy),
                    'std': asdict(s.std_energy),
                    'proportions': s.mean_proportions,
                })
        else:
            for s in results_2d:
                payload.append({
                    'corruption': s.name,
                    'n_videos': s.n_videos,
                    'mean': asdict(s.mean_energy),
                    'std': asdict(s.std_energy),
                    'proportions': s.mean_proportions,
                })
        with open(args.save_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved JSON to {args.save_json}")


if __name__ == '__main__':
    main()
