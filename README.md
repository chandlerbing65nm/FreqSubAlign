# FreqSubAlign — Frequency Subband Alignment for Video Test-Time Adaptation

FreqSubAlign (Frequency Subband Alignment) provides a simple, script-driven workflow to run Video Test-Time Adaptation (ViTTA) with frequency-subband regularization on corrupted video benchmarks.

This repository is configured to be edited in-script (no CLI flags required). The primary entry point for UCF101 is:

- `Repo/FreqSubAlign/main_tta_ucf.py`


## Quick Start (UCF101)

1) Open `Repo/FreqSubAlign/main_tta_ucf.py` and edit the small config block under `if __name__ == '__main__':`
   - `args.arch`: `videoswintransformer` or `tanet`
   - `args.tta`: `True` (ViTTA) or `False` (source-only/baselines)
   - `args.corruption_list`: `mini`, `full`, `continual`, `random`, or `continual_alternate`
   - Subband alignment: set `args.dwt_align_enable = True` to enable Frequency Subband Alignment
     - 2D/3D toggle: `args.dwt_align_3d`
     - Levels: `args.dwt_align_levels` (typically `1`; must match NPZ stats)
     - Transform: `args.subband_transform` in `{dwt, fft, dct}` (DWT supports `args.dwt_wavelet`)

2) Verify/adjust paths (inside the same script):
   - Model checkpoint: set by `get_model_config(arch, dataset, tta_mode)`
   - Clean stats for ViTTA: `spatiotemp_mean_clean_file` and `spatiotemp_var_clean_file`
   - Subband stats NPZ: `args.dwt_stats_npz_file` (auto-picked for common setups; adjust if paths differ)
   - Data roots and list files are constructed from `dataset_to_dir` and `args.corruption_list`

3) Run:
```bash
python Repo/FreqSubAlign/main_tta_ucf.py
```


## What It Does

- When `args.tta = True`, runs ViTTA with Frequency Subband Alignment hooks that match target feature subband statistics to clean (source) statistics.
- When `args.tta = False`, evaluates source-only and baselines (e.g., `norm`, `shot`, `tent`, `t3a`, `dua`).
- Results are written under:
  - TTA: `/scratch/project_465001897/datasets/{dataset_dir}/results/corruptions/{args.arch}_{args.dataset}`
  - Source-only: `/scratch/project_465001897/datasets/{dataset_dir}/results/source/{args.arch}_{args.dataset}`


## Minimal knobs to check

- `args.arch`/`args.dataset` (defaults: `tanet` + `ucf101` in-script)
- `args.tta` (True for adaptation; False for baselines)
- `args.dwt_align_enable` and `args.dwt_stats_npz_file` when using subband alignment
- GPU and batch: `args.gpus = [0]`, `args.batch_size = 1`


## Notes

- Requirements: see `Repo/FreqSubAlign/requirements.txt`
- To preview the corruption order without evaluating, set `args.print_val_corrupt_order = True` and run the script.
- Everything is controlled in `main_tta_ucf.py`; there are no mandatory CLI flags.
