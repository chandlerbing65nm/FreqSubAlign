# SWaveletA – How to run main_tta_ucf.py and main_tta_ss2.py without CLI args

This repo provides two entry scripts that are already set up to be configured by editing variables inside the files (no --args on the command line):

- `Repo/SWaveletA/main_tta_ucf.py` (UCF101)
- `Repo/SWaveletA/main_tta_ss2.py` (Something-Something V2)

Below is a concise guide to which lines to edit and how to run.


## 1) Quick start

1) Open the script for your dataset:
- UCF101: `main_tta_ucf.py`
- Something-Something V2: `main_tta_ss2.py`

2) Edit the configuration block near the top of `if __name__ == '__main__':` to set the following keys (no CLI flags needed):
- `args.arch`
- `args.tta`
- `args.corruption_list`
- `args.dwt_align_enable`
- `args.dwt_align_adaptive_lambda`
- `args.dwt_align_3d`
- `args.dwt_align_levels`
- If `args.tta` is False (source-only), also set:
  - `args.evaluate_baselines`
  - `args.baseline`

3) Run the script directly:
```bash
# UCF101
python Repo/SWaveletA/main_tta_ucf.py

# Something-Something V2
python Repo/SWaveletA/main_tta_ss2.py
```

No additional command-line options are required; everything is controlled by the variables inside the script.


## 2) What to edit (by key)

All of the following are set in the `__main__` section of each script.

- `args.arch`
  - Choose model: `'videoswintransformer'` or `'tanet'`.
  - Both scripts default to `'videoswintransformer'` in-code; change as needed.

- `args.tta`
  - `True` = Test-Time Adaptation (ViTTA mode)
  - `False` = source-only (no adaptation; baseline evaluation)
  - When `True`, the script uses precomputed clean statistics (paths are set via `get_model_config()` and copied into `args.spatiotemp_*`).

- `args.corruption_list`
  - Supported values: `'mini'`, `'full'`, `'continual'`, `'random'`, `'continual_alternate'`.
  - Controls which corruption(s) will be evaluated and which list file(s) to load.

- `args.dwt_align_enable`
  - Enable Wavelet subband alignment regularization hooks.
  - When enabled, make sure `args.dwt_align_levels` matches the NPZ statistics file you are using (see below).

- `args.dwt_align_adaptive_lambda`
  - If `True`, uses adaptive per-band weighting (as supported in the codebase).

- `args.dwt_align_3d`
  - If `True`, uses 3D variant for alignment (primarily relevant for TANet). When `True` and `arch == 'tanet'`, the script selects a 3D NPZ stats file.

- `args.dwt_align_levels`
  - Pyramid depth for DWT alignment stats you intend to use (e.g., `1` or `2`).
  - Must match the `.npz` file used by the alignment hook.

- If `args.tta` is False (source-only / baseline mode):
  - `args.evaluate_baselines` should be `True`.
  - `args.baseline`: choose one of `source, shot, tent, dua, rem, norm, t3a` (both scripts include these; the UCF script comment includes all, SSv2 script comment mirrors the set). Default is `'shot'` in both scripts as provided.
  - Optional: `args.t3a_filter_k` (already set to 100 in both scripts).


## 3) Paths you may need to update

Both scripts centralize model and statistics paths in `get_model_config(arch, dataset, tta_mode)`.

- Model checkpoint: `config['model_path']`
- ViTTA clean statistics (used only when `args.tta` is True):
  - `config['spatiotemp_mean_clean_file']`
  - `config['spatiotemp_var_clean_file']`
- DWT subband stats for alignment (NPZ): `config['additional_args']['dwt_stats_npz_file']`

Additionally, `main_tta_ucf.py` contains small overrides for TANet to select different NPZ files when `args.dwt_align_3d is True` or when `args.dwt_align_levels == 2` with 2D alignment. Adjust those paths if your environment differs:
- `main_tta_ucf.py` lines around:
  - 173–178: 3D and L2 NPZ overrides for TANet.

Dataset directories and list files are constructed from `dataset_to_dir` and `args.corruption_list`:
- Videos are read from: `/scratch/project_465001897/datasets/{dataset_dir}/val_corruptions`
- List files per corruption: `/scratch/project_465001897/datasets/{dataset_dir}/list_video_perturbations/{corruption}.txt`
- Results are written to:
  - TTA: `/scratch/project_465001897/datasets/{dataset_dir}/results/corruptions/{args.arch}_{args.dataset}`
  - Source-only: `/scratch/project_465001897/datasets/{dataset_dir}/results/source/{args.arch}_{args.dataset}`

If your data/checkpoint paths differ, edit these fields accordingly.


## 4) Examples

- TTA on UCF101 with Video Swin, continual corruption, no DWT alignment:
```python
# In main_tta_ucf.py
args.arch = 'videoswintransformer'
args.dataset = 'ucf101'
args.tta = True
args.corruption_list = 'continual'
# DWT alignment disabled
args.dwt_align_enable = False
```
Run:
```bash
python Repo/SWaveletA/main_tta_ucf.py
```

- Source-only SHOT baseline on Something-Something V2 with Video Swin:
```python
# In main_tta_ss2.py
args.arch = 'videoswintransformer'
args.dataset = 'somethingv2'
args.tta = False
args.evaluate_baselines = True
args.baseline = 'shot'
args.corruption_list = 'full'
```
Run:
```bash
python Repo/SWaveletA/main_tta_ss2.py
```

- TTA with DWT alignment (2D, L1) on SSv2 (ensure NPZ path is valid):
```python
# In main_tta_ss2.py
args.tta = True
args.dwt_align_enable = True
args.dwt_align_adaptive_lambda = True   # optional
args.dwt_align_3d = False               # 2D alignment
args.dwt_align_levels = 1               # must match NPZ
```


## 5) Optional helpers

- Print the validation corruption order (no evaluation):
```python
# In either script
args.print_val_corrupt_order = True
```
Run the script; it will print the order and exit.

- Verbose configuration dump per corruption:
```python
args.verbose = True
```


## 6) Notes

- Batch size is set inside the script (`args.batch_size = 1` by default). Increase only if your GPU memory allows.
- For TTA, learning rate and selected blocks are set via `get_model_config(..., tta_mode=True)`. Adjust if you deviate from defaults.
- Result file naming includes a compact suffix with key settings (e.g., DWT flags, corruption list, batch size) handled automatically in each script.
