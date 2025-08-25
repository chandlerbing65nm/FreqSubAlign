import os
import torch
import random
import numpy as np
from utils.opts import get_opts
from utils.utils_ import get_writer_to_all_result
from corpus.main_eval import eval
from config import device

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Enable memory optimization
torch.cuda.empty_cache()


def get_model_config(arch, dataset='somethingv2', tta_mode=True):
    """Get model configuration based on architecture and dataset.
    
    Args:
        arch: Model architecture ('videoswintransformer' or 'tanet')
        dataset: Dataset name ('somethingv2', 'ucf101', or 'uffia')
        tta_mode: Whether to use TTA (Test-Time Adaptation) or source-only evaluation
    """
    config = {
        'model_path': '',
        'spatiotemp_mean_clean_file': '',
        'spatiotemp_var_clean_file': '',
        'additional_args': {}
    }
    
    # Common parameters for both architectures
    common_args = {
        'frame_uniform': True,
        'frame_interval': 2,
    }
    
    if arch == 'videoswintransformer':
        if dataset == 'ucf101':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ucf/model_swin/swin_ucf_base_patch244_window877_pretrain_kinetics400_30epoch_lr3e-5.pth',
                'additional_args': {
                    **common_args,
                    'clip_length': 16,
                    'num_clips': 1,
                    'patch_size': (2, 4, 4),
                    'window_size': (8, 7, 7),
                    'scale_size': 224,
                }
            })
            if tta_mode:
                config.update({
                    'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_swin/list_spatiotemp_mean_20221004_192722.npy',
                    'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_swin/list_spatiotemp_var_20221004_192722.npy',
                    'additional_args': {
                        **config['additional_args'],
                        'chosen_blocks': ['module.backbone.layers.2', 'module.backbone.layers.3', 'module.backbone.norm'],
                        'lr': 1e-5,
                        'lambda_pred_consis': 0.05,
                        'momentum_mvg': 0.05,
                    }
                })

    elif arch == 'tanet':
        common_args.update({
            'sample_style': 'uniform-1',
            'scale_size': 256,
        })
        if dataset == 'ucf101':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar',
                'additional_args': {
                    **common_args,
                    'clip_length': 16,
                }
            })
            if tta_mode:
                config.update({
                    'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20220908_235138.npy',
                    'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_var_20220908_235138.npy',
                    'additional_args': {
                        **config['additional_args'],
                        'lr': 5e-5
                    }
                })
    
    return config

if __name__ == '__main__':
    # Parse command line arguments
    args = get_opts()
    
    # Set seed for reproducibility
    set_seed(142)
    
    # Choose model architecture and dataset
    args.arch = 'tanet'  # videoswintransformer, tanet
    args.dataset = 'ucf101'  # somethingv2, ucf101, uffia

    # Map dataset names to directory names
    dataset_to_dir = {
        'ucf101': 'ucf',
        'uffia': 'uffia', 
        'somethingv2': 'ss2'
    }
    dataset_dir = dataset_to_dir.get(args.dataset, args.dataset)

    # Choose evaluation mode (TTA or source-only)
    args.tta = True  # Set to False for source-only evaluation
    
    # Get model configuration based on architecture and dataset
    model_config = get_model_config(args.arch, args.dataset, tta_mode=args.tta)
    
    # Set model paths and parameters
    args.model_path = model_config['model_path']
    
    # Set additional arguments for the selected architecture
    for key, value in model_config['additional_args'].items():
        setattr(args, key, value)

    # Critical arguments
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.scale_size = 256 # 256 for TANet, 224 for VideoSwin
    # args.crop_size = 256
    args.input_size = 224

    # Default arguments
    args.gpus = [0]
    args.video_data_dir = f'/scratch/project_465001897/datasets/{dataset_dir}/val_corruptions'
    # args.vid_format = '.mp4'  # Only for somethingv2

    args.n_augmented_views = 2
    args.if_sample_tta_aug_views = True
    args.batch_size = 1
    args.n_epoch_adapat = 1

    # ========================= New Arguments ==========================
    args.corruption_list = 'full' # mini, full, continual, continual_mini
    # args.dwt_preprocessing = True
    # args.dwt_component = 'LL'
    # args.dwt_levels = 1

    # # DWT subband alignment hook
    # args.dwt_align_enable = True
    # args.dwt_align_levels = 1  # must match the NPZ (L1)
    # args.dwt_stats_npz_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet_dwt/dwt_subband_stats_L1_20250825_134144.npz'

    # # Choose alignment weights (example: LL only)
    # args.dwt_align_lambda_ll = 1.0
    # args.dwt_align_lambda_lh = 1.0
    # args.dwt_align_lambda_hl = 1.0
    # args.dwt_align_lambda_hh = 1.0

    # ============================================================================================

    # Set TTA-specific parameters if in TTA mode
    if args.tta:
        args.spatiotemp_mean_clean_file = model_config['spatiotemp_mean_clean_file']
        args.spatiotemp_var_clean_file = model_config['spatiotemp_var_clean_file']
        print(f"Multi-epoch TTA enabled: Using {args.n_epoch_adapat} adaptation epochs")

        suffix = f"adaptepoch={args.n_epoch_adapat}"
        suffix += f"_views{args.n_augmented_views}"
    else:
        # Source-only evaluation parameters
        args.evaluate_baselines = True
        args.baseline = 'rem' # baseline, shot, tent, dua, rem
        
        suffix = f'baseline={args.baseline}'

    # Append preprocessing and alignment settings
    if getattr(args, 'dwt_preprocessing', False):
        suffix += f"_dwt{args.dwt_component}-L{args.dwt_levels}"
    # DWT subband alignment hook settings (for reproducibility)
    if getattr(args, 'dwt_align_enable', False):
        suffix += f"_dwtAlign-L{getattr(args, 'dwt_align_levels', 1)}"
        # Compact lambda encoding: include only lambdas > 0 to keep suffix short
        lam_ll = getattr(args, 'dwt_align_lambda_ll', 1.0)
        lam_lh = getattr(args, 'dwt_align_lambda_lh', 1.0)
        lam_hl = getattr(args, 'dwt_align_lambda_hl', 1.0)
        lam_hh = getattr(args, 'dwt_align_lambda_hh', 1.0)
        parts = []
        if lam_ll > 0: parts.append(f"LL{lam_ll}")
        if lam_lh > 0: parts.append(f"LH{lam_lh}")
        if lam_hl > 0: parts.append(f"HL{lam_hl}")
        if lam_hh > 0: parts.append(f"HH{lam_hh}")
        if parts:
            suffix += "_" + "+".join(parts)
    if getattr(args, 'update_only_bn_affine', False):
        suffix += "_bnaffine"
    suffix += f"_corruption={args.corruption_list}"
    suffix += f"_bs{args.batch_size}"
    args.result_suffix = suffix

    # Set up corruption types to evaluate
    if getattr(args, 'corruption_list', 'full') == 'mini':
        corruptions = [
            'gauss_mini', 'pepper_mini', 'salt_mini','shot_mini',
            'zoom_mini', 'impulse_mini', 'defocus_mini', 'motion_mini',
            'jpeg_mini', 'contrast_mini', 'rain_mini', 'h265_abr_mini',
        ]
    elif getattr(args, 'corruption_list', 'full') == 'full':
        corruptions = [
            'gauss', 'pepper', 'salt', 'shot',
            'zoom', 'impulse', 'defocus', 'motion',
            'jpeg', 'contrast', 'rain', 'h265_abr',
        ]
    elif getattr(args, 'corruption_list', 'full') == 'continual':
        corruptions = [
            'continual',
        ]
    elif getattr(args, 'corruption_list', 'full') == 'continual_mini':
        corruptions = [
            'continual_mini',
        ]
    
    # Set up result directory based on evaluation mode
    if args.tta:
        parent_result_dir = f'/scratch/project_465001897/datasets/{dataset_dir}/results/corruptions/{args.arch}_{args.dataset}'
        result_prefix = 'tta_'
    else:
        parent_result_dir = f'/scratch/project_465001897/datasets/{dataset_dir}/results/source/{args.arch}_{args.dataset}'
        result_prefix = 'source_'
    
    # Create parent results directory
    os.makedirs(parent_result_dir, exist_ok=True)
    
    # Create a single results file for all corruptions
    f_write = get_writer_to_all_result(args, custom_path=parent_result_dir)
    f_write.write('Evaluation Results (TTA mode)' if args.tta else 'Source-only Evaluation Results')
    f_write.write('\n#############################\n')

    # Evaluate on each corruption type
    for corr_id, args.corruptions in enumerate(corruptions):
        print(f'#### Starting Evaluation for ::: {args.corruptions} corruption ####')
        
        # Set up file paths
        args.val_vid_list = f'/scratch/project_465001897/datasets/{dataset_dir}/list_video_perturbations/{args.corruptions}.txt'
        args.result_dir = os.path.join(parent_result_dir, f'{result_prefix}{args.corruptions}')
        
        # Print verbose arguments for each corruption if verbose is enabled
        if args.verbose:
            print(f'\n=== Arguments for {args.corruptions} corruption ===')
            for arg in dir(args):
                if arg[0] != '_':
                    print(f'{arg}: {getattr(args, arg)}')
            print('=' * 50)
        
        # Clear GPU memory before each corruption
        torch.cuda.empty_cache()
        
        # Run evaluation
        epoch_result_list, _ = eval(args=args)

        # Write results
        f_write.write(f'\n{args.corruptions}:\n')
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in epoch_result_list]) + '\n')
        f_write.flush()

    f_write.close()
    print("Evaluation completed successfully.")