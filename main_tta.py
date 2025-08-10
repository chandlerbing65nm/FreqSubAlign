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
                    'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_swin/list_spatiot_var_20221004_192722.npy',
                    'additional_args': {
                        **config['additional_args'],
                        'chosen_blocks': ['module.backbone.layers.2', 'module.backbone.layers.3', 'module.backbone.norm'],
                        'lr': 1e-5,
                        'lambda_pred_consis': 0.05,
                        'momentum_mvg': 0.05,
                    }
                })
                
        elif dataset == 'somethingv2':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ss2/model_swin/swin_base_patch244_window1677_sthv2.pth',
                'additional_args': {
                    **common_args,
                    'clip_length': 16,
                    'num_clips': 1,
                    'patch_size': (2, 4, 4),
                    'window_size': (16, 7, 7),
                    'scale_size': 224,
                }
            })
            if tta_mode:
                config.update({
                    'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ss2/source_statistics_swin/list_spatiotemp_mean_20250603_172215.npy',
                    'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ss2/source_statistics_swin/list_spatiotemp_var_20250603_172215.npy',
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
        
        if dataset == 'somethingv2':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ss2/model_tanet/TR50_S2_256_8x3x2.pth.tar',
                'additional_args': {
                    **common_args,
                    'clip_length': 8,
                }
            })
            if tta_mode:
                config.update({
                    'spatiotemp_mean_clean_file': '',
                    'spatiotemp_var_clean_file': '',
                    'additional_args': {
                        **config['additional_args'],
                        'lr': 1e-5
                    }
                })
                
        elif dataset == 'ucf101':
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
                
        elif dataset == 'uffia':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/uffia/model_tanet/20250805_221622_uffia_rgb_model_best.pth.tar',
                'additional_args': {
                    **common_args,
                    'tsn_style': True,
                    'input_size': 224,
                    'clip_length': 8,
                }
            })
            if tta_mode:
                config.update({
                    'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/uffia/source_statistics_tanet/list_spatiotemp_mean_20250805_200521.npy',
                    'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/uffia/source_statistics_tanet/list_spatiotemp_var_20250805_200521.npy',
                    'additional_args': {
                        **config['additional_args'],
                        'lr': 1e-6,
                        'lambda_pred_consis': 0.1,
                        'momentum_mvg': 0.1,
                        'momentum_bns': 0.1,
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
    args.scale_size = 256
    # args.crop_size = 224
    args.input_size = 224

    # Default arguments
    args.gpus = [0]
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/val_corruptions'
    # args.vid_format = '.mp4'  # Only for somethingv2

    # args.use_cgo = True
    # args.cgo_confidence_threshold = 0.4  # Moderate threshold
    # args.cgo_confidence_metric = 'max_softmax'
    # args.cgo_enable_logging = True
    # args.cgo_adaptive = False  # Use fixed threshold for stability
    
    # # CIMO (Inverted Meta-Optimizer) arguments - learn from hard samples
    # args.use_cimo = False  # Set to True to enable CIMO
    # args.confidence_metric = 'max_softmax'
    # args.cimo_confidence_threshold = 0.3
    # args.cimo_min_lr_scale = 0.1
    # args.cimo_max_lr_scale = 3.0
    # args.cimo_confidence_power = 6.0
    # args.cimo_enable_momentum_correction = True
    # args.cimo_adaptive = False
    
    # EMA Teacher arguments for inverted self-distillation
    args.use_ema_teacher = True  # Enable EMA teacher-based distillation
    args.ema_momentum = 0.9995  # EMA momentum for teacher updates
    args.ema_temperature = 6.0  # Base temperature for soft targets
    args.ema_adaptive_temp = False  # Disable adaptive temperature by default
    args.ema_min_temp = 0.1  # Minimum temperature (unused when adaptive off)
    args.ema_max_temp = 8.0  # Maximum temperature (unused when adaptive off)
    args.ema_temp_alpha = 2.0  # Temperature adaptation scaling (unused when adaptive off)
    args.lambda_distill = 0.2  # Lower weight for distillation loss by default
    # Distillation control arguments
    args.distill_conf_power = 1.5
    args.ema_distill_conf_thresh = 1.0
    args.ema_distill_warmup_steps = 0

    args.n_augmented_views = 4
    args.if_sample_tta_aug_views = True
    args.batch_size = 1  # Default to 1 for TTA, can be overridden
    
    # Set TTA-specific parameters if in TTA mode
    if args.tta:
        args.spatiotemp_mean_clean_file = model_config['spatiotemp_mean_clean_file']
        args.spatiotemp_var_clean_file = model_config['spatiotemp_var_clean_file']
        args.n_epoch_adapat = 2
        print(f"Multi-epoch TTA enabled: Using {args.n_epoch_adapat} adaptation epochs")

        args.include_ce_in_consistency = False
        suffix = f"celoss={args.include_ce_in_consistency}_adaptepoch={args.n_epoch_adapat}"
        
        # Add EMA teacher parameters
        if hasattr(args, 'use_ema_teacher') and args.use_ema_teacher:
            suffix += f"_ema{args.ema_momentum}_temp{args.ema_temperature}"
            if args.ema_adaptive_temp:
                suffix += f"_adapttemp"
            suffix += f"_distill{args.lambda_distill}"
        suffix += f"_views{args.n_augmented_views}_bs{args.batch_size}"
        
        args.result_suffix = suffix
    else:
        # Source-only evaluation parameters
        args.evaluate_baselines = True
        args.baseline = 'source'
        args.result_suffix=f'tta={args.tta}_evalbaseline={args.evaluate_baselines}_baseline={args.baseline}'

    # Set up corruption types to evaluate
    # corruptions = ['random_mini', 'gauss_mini']
    corruptions = ['gauss_mini', 'pepper_mini', 'salt_mini','shot_mini',
                'zoom_mini', 'impulse_mini', 'defocus_mini', 'motion_mini',
                'jpeg_mini', 'contrast_mini', 'rain_mini', 'h265_abr_mini',
                'random_mini'  
                ]

    # Set up result directory based on evaluation mode
    if args.tta:
        parent_result_dir = f'/scratch/project_465001897/datasets/ucf/results/corruptions/{args.arch}_{args.dataset}'
        result_prefix = 'tta_'
    else:
        parent_result_dir = f'/scratch/project_465001897/datasets/ucf/results/source/{args.arch}_{args.dataset}'
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
        args.val_vid_list = f'/scratch/project_465001897/datasets/ucf/list_video_perturbations/{args.corruptions}.txt'
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