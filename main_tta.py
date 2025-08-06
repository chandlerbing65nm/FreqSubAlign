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
                    'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20250529_134901.npy',
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
                        'lambda_pred_consis': 0.05,
                        'momentum_mvg': 0.07,
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
    
    # Set TTA-specific parameters if in TTA mode
    if args.tta:
        args.spatiotemp_mean_clean_file = model_config['spatiotemp_mean_clean_file']
        args.spatiotemp_var_clean_file = model_config['spatiotemp_var_clean_file']
        args.n_epoch_adapat = 1  # Number of adaptation epochs for TTA
    else:
        # Source-only evaluation parameters
        args.evaluate_baselines = True
        args.baseline = 'source'
    
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
    args.batch_size = 1  # Default to 1 for TTA, can be overridden
    # args.vid_format = '.mp4'  # Only for somethingv2

    # Set up corruption types to evaluate
    corruptions = [
        # 'gauss', 
        # 'pepper', 
        # 'salt', 
        # 'shot',
        # 'zoom', 
        # 'impulse', 
        # 'defocus', 
        # 'motion',
        'jpeg', 
        'contrast', 
        'rain', 
        'h265_abr'
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