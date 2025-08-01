import os
# Configure AMD GPU

from utils.opts import get_opts
from utils.utils_ import get_writer_to_all_result
from corpus.main_eval import eval
import torch
import random
import numpy as np

# Set device configuration
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
torch.backends.cudnn.benchmark = True

# corruptions = [
#     'gauss_shuffled', 'pepper_shuffled', 'salt_shuffled', 'shot_shuffled',
#     'zoom_shuffled', 'impulse_shuffled', 'defocus_shuffled', 'motion_shuffled',
#     'jpeg_shuffled', 'contrast_shuffled', 'rain_shuffled', 'h265_abr_shuffled',  
# ]

corruptions = [
    'clean']


def get_model_config(arch, dataset='somethingv2'):
    """Get model configuration based on architecture and dataset."""
    config = {
        'model_path': '',
        'spatiotemp_mean_clean_file': '',
        'spatiotemp_var_clean_file': '',
        'additional_args': {}
    }
    
    # Common parameters for both architectures
    common_args = {
        'test_crops': 3,
        'frame_uniform': True,
        'frame_interval': 2,
    }
    
    if arch == 'videoswintransformer':
        if dataset == 'ucf101':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ucf/model_swin/swin_ucf_base_patch244_window877_pretrain_kinetics400_30epoch_lr3e-5.pthh',
                'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_swin/list_spatiotemp_mean_20221004_192722.npy',
                'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_swin/list_spatiotemp_var_20221004_192722.npy',
                'additional_args': {
                    **common_args,
                    'clip_length': 16,
                    'num_clips': 1,
                    'patch_size': (2, 4, 4),
                    'window_size': (8, 7, 7),
                    'chosen_blocks': ['module.backbone.layers.2', 'module.backbone.layers.3', 'module.backbone.norm'],
                    'lr': 1e-5,  # Swin-UCF
                    'lambda_pred_consis': 0.05,
                    'momentum_mvg': 0.05,
                    'scale_size': 224,
                }
            })
        elif dataset == 'somethingv2':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ss2/model_swin/swin_base_patch244_window1677_sthv2.pth',
                'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ss2/source_statistics_swin/list_spatiotemp_mean_20250603_172215.npy',
                'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ss2/source_statistics_swin/list_spatiotemp_var_20250603_172215.npy',
                'additional_args': {
                    **common_args,
                    'clip_length': 16,
                    'num_clips': 1,
                    'patch_size': (2, 4, 4),
                    'window_size': (16, 7, 7),
                    'chosen_blocks': ['module.backbone.layers.2', 'module.backbone.layers.3', 'module.backbone.norm'],
                    'lr': 1e-5,  # Swin-SS2
                    'lambda_pred_consis': 0.05,
                    'momentum_mvg': 0.05,
                    'scale_size': 224,
                }
            })
            
    elif arch == 'tanet':
        if dataset == 'somethingv2':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ss2/source_statistics_tanet/TR50_S2_256_8x3x2.pth.tar',
                'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ss2/source_statistics_tanet/list_spatiotemp_mean_20250606_173132.npy',
                'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ss2/source_statistics_tanet/list_spatiotemp_mean_20250606_173132.npy',
                'additional_args': {
                    **common_args,
                    'clip_length': 8,
                    'scale_size': 256,
                    'lr': 1e-5  # TANet-SS2
                }
            })
        elif dataset == 'ucf101':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar',
                'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20220908_235138.npy',
                'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20250529_134901.npy',
                'additional_args': {
                    **common_args,
                    'clip_length': 16,
                    'scale_size': 256,
                    'lr': 5e-5  # TANet-UCF
                }
            })
        elif dataset == 'uffia':
            config.update({
                'model_path': '/scratch/project_465001897/datasets/uffia/results/train/tanet_20250731_195801/20250731_195801_uffia_rgb_model_best.pth.tar',
                'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/uffia/source_statistics_tanet/list_spatiotemp_mean_20250801_135307.npy',
                'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/uffia/source_statistics_tanet/list_spatiotemp_var_20250801_135307.npy',
                'additional_args': {
                    **common_args,
                    'clip_length': 16,
                    'scale_size': 256,
                    'lr': 1e-3  # TANet-UFFIA
                }
            })
    
    return config

if __name__ == '__main__':
    global args
    args = get_opts()
    
    # Set seed
    set_seed(142)
    
    args.gpus = [0]
    args.video_data_dir = '/scratch/project_465001897/datasets/uffia/video'
    args.batch_size = 1 
    args.n_epoch_adapat = 1
    args.vid_format = '.mp4' # only for ss2

    # Choose model architecture
    args.arch = 'tanet' # videoswintransformer, tanet
    args.dataset = 'uffia' # somethingv2, ucf101, uffia
    
    # Get model-specific configuration
    model_config = get_model_config(args.arch, args.dataset)
    args.model_path = model_config['model_path']
    args.spatiotemp_mean_clean_file = model_config['spatiotemp_mean_clean_file']
    args.spatiotemp_var_clean_file = model_config['spatiotemp_var_clean_file']
    
    # Set additional arguments for the selected architecture
    for key, value in model_config['additional_args'].items():
        setattr(args, key, value)

    # Create parent results directory
    parent_result_dir = f'/scratch/project_465001897/datasets/uffia/results/corruptions/{args.arch}_{args.dataset}'
    os.makedirs(parent_result_dir, exist_ok=True)
    
    # Create a single results file for all corruptions
    f_write = get_writer_to_all_result(args, custom_path=parent_result_dir)
    f_write.write('Corruption Results:\n')
    f_write.write('#############################\n')

    for corr_id, args.corruptions in enumerate(corruptions):
        print(f'####Starting Evaluation for ::: {args.corruptions} corruption####')
        args.val_vid_list = f'/scratch/project_465001897/datasets/uffia/list_video_perturbations/{args.corruptions}.txt'
        args.result_dir = f'/scratch/project_465001897/datasets/uffia/results/corruptions/{args.arch}_{args.dataset}/tta_{args.corruptions}'

        # Clear GPU memory before each corruption
        torch.cuda.empty_cache()
        
        epoch_result_list, _ = eval(args=args)

        # Write corruption name and results
        f_write.write(f'\n{args.corruptions}:\n')
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in epoch_result_list]) + '\n')
        f_write.flush()

    f_write.close() 