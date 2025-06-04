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
torch.backends.cudnn.benchmark = False

corruptions = [
    # 'gauss', 'pepper', 'salt', 'shot',
    # 'zoom', 'impulse', 'defocus', 'motion',
    # 'jpeg',
    'contrast', 'rain', 'h265_abr']

def get_model_config(arch, dataset='somethingv2'):
    """Get model configuration based on architecture and dataset."""
    config = {
        'model_path': '',
        'video_data_dir': '',
        'val_vid_list': '',
        'result_dir': '',
        'additional_args': {}
    }
    
    # Common parameters for both architectures
    common_args = {
        'batch_size': 1,
        'test_crops': 1,
        'frame_uniform': True,
        'frame_interval': 2,
        'scale_size': 224,
    }
    
    if arch == 'videoswintransformer':
        config.update({
            'model_path': '/scratch/project_465001897/datasets/ss2/model_swin/swin_base_patch244_window1677_sthv2.pth',
            'video_data_dir': '',
            'val_vid_list': '',
            'result_dir': '',
            'additional_args': {
                **common_args,
                'patch_size': (2, 4, 4),
                'num_clips': 1,
            }
        })
        
        # Set architecture-specific parameters
        if dataset == 'somethingv2':
            config['additional_args'].update({
                'clip_length': 16,
                'window_size': (16, 7, 7)
            })
        elif dataset == 'ucf101':
            config['additional_args'].update({
                'clip_length': 16,
                'window_size': (8, 7, 7)
            })
            
    elif arch == 'tanet':
        config.update({
            'model_path': '/scratch/project_465001897/datasets/ss2/model_tanet/ckpt.best.pth.tar',
            'video_data_dir': '',
            'val_vid_list': '',
            'result_dir': '',
            'additional_args': {
                **common_args,
                'sample_style': 'uniform-1'
            }
        })
        
        # Set architecture-specific parameters
        if dataset == 'somethingv2':
            config['additional_args'].update({
                'clip_length': 8,
                'window_size': (8, 7, 7)
            })
        elif dataset == 'ucf101':
            config['additional_args'].update({
                'clip_length': 16,
                'window_size': (8, 7, 7)
            })
    
    return config

if __name__ == '__main__':
    global args
    args = get_opts()
    
    # Set seed
    set_seed(142)
    
    args.gpus = [0]
    args.dataset = 'somethingv2'
    args.video_data_dir = '/scratch/project_465001897/datasets/ss2/val_corruptions'
    args.batch_size = 1
    args.vid_format = '.mp4' # only for ss2

    # Choose model architecture (either 'videoswintransformer' or 'tanet')
    args.arch = 'videoswintransformer'  # Change this to switch between models
    
    # Get model-specific configuration
    model_config = get_model_config(args.arch, args.dataset)
    args.model_path = model_config['model_path']
    
    # Set additional arguments for the selected architecture
    for key, value in model_config['additional_args'].items():
        setattr(args, key, value)

    # Set source-only evaluation parameters
    args.tta = False
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    # Create parent results directory
    parent_result_dir = f'/scratch/project_465001897/datasets/ss2/results/source/{args.arch}_{args.dataset}'
    os.makedirs(parent_result_dir, exist_ok=True)
    
    # Create a single results file for all corruptions
    f_write = get_writer_to_all_result(args, custom_path=parent_result_dir)
    f_write.write('Source-only Evaluation Results:\n')
    f_write.write('#############################\n')

    for corr_id, args.corruptions in enumerate(corruptions):
        print(f'####Starting Evaluation for ::: {args.corruptions} corruption####')
        args.val_vid_list = f'/scratch/project_465001897/datasets/ss2/list_video_perturbations/{args.corruptions}.txt'
        args.result_dir = f'/scratch/project_465001897/datasets/ss2/results/source/{args.arch}_{args.dataset}/tta_{args.corruptions}'

        # Clear GPU memory before each corruption
        torch.cuda.empty_cache()
        
        epoch_result_list, _ = eval(args=args)  # Unpack the tuple, ignore the model return value

        # Write corruption name and results
        f_write.write(f'\n{args.corruptions}:\n')
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in epoch_result_list]) + '\n')
        f_write.flush()

    f_write.close() 