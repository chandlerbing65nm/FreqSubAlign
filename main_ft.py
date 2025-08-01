#!/usr/bin/env python3
"""
Training script for finetuning video models on fish feeding dataset.
Based on ViTTA codebase but modified for standard finetuning without TTA.
Uses get_opts() from utils/opts.py for argument parsing.
"""

import os
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

from corpus.main_train import main_train
from utils.opts import get_opts
from config import device

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_config(arch, dataset='uffia'):
    """Get model configuration based on architecture and dataset."""
    config = {
        'model_path': '',
        'num_classes': 4  # uffia has 4 classes for fish feeding
    }
    
    if arch == 'videoswintransformer':
        if dataset == 'uffia':
            config.update({
                'clip_length': 16,
                'patch_size': (2, 4, 4),
                'window_size': (8, 7, 7),
            })
    elif arch == 'tanet':
        if dataset == 'uffia':
            config.update({
                'clip_length': 16,
            })
    
    return config

def main():
    # """Main training function.""
    # Set random seed
    set_seed(42)
    
    # Get arguments using existing get_opts()
    args = get_opts()
    
    # Override arguments for finetuning training
    args.dataset = 'uffia'
    args.video_data_dir = '/scratch/project_465001897/datasets/uffia/video'
    args.train_vid_list = '/scratch/project_465001897/datasets/uffia/split/train_rgb_split_1.txt'
    args.val_vid_list = '/scratch/project_465001897/datasets/uffia/split/val_rgb_split_1.txt'
    args.vid_format = '.mp4'
    
    # Training parameters
    args.epochs = 50
    args.batch_size = 24
    args.lr = 1e-3
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.lr_steps = [20, 40]
    args.start_epoch = 0
    args.eval_freq = 1
    args.if_save_model = True
    args.snapshot_pref = args.dataset
    
    # Data processing parameters
    args.clip_length = 16
    args.scale_size = 256
    args.input_size = 224
    args.frame_uniform = True
    args.frame_interval = 2
    args.test_crops = 1  # Use single crop for training validation
    args.sample_style = 'uniform-1'  # TANet: single clip
    args.num_clips = 1  # Video Swin: single clip
    args.tsn_style = True
    
    # Disable TTA-related functionality
    args.tta = False
    args.evaluate_baselines = False  # Disable baselines for training
    args.baseline = 'source'
    args.evaluate = False
    args.resume = '/scratch/project_465001897/datasets/uffia/results/train/tanet_20250731_195801/20250731_195801_uffia_rgb_checkpoint.pth.tar'
     
    # Get model-specific configuration
    model_config = get_model_config(args.arch, args.dataset)
    if not args.model_path:
        args.model_path = model_config['model_path']
    
    # Create result directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.result_dir = f'/scratch/project_465001897/datasets/uffia/results/train/{args.arch}_{timestamp}'
    
    print(f"Training configuration:")
    print(f"  Architecture: {args.arch}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Model path: {args.model_path}")
    print(f"  Results dir: {args.result_dir}")
    print(f"  Video dir: {args.video_data_dir}")
    print(f"  Train list: {args.train_vid_list}")
    print(f"  Val list: {args.val_vid_list}")
    print(f"  Clip length: {args.clip_length}")
    print(f"  Scale size: {args.scale_size}")
    print(f"  Input size: {args.input_size}")
    
    # Start training
    main_train(args)

if __name__ == '__main__':
    main()