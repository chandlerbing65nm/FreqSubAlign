#!/usr/bin/env python3
"""
Simple Example: Using Confidence-Gated Optimizer with ViTTA

This script demonstrates how to enable and configure the Confidence-Gated Optimizer
for robust test-time adaptation in ViTTA.

The CGO prevents the model from adapting to highly uncertain/corrupted samples
by gating weight updates based on prediction confidence.
"""

import os
import sys
import torch
import random
import numpy as np

# Add ViTTA modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.opts import get_opts
from corpus.main_eval import eval


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_cgo_example():
    """Run a simple example with CGO-enhanced ViTTA."""
    
    print("="*60)
    print("Confidence-Gated Optimizer (CGO) Example")
    print("="*60)
    
    # Parse command line arguments
    args = get_opts()
    
    # Set seed for reproducibility
    set_seed(142)
    
    # ========================================
    # ENABLE CGO CONFIGURATION
    # ========================================
    
    # Enable CGO
    args.use_cgo = True
    args.cgo_confidence_threshold = 0.7  # Only adapt when confidence > 0.7
    args.cgo_confidence_metric = 'max_softmax'  # Use max softmax as confidence
    args.cgo_enable_logging = True  # Enable detailed logging
    
    # Optional: Use adaptive CGO (dynamically adjusts threshold)
    args.cgo_adaptive = False  # Set to True for adaptive behavior
    args.cgo_min_threshold = 0.5
    args.cgo_max_threshold = 0.9
    args.cgo_target_adaptation_rate = 0.7
    
    print(f"CGO Configuration:")
    print(f"  - Enabled: {args.use_cgo}")
    print(f"  - Confidence Threshold: {args.cgo_confidence_threshold}")
    print(f"  - Confidence Metric: {args.cgo_confidence_metric}")
    print(f"  - Adaptive: {args.cgo_adaptive}")
    print()
    
    # ========================================
    # MODEL AND DATA CONFIGURATION
    # ========================================
    
    # Choose model architecture and dataset
    args.arch = 'tanet'  # or 'videoswintransformer'
    args.dataset = 'ucf101'  # or 'somethingv2'
    args.tta = True  # Enable test-time adaptation
    
    # Model paths (update these paths for your setup)
    args.model_path = '/path/to/your/model.pth'
    args.spatiotemp_mean_clean_file = '/path/to/your/mean_stats.npy'
    args.spatiotemp_var_clean_file = '/path/to/your/var_stats.npy'
    
    # Video processing parameters
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.scale_size = 256
    args.input_size = 224
    args.batch_size = 1
    
    # TTA parameters
    args.chosen_blocks = ['layer3', 'layer4']
    args.lr = 5e-5
    args.lambda_pred_consis = 0.05
    args.momentum_mvg = 0.05
    args.lambda_feature_reg = 1
    args.n_epoch_adapat = 1
    args.include_ce_in_consistency = False
    
    # Data paths (update these paths for your setup)
    args.gpus = [0]
    args.video_data_dir = '/path/to/your/corrupted/videos'
    args.workers = 4
    args.verbose = True
    
    print(f"Model Configuration:")
    print(f"  - Architecture: {args.arch}")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Adaptation Epochs: {args.n_epoch_adapat}")
    print()
    
    # ========================================
    # CONSERVATIVE CGO CONFIGURATION AFTER FIX
    # ========================================
    
    args.use_cgo = True
    args.cgo_confidence_threshold = 0.6  # Moderate threshold
    args.cgo_confidence_metric = 'max_softmax'
    args.cgo_enable_logging = True
    args.cgo_adaptive = False  # Use fixed threshold for stability
    
    # Reset learning rate to original
    args.lr = 5e-5
    
    # Generate result suffix for this configuration
    result_suffix = f"cgo_fixed_tau{args.cgo_confidence_threshold}_adaptive{args.cgo_adaptive}"
    
    print(f"CGO Configuration:")
    print(f"  - Fixed CGO implementation")
    print(f"  - Confidence threshold: {args.cgo_confidence_threshold}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Adaptive mode: {args.cgo_adaptive}")
    print(f"  - Result suffix: {result_suffix}")
    print()
    
    # ========================================
    # RUN EVALUATION WITH DIFFERENT CORRUPTIONS
    # ========================================
    
    corruptions = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'random']
    
    print("Starting CGO-enhanced TTA evaluation...")
    print("-" * 40)
    
    results = {}
    
    for corruption in corruptions:
        print(f"\nEvaluating corruption: {corruption}")
        
        # Set corruption-specific parameters
        args.corruptions = corruption
        args.val_vid_list = f'/path/to/your/corruption_lists/{corruption}.txt'
        args.result_dir = f'./results/cgo_{corruption}'
        args.result_suffix = f'cgo_threshold_{args.cgo_confidence_threshold}'
        
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Run evaluation with CGO
            epoch_result_list, _ = eval(args=args)
            
            results[corruption] = epoch_result_list
            print(f"Results for {corruption}: {[round(x, 3) for x in epoch_result_list]}")
            
        except Exception as e:
            print(f"Error evaluating {corruption}: {str(e)}")
            print("Note: Make sure to update the file paths in this example script")
            results[corruption] = [0.0]
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nCGO Configuration:")
    print(f"  - Threshold: {args.cgo_confidence_threshold}")
    print(f"  - Metric: {args.cgo_confidence_metric}")
    print(f"  - Adaptive: {args.cgo_adaptive}")
    
    print(f"\nResults Summary:")
    for corruption, result_list in results.items():
        if result_list:
            final_acc = result_list[-1]
            print(f"  - {corruption}: {final_acc:.3f}")
    
    print(f"\nKey Benefits of CGO:")
    print(f"  ✓ Prevents adaptation to highly corrupted/uncertain samples")
    print(f"  ✓ Maintains model stability under random corruptions")
    print(f"  ✓ Configurable confidence thresholds")
    print(f"  ✓ Optional adaptive threshold adjustment")
    print(f"  ✓ Detailed logging of adaptation decisions")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


def compare_with_without_cgo():
    """Compare performance with and without CGO."""
    
    print("\n" + "="*60)
    print("CGO vs Standard ViTTA Comparison")
    print("="*60)
    
    # This function demonstrates how you would compare
    # CGO-enhanced ViTTA with standard ViTTA
    
    configurations = [
        {
            'name': 'Standard ViTTA',
            'use_cgo': False,
            'description': 'Standard test-time adaptation'
        },
        {
            'name': 'CGO-Enhanced ViTTA',
            'use_cgo': True,
            'cgo_confidence_threshold': 0.7,
            'description': 'Confidence-gated test-time adaptation'
        }
    ]
    
    print("Comparison configurations:")
    for i, config in enumerate(configurations, 1):
        print(f"{i}. {config['name']}: {config['description']}")
    
    print("\nTo run this comparison:")
    print("1. Update the file paths in this script")
    print("2. Run the script with your dataset")
    print("3. Compare the results between configurations")
    print("4. CGO should show more stable performance on random corruptions")


if __name__ == '__main__':
    print("CGO-ViTTA Example Script")
    print("Note: Update file paths before running")
    
    # Run the main example
    try:
        run_cgo_example()
    except Exception as e:
        print(f"\nExample failed (expected - paths need to be updated): {str(e)}")
        print("\nTo use this script:")
        print("1. Update model_path, spatiotemp_mean_clean_file, spatiotemp_var_clean_file")
        print("2. Update video_data_dir and val_vid_list paths")
        print("3. Ensure your environment has the required dependencies")
    
    # Show comparison info
    compare_with_without_cgo()
