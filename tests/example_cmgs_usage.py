#!/usr/bin/env python3
"""
Example script demonstrating CMGS (Confidence-Modulated Gradient Scaling) usage in ViTTA.

This script shows how to:
1. Enable CMGS with different parameter configurations
2. Run multi-epoch TTA with confidence-aware adaptation
3. Compare different CMGS settings
"""

import os
import sys
import torch
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.opts import get_opts
from corpus.test_time_adaptation import compute_cmgs_loss
from corpus.main_eval import eval

def example_basic_cmgs():
    """Example 1: Basic CMGS usage with default parameters."""
    print("=" * 60)
    print("Example 1: Basic CMGS Usage")
    print("=" * 60)
    
    # Parse arguments
    args = get_opts()
    
    # Basic configuration
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    args.tta = True
    args.verbose = True
    
    # Enable CMGS with default parameters
    args.use_cmgs = True
    args.cmgs_gamma = 0.367879  # e^(-1)
    args.cmgs_alpha = 1.0       # High confidence scaling
    args.cmgs_beta = 0.1        # Low confidence scaling
    
    # Multi-epoch settings
    args.n_epoch_adapat = 2
    args.n_gradient_steps = 1
    
    # Model and data configuration
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'
    args.spatiotemp_mean_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20220908_235138.npy'
    args.spatiotemp_var_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_var_20220908_235138.npy'
    
    # TTA settings
    args.if_tta_standard = 'tta_standard'
    args.stat_reg = 'mean_var'
    args.reg_type = 'l1_loss'
    args.chosen_blocks = ['layer3', 'layer4']
    args.lr = 1e-5
    args.lambda_feature_reg = 1.0
    args.lambda_pred_consis = 0.05
    args.momentum_mvg = 0.05
    
    # Data settings
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/val_corruptions'
    args.val_vid_list = '/scratch/project_465001897/datasets/ucf/list_video_perturbations/random_mini.txt'
    args.batch_size = 1
    args.workers = 2
    args.debug = True  # Only process first few samples
    
    # Video processing
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.scale_size = 256
    args.input_size = 224
    args.sample_style = 'uniform-1'
    args.gpus = [0]
    
    print(f"Configuration:")
    print(f"  CMGS enabled: {args.use_cmgs}")
    print(f"  Gamma: {args.cmgs_gamma}")
    print(f"  Alpha: {args.cmgs_alpha}")
    print(f"  Beta: {args.cmgs_beta}")
    print(f"  Multi-epoch: {args.n_epoch_adapat} epochs")
    print()
    
    try:
        print("Running TTA with CMGS...")
        results, _ = eval(args=args)
        print(f"Results: {results}")
        print("Basic CMGS Example: SUCCESS")
    except Exception as e:
        print(f"Error: {e}")
        print("Basic CMGS Example: FAILED")
    
    print("=" * 60)

def example_conservative_cmgs():
    """Example 2: Conservative CMGS with higher confidence threshold."""
    print("=" * 60)
    print("Example 2: Conservative CMGS")
    print("=" * 60)
    
    # Parse arguments
    args = get_opts()
    
    # Basic configuration
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    args.tta = True
    args.verbose = True
    
    # Conservative CMGS parameters
    args.use_cmgs = True
    args.cmgs_gamma = 0.7        # Higher confidence threshold
    args.cmgs_alpha = 1.5        # More aggressive high-confidence scaling
    args.cmgs_beta = 0.05        # More aggressive low-confidence dampening
    
    # Multi-epoch settings
    args.n_epoch_adapat = 3
    args.n_gradient_steps = 2
    
    # Model and data configuration (same as basic example)
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'
    args.spatiotemp_mean_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20220908_235138.npy'
    args.spatiotemp_var_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_var_20220908_235138.npy'
    
    # TTA settings
    args.if_tta_standard = 'tta_standard'
    args.stat_reg = 'mean_var'
    args.reg_type = 'l1_loss'
    args.chosen_blocks = ['layer3', 'layer4']
    args.lr = 1e-5
    args.lambda_feature_reg = 1.0
    args.lambda_pred_consis = 0.05
    args.momentum_mvg = 0.05
    
    # Data settings
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/val_corruptions'
    args.val_vid_list = '/scratch/project_465001897/datasets/ucf/list_video_perturbations/random_mini.txt'
    args.batch_size = 1
    args.workers = 2
    args.debug = True
    
    # Video processing
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.scale_size = 256
    args.input_size = 224
    args.sample_style = 'uniform-1'
    args.gpus = [0]
    
    print(f"Conservative Configuration:")
    print(f"  Gamma: {args.cmgs_gamma} (higher threshold)")
    print(f"  Alpha: {args.cmgs_alpha} (more aggressive)")
    print(f"  Beta: {args.cmgs_beta} (more dampening)")
    print(f"  Multi-epoch: {args.n_epoch_adapat} epochs")
    print(f"  Gradient steps: {args.n_gradient_steps}")
    print()
    
    try:
        print("Running conservative CMGS TTA...")
        results, _ = eval(args=args)
        print(f"Results: {results}")
        print("Conservative CMGS Example: SUCCESS")
    except Exception as e:
        print(f"Error: {e}")
        print("Conservative CMGS Example: FAILED")
    
    print("=" * 60)

def example_aggressive_cmgs():
    """Example 3: Aggressive CMGS with lower confidence threshold."""
    print("=" * 60)
    print("Example 3: Aggressive CMGS")
    print("=" * 60)
    
    # Parse arguments
    args = get_opts()
    
    # Basic configuration
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    args.tta = True
    args.verbose = True
    
    # Aggressive CMGS parameters
    args.use_cmgs = True
    args.cmgs_gamma = 0.2        # Lower confidence threshold
    args.cmgs_alpha = 2.0        # Very aggressive high-confidence scaling
    args.cmgs_beta = 0.01        # Very aggressive low-confidence dampening
    
    # Multi-epoch settings
    args.n_epoch_adapat = 4
    args.n_gradient_steps = 3
    
    # Model and data configuration (same as basic example)
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'
    args.spatiotemp_mean_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20220908_235138.npy'
    args.spatiotemp_var_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_var_20220908_235138.npy'
    
    # TTA settings
    args.if_tta_standard = 'tta_standard'
    args.stat_reg = 'mean_var'
    args.reg_type = 'l1_loss'
    args.chosen_blocks = ['layer3', 'layer4']
    args.lr = 1e-5
    args.lambda_feature_reg = 1.0
    args.lambda_pred_consis = 0.05
    args.momentum_mvg = 0.05
    
    # Data settings
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/val_corruptions'
    args.val_vid_list = '/scratch/project_465001897/datasets/ucf/list_video_perturbations/random_mini.txt'
    args.batch_size = 1
    args.workers = 2
    args.debug = True
    
    # Video processing
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.scale_size = 256
    args.input_size = 224
    args.sample_style = 'uniform-1'
    args.gpus = [0]
    
    print(f"Aggressive Configuration:")
    print(f"  Gamma: {args.cmgs_gamma} (lower threshold)")
    print(f"  Alpha: {args.cmgs_alpha} (very aggressive)")
    print(f"  Beta: {args.cmgs_beta} (very aggressive dampening)")
    print(f"  Multi-epoch: {args.n_epoch_adapat} epochs")
    print(f"  Gradient steps: {args.n_gradient_steps}")
    print()
    
    try:
        print("Running aggressive CMGS TTA...")
        results, _ = eval(args=args)
        print(f"Results: {results}")
        print("Aggressive CMGS Example: SUCCESS")
    except Exception as e:
        print(f"Error: {e}")
        print("Aggressive CMGS Example: FAILED")
    
    print("=" * 60)

def example_parameter_comparison():
    """Example 4: Compare different CMGS parameter settings."""
    print("=" * 60)
    print("Example 4: CMGS Parameter Comparison")
    print("=" * 60)
    
    # Create test data
    batch_size = 8
    num_classes = 10
    torch.manual_seed(42)
    
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Different parameter configurations
    configs = [
        {"name": "Default", "gamma": 0.367879, "alpha": 1.0, "beta": 0.1},
        {"name": "Conservative", "gamma": 0.7, "alpha": 1.5, "beta": 0.05},
        {"name": "Aggressive", "gamma": 0.2, "alpha": 2.0, "beta": 0.01},
        {"name": "Balanced", "gamma": 0.5, "alpha": 1.2, "beta": 0.2},
    ]
    
    print("Comparing different CMGS configurations:")
    print()
    
    for config in configs:
        loss, conf = compute_cmgs_loss(
            outputs, targets, 
            gamma=config["gamma"], 
            alpha=config["alpha"], 
            beta=config["beta"]
        )
        
        high_conf_ratio = (conf > config["gamma"]).float().mean().item()
        avg_confidence = conf.mean().item()
        
        print(f"{config['name']}:")
        print(f"  Gamma: {config['gamma']:.3f}")
        print(f"  Alpha: {config['alpha']:.1f}")
        print(f"  Beta: {config['beta']:.2f}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Avg confidence: {avg_confidence:.4f}")
        print(f"  High-conf ratio: {high_conf_ratio:.3f}")
        print()
    
    print("Parameter Comparison: COMPLETED")
    print("=" * 60)

def main():
    """Run all CMGS examples."""
    print("CMGS Usage Examples")
    print("=" * 60)
    
    # Example 1: Basic CMGS
    example_basic_cmgs()
    print()
    
    # Example 2: Conservative CMGS
    example_conservative_cmgs()
    print()
    
    # Example 3: Aggressive CMGS
    example_aggressive_cmgs()
    print()
    
    # Example 4: Parameter comparison
    example_parameter_comparison()
    
    print("\n" + "=" * 60)
    print("All CMGS Examples Completed")
    print("=" * 60)

if __name__ == '__main__':
    main() 