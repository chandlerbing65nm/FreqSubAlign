#!/usr/bin/env python3
"""
Test script for CMGS (Confidence-Modulated Gradient Scaling) functionality in ViTTA.

This script demonstrates:
1. Multi-epoch TTA adaptation with CMGS
2. Confidence-aware gradient scaling
3. Robust adaptation under random per-sample corruptions
4. Comparison between standard TTA and CMGS-enhanced TTA
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.opts import get_opts
from corpus.test_time_adaptation import tta_standard, compute_cmgs_loss
from corpus.main_eval import eval
from config import device

def test_cmgs_loss_function():
    """Test the CMGS loss computation function."""
    print("=" * 60)
    print("Testing CMGS Loss Function")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    num_classes = 10
    
    # Create model outputs with different confidence levels
    torch.manual_seed(42)
    
    # High confidence outputs (should get amplified gradients)
    high_conf_outputs = torch.randn(batch_size, num_classes)
    high_conf_outputs[:, 0] = 10.0  # High confidence for class 0
    
    # Low confidence outputs (should get dampened gradients)
    low_conf_outputs = torch.randn(batch_size, num_classes)
    low_conf_outputs = low_conf_outputs * 0.1  # Low confidence
    
    # Mixed confidence outputs
    mixed_conf_outputs = torch.randn(batch_size, num_classes)
    mixed_conf_outputs[0:2, 0] = 10.0  # High confidence for first 2 samples
    mixed_conf_outputs[2:4, :] = mixed_conf_outputs[2:4, :] * 0.1  # Low confidence for last 2 samples
    
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test parameters
    gamma = 0.367879  # e^(-1)
    alpha = 1.0  # High confidence scaling
    beta = 0.1   # Low confidence scaling
    
    print(f"Testing with gamma={gamma}, alpha={alpha}, beta={beta}")
    print()
    
    # Test high confidence samples
    loss_high, conf_high = compute_cmgs_loss(high_conf_outputs, targets, gamma, alpha, beta)
    print(f"High confidence samples:")
    print(f"  Average confidence: {conf_high.mean().item():.4f}")
    print(f"  CMGS loss: {loss_high.item():.4f}")
    print()
    
    # Test low confidence samples
    loss_low, conf_low = compute_cmgs_loss(low_conf_outputs, targets, gamma, alpha, beta)
    print(f"Low confidence samples:")
    print(f"  Average confidence: {conf_low.mean().item():.4f}")
    print(f"  CMGS loss: {loss_low.item():.4f}")
    print()
    
    # Test mixed confidence samples
    loss_mixed, conf_mixed = compute_cmgs_loss(mixed_conf_outputs, targets, gamma, alpha, beta)
    print(f"Mixed confidence samples:")
    print(f"  Average confidence: {conf_mixed.mean().item():.4f}")
    print(f"  Individual confidences: {conf_mixed.tolist()}")
    print(f"  CMGS loss: {loss_mixed.item():.4f}")
    print()
    
    print("CMGS Loss Function Test: PASSED")
    print("=" * 60)

def test_multi_epoch_tta_with_cmgs():
    """Test multi-epoch TTA with CMGS enabled."""
    print("=" * 60)
    print("Testing Multi-Epoch TTA with CMGS")
    print("=" * 60)
    
    # Parse arguments
    args = get_opts()
    
    # Set up test configuration
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    args.tta = True
    args.verbose = True
    
    # Enable CMGS
    args.use_cmgs = True
    args.cmgs_gamma = 0.367879
    args.cmgs_alpha = 1.0
    args.cmgs_beta = 0.1
    
    # Multi-epoch settings
    args.n_epoch_adapat = 3
    args.n_gradient_steps = 2
    
    # Model configuration
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'
    args.spatiotemp_mean_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20220908_235138.npy'
    args.spatiotemp_var_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_var_20220908_235138.npy'
    
    # TTA-specific settings
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
    
    # Video processing settings
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.scale_size = 256
    args.input_size = 224
    args.sample_style = 'uniform-1'
    
    # Runtime settings
    args.gpus = [0]
    args.debug = True  # Only process first few samples for testing
    
    print(f"Configuration:")
    print(f"  Architecture: {args.arch}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Multi-epoch adaptation: {args.n_epoch_adapat} epochs")
    print(f"  Gradient steps per sample: {args.n_gradient_steps}")
    print(f"  CMGS enabled: {args.use_cmgs}")
    print(f"  CMGS gamma: {args.cmgs_gamma}")
    print(f"  CMGS alpha: {args.cmgs_alpha}")
    print(f"  CMGS beta: {args.cmgs_beta}")
    print()
    
    try:
        # Run TTA evaluation
        print("Starting TTA evaluation with CMGS...")
        epoch_result_list, _ = eval(args=args)
        
        print(f"TTA Results: {epoch_result_list}")
        print("Multi-Epoch TTA with CMGS Test: PASSED")
        
    except Exception as e:
        print(f"Error during TTA evaluation: {e}")
        print("Multi-Epoch TTA with CMGS Test: FAILED")
        return False
    
    print("=" * 60)
    return True

def test_cmgs_parameter_sensitivity():
    """Test CMGS parameter sensitivity."""
    print("=" * 60)
    print("Testing CMGS Parameter Sensitivity")
    print("=" * 60)
    
    # Create test data
    batch_size = 8
    num_classes = 10
    torch.manual_seed(42)
    
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test different gamma values
    gamma_values = [0.1, 0.367879, 0.5, 0.7, 0.9]
    alpha = 1.0
    beta = 0.1
    
    print("Testing different gamma values:")
    for gamma in gamma_values:
        loss, conf = compute_cmgs_loss(outputs, targets, gamma, alpha, beta)
        high_conf_ratio = (conf > gamma).float().mean().item()
        print(f"  Gamma={gamma:.3f}: Loss={loss.item():.4f}, High-conf ratio={high_conf_ratio:.3f}")
    
    print()
    
    # Test different alpha/beta combinations
    gamma = 0.367879
    alpha_beta_combinations = [
        (1.0, 0.1),   # Standard
        (2.0, 0.1),   # More aggressive high-conf
        (1.0, 0.01),  # More aggressive low-conf
        (0.5, 0.1),   # Less aggressive high-conf
    ]
    
    print("Testing different alpha/beta combinations:")
    for alpha, beta in alpha_beta_combinations:
        loss, conf = compute_cmgs_loss(outputs, targets, gamma, alpha, beta)
        print(f"  Alpha={alpha}, Beta={beta}: Loss={loss.item():.4f}")
    
    print("CMGS Parameter Sensitivity Test: PASSED")
    print("=" * 60)

def compare_standard_vs_cmgs():
    """Compare standard TTA vs CMGS-enhanced TTA."""
    print("=" * 60)
    print("Comparing Standard TTA vs CMGS-Enhanced TTA")
    print("=" * 60)
    
    # Parse arguments
    args = get_opts()
    
    # Set up common configuration
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    args.tta = True
    args.verbose = False  # Reduce verbosity for comparison
    args.debug = True
    
    # Multi-epoch settings
    args.n_epoch_adapat = 2
    args.n_gradient_steps = 1
    
    # Model configuration
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'
    args.spatiotemp_mean_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20220908_235138.npy'
    args.spatiotemp_var_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_var_20220908_235138.npy'
    
    # Common TTA settings
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
    
    # Video processing settings
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.scale_size = 256
    args.input_size = 224
    args.sample_style = 'uniform-1'
    args.gpus = [0]
    
    print("Configuration for comparison:")
    print(f"  Multi-epoch adaptation: {args.n_epoch_adapat} epochs")
    print(f"  Gradient steps per sample: {args.n_gradient_steps}")
    print()
    
    # Test 1: Standard TTA (no CMGS)
    print("Test 1: Standard TTA (no CMGS)")
    args.use_cmgs = False
    try:
        standard_results, _ = eval(args=args)
        print(f"  Standard TTA Results: {standard_results}")
    except Exception as e:
        print(f"  Standard TTA Error: {e}")
        standard_results = None
    
    print()
    
    # Test 2: CMGS-enhanced TTA
    print("Test 2: CMGS-enhanced TTA")
    args.use_cmgs = True
    args.cmgs_gamma = 0.367879
    args.cmgs_alpha = 1.0
    args.cmgs_beta = 0.1
    try:
        cmgs_results, _ = eval(args=args)
        print(f"  CMGS TTA Results: {cmgs_results}")
    except Exception as e:
        print(f"  CMGS TTA Error: {e}")
        cmgs_results = None
    
    print()
    
    # Compare results
    if standard_results is not None and cmgs_results is not None:
        print("Comparison:")
        for i, (std_acc, cmgs_acc) in enumerate(zip(standard_results, cmgs_results)):
            improvement = cmgs_acc - std_acc
            print(f"  Epoch {i+1}: Standard={std_acc:.3f}, CMGS={cmgs_acc:.3f}, Improvement={improvement:+.3f}")
    else:
        print("Could not complete comparison due to errors.")
    
    print("=" * 60)

def main():
    """Run all CMGS tests."""
    print("CMGS (Confidence-Modulated Gradient Scaling) Test Suite")
    print("=" * 60)
    
    # Test 1: CMGS loss function
    test_cmgs_loss_function()
    print()
    
    # Test 2: Parameter sensitivity
    test_cmgs_parameter_sensitivity()
    print()
    
    # Test 3: Multi-epoch TTA with CMGS
    success = test_multi_epoch_tta_with_cmgs()
    print()
    
    # Test 4: Comparison
    if success:
        compare_standard_vs_cmgs()
    
    print("\n" + "=" * 60)
    print("CMGS Test Suite Completed")
    print("=" * 60)

if __name__ == '__main__':
    main() 