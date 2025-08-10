#!/usr/bin/env python3
"""
Comprehensive Test Script for Confidence-Gated Optimizer (CGO) with ViTTA

This script tests the CGO-enhanced ViTTA pipeline with various configurations
and corruption scenarios to demonstrate robust test-time adaptation.

Usage:
    python test_cgo_vitta.py [--config CONFIG_NAME]
    
Available configurations:
    - standard_cgo: Standard CGO with fixed threshold
    - adaptive_cgo: Adaptive CGO with dynamic threshold
    - baseline: Standard ViTTA without CGO
    - comparison: Run all configurations for comparison
"""

import os
import sys
import torch
import random
import numpy as np
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

# Add ViTTA modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.opts import get_opts
from utils.utils_ import get_writer_to_all_result
from corpus.main_eval import eval
from config import device
from utils.confidence_gated_optimizer import ConfidenceGatedOptimizer, AdaptiveConfidenceGatedOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CGOTestConfig:
    """Configuration class for CGO testing scenarios."""
    
    def __init__(self, name, description, args_override):
        self.name = name
        self.description = description
        self.args_override = args_override


def get_test_configurations():
    """Define test configurations for different CGO scenarios."""
    
    configs = {
        'baseline': CGOTestConfig(
            name='baseline',
            description='Standard ViTTA without CGO',
            args_override={
                'use_cgo': False,
                'n_epoch_adapat': 1,
                'result_suffix': 'baseline_no_cgo'
            }
        ),
        
        'standard_cgo': CGOTestConfig(
            name='standard_cgo',
            description='Standard CGO with fixed threshold (0.7)',
            args_override={
                'use_cgo': True,
                'cgo_confidence_threshold': 0.7,
                'cgo_confidence_metric': 'max_softmax',
                'cgo_adaptive': False,
                'cgo_enable_logging': True,
                'n_epoch_adapat': 1,
                'result_suffix': 'cgo_standard_threshold_0.7'
            }
        ),
        
        'conservative_cgo': CGOTestConfig(
            name='conservative_cgo',
            description='Conservative CGO with high threshold (0.8)',
            args_override={
                'use_cgo': True,
                'cgo_confidence_threshold': 0.8,
                'cgo_confidence_metric': 'max_softmax',
                'cgo_adaptive': False,
                'cgo_enable_logging': True,
                'n_epoch_adapat': 1,
                'result_suffix': 'cgo_conservative_threshold_0.8'
            }
        ),
        
        'aggressive_cgo': CGOTestConfig(
            name='aggressive_cgo',
            description='Aggressive CGO with low threshold (0.6)',
            args_override={
                'use_cgo': True,
                'cgo_confidence_threshold': 0.6,
                'cgo_confidence_metric': 'max_softmax',
                'cgo_adaptive': False,
                'cgo_enable_logging': True,
                'n_epoch_adapat': 1,
                'result_suffix': 'cgo_aggressive_threshold_0.6'
            }
        ),
        
        'adaptive_cgo': CGOTestConfig(
            name='adaptive_cgo',
            description='Adaptive CGO with dynamic threshold adjustment',
            args_override={
                'use_cgo': True,
                'cgo_confidence_threshold': 0.7,
                'cgo_confidence_metric': 'max_softmax',
                'cgo_adaptive': True,
                'cgo_min_threshold': 0.5,
                'cgo_max_threshold': 0.9,
                'cgo_target_adaptation_rate': 0.7,
                'cgo_enable_logging': True,
                'n_epoch_adapat': 1,
                'result_suffix': 'cgo_adaptive_target_0.7'
            }
        ),
        
        'entropy_cgo': CGOTestConfig(
            name='entropy_cgo',
            description='CGO using entropy-based confidence metric',
            args_override={
                'use_cgo': True,
                'cgo_confidence_threshold': 0.7,
                'cgo_confidence_metric': 'entropy',
                'cgo_adaptive': False,
                'cgo_enable_logging': True,
                'n_epoch_adapat': 1,
                'result_suffix': 'cgo_entropy_threshold_0.7'
            }
        ),
        
        'multi_epoch_cgo': CGOTestConfig(
            name='multi_epoch_cgo',
            description='Multi-epoch adaptation with CGO',
            args_override={
                'use_cgo': True,
                'cgo_confidence_threshold': 0.7,
                'cgo_confidence_metric': 'max_softmax',
                'cgo_adaptive': False,
                'cgo_enable_logging': True,
                'n_epoch_adapat': 3,
                'result_suffix': 'cgo_multi_epoch_3'
            }
        )
    }
    
    return configs


def setup_base_args():
    """Setup base arguments for ViTTA evaluation."""
    args = get_opts()
    
    # Set seed for reproducibility
    set_seed(142)
    
    # Base model configuration
    args.arch = 'tanet'  # or 'videoswintransformer'
    args.dataset = 'ucf101'
    args.tta = True
    
    # Model paths (these would need to be updated for actual testing)
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf_rgb_r50_seg8_f1s1_b16_g8_lr0.01_wd0.0001_dp0.5_cosine_warmup_e100.pth'
    args.spatiotemp_mean_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_mean_20221004_192722.npy'
    args.spatiotemp_var_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet/list_spatiotemp_var_20221004_192722.npy'
    
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
    args.include_ce_in_consistency = False
    
    # Data paths (these would need to be updated for actual testing)
    args.gpus = [0]
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/val_corruptions'
    args.workers = 4
    args.verbose = True
    
    return args


def run_single_test(config_name, test_config, base_args, corruptions, output_dir):
    """Run a single test configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running test: {config_name}")
    logger.info(f"Description: {test_config.description}")
    logger.info(f"{'='*60}")
    
    # Create a copy of base args and apply overrides
    args = argparse.Namespace(**vars(base_args))
    for key, value in test_config.args_override.items():
        setattr(args, key, value)
    
    # Create results directory for this configuration
    config_result_dir = output_dir / config_name
    config_result_dir.mkdir(exist_ok=True)
    
    # Results storage
    results = {}
    
    # Create a single results file for all corruptions
    results_file = config_result_dir / f'{config_name}_results.txt'
    
    with open(results_file, 'w') as f_write:
        f_write.write(f'CGO Test Results - Configuration: {config_name}\n')
        f_write.write(f'Description: {test_config.description}\n')
        f_write.write(f'Timestamp: {datetime.now().isoformat()}\n')
        f_write.write('\n' + '='*50 + '\n')
        
        # Test each corruption
        for corruption in corruptions:
            logger.info(f"\nTesting corruption: {corruption}")
            
            # Set corruption-specific parameters
            args.corruptions = corruption
            args.val_vid_list = f'/scratch/project_465001897/datasets/ucf/list_video_perturbations/{corruption}.txt'
            args.result_dir = str(config_result_dir / f'{config_name}_{corruption}')
            
            try:
                # Clear GPU memory
                torch.cuda.empty_cache()
                
                # Run evaluation
                epoch_result_list, _ = eval(args=args)
                
                # Store results
                results[corruption] = epoch_result_list
                
                # Write results
                f_write.write(f'\n{corruption}:\n')
                f_write.write(' '.join([str(round(float(x), 3)) for x in epoch_result_list]) + '\n')
                f_write.flush()
                
                logger.info(f"Completed {corruption}: {epoch_result_list}")
                
            except Exception as e:
                logger.error(f"Error testing {corruption}: {str(e)}")
                results[corruption] = [0.0]  # Default to 0 on error
                f_write.write(f'\n{corruption}: ERROR - {str(e)}\n')
                f_write.flush()
    
    # Save detailed results as JSON
    json_results_file = config_result_dir / f'{config_name}_results.json'
    with open(json_results_file, 'w') as f:
        json.dump({
            'config_name': config_name,
            'description': test_config.description,
            'args_override': test_config.args_override,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file} and {json_results_file}")
    return results


def run_comparison_analysis(results_dict, output_dir):
    """Run comparison analysis across all configurations."""
    logger.info(f"\n{'='*60}")
    logger.info("Running Comparison Analysis")
    logger.info(f"{'='*60}")
    
    # Create comparison results file
    comparison_file = output_dir / 'comparison_analysis.txt'
    
    with open(comparison_file, 'w') as f:
        f.write('CGO vs Baseline Comparison Analysis\n')
        f.write(f'Timestamp: {datetime.now().isoformat()}\n')
        f.write('='*50 + '\n\n')
        
        # Get all corruptions tested
        all_corruptions = set()
        for config_results in results_dict.values():
            all_corruptions.update(config_results.keys())
        
        # Compare each configuration against baseline
        if 'baseline' in results_dict:
            baseline_results = results_dict['baseline']
            
            f.write('Performance Comparison (vs Baseline):\n')
            f.write('-' * 40 + '\n')
            
            for config_name, config_results in results_dict.items():
                if config_name == 'baseline':
                    continue
                    
                f.write(f'\n{config_name.upper()}:\n')
                
                improvements = []
                for corruption in all_corruptions:
                    if corruption in baseline_results and corruption in config_results:
                        baseline_acc = baseline_results[corruption][-1] if baseline_results[corruption] else 0.0
                        config_acc = config_results[corruption][-1] if config_results[corruption] else 0.0
                        improvement = config_acc - baseline_acc
                        improvements.append(improvement)
                        
                        f.write(f'  {corruption}: {config_acc:.3f} vs {baseline_acc:.3f} '
                               f'({"+" if improvement >= 0 else ""}{improvement:.3f})\n')
                
                if improvements:
                    avg_improvement = np.mean(improvements)
                    f.write(f'  Average Improvement: {"+" if avg_improvement >= 0 else ""}{avg_improvement:.3f}\n')
        
        # Summary statistics
        f.write('\n' + '='*50 + '\n')
        f.write('SUMMARY STATISTICS:\n')
        f.write('-' * 20 + '\n')
        
        for config_name, config_results in results_dict.items():
            all_accuracies = []
            for corruption_results in config_results.values():
                if corruption_results:
                    all_accuracies.append(corruption_results[-1])
            
            if all_accuracies:
                f.write(f'\n{config_name.upper()}:\n')
                f.write(f'  Mean Accuracy: {np.mean(all_accuracies):.3f}\n')
                f.write(f'  Std Accuracy: {np.std(all_accuracies):.3f}\n')
                f.write(f'  Min Accuracy: {np.min(all_accuracies):.3f}\n')
                f.write(f'  Max Accuracy: {np.max(all_accuracies):.3f}\n')
    
    logger.info(f"Comparison analysis saved to {comparison_file}")


def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description='Test CGO-enhanced ViTTA')
    parser.add_argument('--config', type=str, default='comparison',
                       help='Test configuration to run')
    parser.add_argument('--output-dir', type=str, default='./cgo_test_results',
                       help='Output directory for test results')
    parser.add_argument('--corruptions', nargs='+', 
                       default=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'random'],
                       help='Corruptions to test')
    
    # Parse only known arguments to avoid conflicts with ViTTA's parser
    test_args, _ = parser.parse_known_args()
    
    # Setup output directory
    output_dir = Path(test_args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get test configurations
    configs = get_test_configurations()
    
    # Setup base arguments
    base_args = setup_base_args()
    
    logger.info(f"Starting CGO-ViTTA Test Suite")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Testing corruptions: {test_args.corruptions}")
    
    # Run tests
    results_dict = {}
    
    if test_args.config == 'comparison':
        # Run all configurations for comparison
        for config_name, test_config in configs.items():
            try:
                results = run_single_test(config_name, test_config, base_args, 
                                        test_args.corruptions, output_dir)
                results_dict[config_name] = results
            except Exception as e:
                logger.error(f"Failed to run configuration {config_name}: {str(e)}")
        
        # Run comparison analysis
        if len(results_dict) > 1:
            run_comparison_analysis(results_dict, output_dir)
            
    elif test_args.config in configs:
        # Run single configuration
        test_config = configs[test_args.config]
        results = run_single_test(test_args.config, test_config, base_args,
                                test_args.corruptions, output_dir)
        results_dict[test_args.config] = results
        
    else:
        logger.error(f"Unknown configuration: {test_args.config}")
        logger.info(f"Available configurations: {list(configs.keys()) + ['comparison']}")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("Test Suite Completed Successfully!")
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
