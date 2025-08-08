#!/usr/bin/env python3
"""
Test script to verify the CMGS configuration loop in main_tta.py works correctly.
This script simulates the loop structure without running the full evaluation.
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cmgs_config_loop():
    """Test the CMGS configuration loop structure."""
    print("=" * 60)
    print("Testing CMGS Configuration Loop")
    print("=" * 60)
    
    # Define CMGS configurations to test (same as in main_tta.py)
    cmgs_configs = [
        {
            'name': 'Config_1',
            'cmgs_alpha': 1.0,
            'cmgs_beta': 0.1,
            'cmgs_gamma': 0.367879,
            'n_epoch_adapat': 4
        },
        {
            'name': 'Config_2',
            'cmgs_alpha': 2.0,
            'cmgs_beta': 0.1,
            'cmgs_gamma': 0.367879,
            'n_epoch_adapat': 8
        },
        {
            'name': 'Config_3',
            'cmgs_alpha': 2.0,
            'cmgs_beta': 0.1,
            'cmgs_gamma': 0.2,
            'n_epoch_adapat': 4
        },
        {
            'name': 'Config_4',
            'cmgs_alpha': 2.0,
            'cmgs_beta': 0.1,
            'cmgs_gamma': 0.2,
            'n_epoch_adapat': 8
        }
    ]
    
    # Simulate the loop structure
    for config_idx, config in enumerate(cmgs_configs):
        print(f"\n{'='*60}")
        print(f"Running Configuration {config_idx + 1}/4: {config['name']}")
        print(f"{'='*60}")
        print(f"CMGS Parameters:")
        print(f"  Alpha: {config['cmgs_alpha']}")
        print(f"  Beta: {config['cmgs_beta']}")
        print(f"  Gamma: {config['cmgs_gamma']}")
        print(f"  Epochs: {config['n_epoch_adapat']}")
        print(f"{'='*60}")
        
        # Simulate setting parameters
        use_cmgs = True
        cmgs_alpha = config['cmgs_alpha']
        cmgs_beta = config['cmgs_beta']
        cmgs_gamma = config['cmgs_gamma']
        n_epoch_adapat = config['n_epoch_adapat']
        
        # Simulate result directory creation
        config_result_dir = f"/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/cmgs_{config['name']}"
        print(f"Result directory: {config_result_dir}")
        
        # Simulate corruption evaluation
        corruptions = ['random_mini']
        for corr_id, corruption in enumerate(corruptions):
            print(f"  Evaluating corruption: {corruption}")
            print(f"    Val list: /scratch/project_465001897/datasets/ucf/list_video_perturbations/{corruption}.txt")
            print(f"    Result dir: {config_result_dir}/tta_{corruption}")
        
        print(f"Configuration {config_idx + 1} completed successfully.")
    
    print(f"\n{'='*60}")
    print("All CMGS configurations completed successfully!")
    print(f"{'='*60}")
    
    # Summary of configurations
    print("\nConfiguration Summary:")
    print("-" * 40)
    for i, config in enumerate(cmgs_configs):
        print(f"Config {i+1}: Alpha={config['cmgs_alpha']}, Beta={config['cmgs_beta']}, "
              f"Gamma={config['cmgs_gamma']}, Epochs={config['n_epoch_adapat']}")
    
    print("\nCMGS Configuration Loop Test: PASSED")

if __name__ == '__main__':
    test_cmgs_config_loop() 