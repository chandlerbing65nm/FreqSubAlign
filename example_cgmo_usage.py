#!/usr/bin/env python3
"""
Example usage of Confidence-Gated Meta-Optimizer (CGMO) in ViTTA
Shows how CGMO can surpass vanilla TTA accuracy through intelligent confidence-weighted updates.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_example():
    """Run CGMO example with configurations designed to surpass vanilla TTA."""
    
    # Base ViTTA directory
    base_dir = Path("/users/doloriel/work/Repo/ViTTA")
    
    # Example configurations that should surpass vanilla TTA
    configs = [
        {
            "name": "vanilla_tta",
            "description": "Vanilla TTA baseline",
            "cmd": [
                sys.executable, str(base_dir / "main_tta.py"),
                "--dataset", "kinetics400",
                "--data_root", "/data/kinetics400",
                "--model", "vit_base_patch16_224",
                "--pretrained", "True",
                "--batch_size", "8",
                "--lr", "5e-5",
                "--epochs", "1",
                "--tta_steps", "10",
                "--corruptions", "gaussian_noise",
                "--severity", "3",
                "--output_dir", str(base_dir / "example_results" / "vanilla_tta"),
                "--use_cgo", "False",
                "--use_cgmo", "False"
            ]
        },
        {
            "name": "cgmo_surpass",
            "description": "CGMO configuration designed to surpass vanilla TTA",
            "cmd": [
                sys.executable, str(base_dir / "main_tta.py"),
                "--dataset", "kinetics400",
                "--data_root", "/data/kinetics400",
                "--model", "vit_base_patch16_224",
                "--pretrained", "True",
                "--batch_size", "8",
                "--lr", "5e-5",
                "--epochs", "1",
                "--tta_steps", "10",
                "--corruptions", "gaussian_noise",
                "--severity", "3",
                "--output_dir", str(base_dir / "example_results" / "cgmo_surpass"),
                "--use_cgmo", "True",
                "--cgmo_confidence_threshold", "0.4",
                "--cgmo_min_lr_scale", "0.3",
                "--cgmo_max_lr_scale", "2.2",
                "--cgmo_confidence_power", "2.5",
                "--cgmo_enable_momentum_correction", "True"
            ]
        },
        {
            "name": "cgmo_adaptive",
            "description": "CGMO with adaptive threshold for optimal performance",
            "cmd": [
                sys.executable, str(base_dir / "main_tta.py"),
                "--dataset", "kinetics400",
                "--data_root", "/data/kinetics400",
                "--model", "vit_base_patch16_224",
                "--pretrained", "True",
                "--batch_size", "8",
                "--lr", "5e-5",
                "--epochs", "1",
                "--tta_steps", "10",
                "--corruptions", "gaussian_noise",
                "--severity", "3",
                "--output_dir", str(base_dir / "example_results" / "cgmo_adaptive"),
                "--use_cgmo", "True",
                "--cgmo_confidence_threshold", "0.5",
                "--cgmo_min_lr_scale", "0.1",
                "--cgmo_max_lr_scale", "2.0",
                "--cgmo_confidence_power", "2.0",
                "--cgmo_adaptive_threshold", "True",
                "--cgmo_enable_momentum_correction", "True"
            ]
        }
    ]
    
    print("="*70)
    print("CGMO USAGE EXAMPLE")
    print("="*70)
    print("This example demonstrates how CGMO can surpass vanilla TTA accuracy")
    print("through intelligent confidence-weighted updates instead of binary gating.")
    print()
    
    # Create results directory
    (base_dir / "example_results").mkdir(exist_ok=True)
    
    results = {}
    
    for config in configs:
        print(f"Running: {config['description']}")
        print(f"Command: {' '.join(config['cmd'])}")
        print("-" * 50)
        
        try:
            # Run the test
            result = subprocess.run(config['cmd'], capture_output=True, text=True, cwd=base_dir)
            
            # Parse and store results
            accuracy = 0.0
            for line in result.stdout.split('\n'):
                if "Test Accuracy:" in line or "Accuracy:" in line:
                    try:
                        if "%" in line:
                            acc_str = line.split(":")[-1].replace("%", "").strip()
                            accuracy = float(acc_str)
                        else:
                            acc_str = line.split(":")[-1].strip()
                            accuracy = float(acc_str)
                        break
                    except:
                        pass
            
            results[config['name']] = {
                "accuracy": accuracy,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            print(f"✓ Completed: {accuracy:.2f}% accuracy")
            if result.returncode != 0:
                print(f"⚠ Warning: Return code {result.returncode}")
                print(result.stderr[-200:])  # Last 200 chars of stderr
            print()
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[config['name']] = {
                "accuracy": 0.0,
                "success": False,
                "error": str(e)
            }
    
    # Display results
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    vanilla_acc = results.get("vanilla_tta", {}).get("accuracy", 0.0)
    
    for name, data in results.items():
        if data["success"] and name != "vanilla_tta":
            improvement = data["accuracy"] - vanilla_acc
            print(f"{name:20}: {data['accuracy']:6.2f}% ({improvement:+6.2f}% vs vanilla)")
        elif name == "vanilla_tta":
            print(f"{name:20}: {data['accuracy']:6.2f}% (baseline)")
        else:
            print(f"{name:20}: FAILED")
    
    print()
    print("KEY INSIGHTS:")
    print("- CGMO uses confidence to scale learning rates (0.1x to 2.2x)")
    print("- High-confidence samples get stronger updates")
    print("- Low-confidence samples get gentler updates (not skipped)")
    print("- This approach can surpass vanilla TTA accuracy")
    print()
    
    # Save results
    import json
    with open(base_dir / "example_results" / "cgmo_example_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to:", base_dir / "example_results" / "cgmo_example_results.json")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="CGMO Usage Example")
    parser.add_argument("--dataset", default="kinetics400", help="Dataset to use")
    parser.add_argument("--corruption", default="gaussian_noise", help="Corruption type")
    parser.add_argument("--severity", type=int, default=3, help="Corruption severity")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick CGMO test...")
        # Modify the example for quick testing
        for config in configs:
            config["cmd"][config["cmd"].index("--tta_steps") + 1] = "3"  # Reduce steps
            config["cmd"][config["cmd"].index("--epochs") + 1] = "1"   # Single epoch
    
    run_example()


if __name__ == "__main__":
    main()
