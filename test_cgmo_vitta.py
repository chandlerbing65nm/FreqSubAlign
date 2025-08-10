#!/usr/bin/env python3
"""
Comprehensive test script for Confidence-Gated Meta-Optimizer (CGMO) in ViTTA
Demonstrates how CGMO can surpass vanilla TTA accuracy through intelligent confidence-weighted updates.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
import subprocess
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

def setup_logging():
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cgmo_test.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class CGMOTester:
    """Test suite for CGMO-enhanced ViTTA."""
    
    def __init__(self, base_dir: str = "/users/doloriel/work/Repo/ViTTA"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "cgmo_test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configurations designed to surpass vanilla TTA
        self.test_configs = [
            {
                "name": "vanilla_tta",
                "description": "Vanilla TTA baseline",
                "args": ["--use_cgo", "False", "--use_cgmo", "False"]
            },
            {
                "name": "cgmo_conservative",
                "description": "CGMO with conservative settings",
                "args": [
                    "--use_cgmo", "True",
                    "--cgmo_confidence_threshold", "0.5",
                    "--cgmo_min_lr_scale", "0.5",
                    "--cgmo_max_lr_scale", "1.5",
                    "--cgmo_confidence_power", "1.5"
                ]
            },
            {
                "name": "cgmo_aggressive",
                "description": "CGMO with aggressive settings",
                "args": [
                    "--use_cgmo", "True",
                    "--cgmo_confidence_threshold", "0.3",
                    "--cgmo_min_lr_scale", "0.2",
                    "--cgmo_max_lr_scale", "2.5",
                    "--cgmo_confidence_power", "2.5"
                ]
            },
            {
                "name": "cgmo_adaptive",
                "description": "CGMO with adaptive threshold",
                "args": [
                    "--use_cgmo", "True",
                    "--cgmo_confidence_threshold", "0.5",
                    "--cgmo_min_lr_scale", "0.1",
                    "--cgmo_max_lr_scale", "2.0",
                    "--cgmo_confidence_power", "2.0",
                    "--cgmo_adaptive_threshold", "True"
                ]
            },
            {
                "name": "cgmo_entropy",
                "description": "CGMO with entropy-based confidence",
                "args": [
                    "--use_cgmo", "True",
                    "--cgmo_confidence_threshold", "0.6",
                    "--cgmo_min_lr_scale", "0.1",
                    "--cgmo_max_lr_scale", "2.0",
                    "--cgmo_confidence_power", "2.0"
                ]
            },
            {
                "name": "cgo_baseline",
                "description": "Original CGO for comparison",
                "args": [
                    "--use_cgo", "True",
                    "--cgo_confidence_threshold", "0.5",
                    "--cgo_confidence_metric", "max_softmax"
                ]
            }
        ]
        
    def create_test_command(self, config: Dict, corruption_type: str = "gaussian_noise", 
                          severity: int = 3, dataset: str = "kinetics400") -> List[str]:
        """Create command for running a test configuration."""
        
        cmd = [
            sys.executable, str(self.base_dir / "main_tta.py"),
            "--dataset", dataset,
            "--data_root", "/data/kinetics400",
            "--model", "vit_base_patch16_224",
            "--pretrained", "True",
            "--batch_size", "8",
            "--lr", "5e-5",
            "--epochs", "1",
            "--tta_steps", "10",
            "--corruptions", corruption_type,
            "--severity", str(severity),
            "--output_dir", str(self.results_dir / f"{config['name']}_{corruption_type}_{severity}"),
            "--log_level", "INFO"
        ]
        
        # Add configuration-specific arguments
        cmd.extend(config["args"])
        
        return cmd
    
    def run_single_test(self, config: Dict, corruption_type: str = "gaussian_noise", 
                       severity: int = 3) -> Dict:
        """Run a single test configuration."""
        
        logger.info(f"Running test: {config['name']} with {corruption_type} severity {severity}")
        
        cmd = self.create_test_command(config, corruption_type, severity)
        
        # Run the test
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            elapsed_time = time.time() - start_time
            
            # Parse results
            result_data = self.parse_test_output(result.stdout, result.stderr)
            result_data.update({
                "config_name": config["name"],
                "corruption": corruption_type,
                "severity": severity,
                "elapsed_time": elapsed_time,
                "success": result.returncode == 0
            })
            
            if result.returncode != 0:
                logger.error(f"Test failed: {config['name']}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                result_data["error"] = result.stderr
            
            return result_data
            
        except Exception as e:
            logger.error(f"Exception running test {config['name']}: {e}")
            return {
                "config_name": config["name"],
                "corruption": corruption_type,
                "severity": severity,
                "success": False,
                "error": str(e)
            }
    
    def parse_test_output(self, stdout: str, stderr: str) -> Dict:
        """Parse accuracy and other metrics from test output."""
        
        result = {
            "accuracy": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "adaptation_stats": {}
        }
        
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse accuracy
            if "Test Accuracy:" in line or "Accuracy:" in line:
                try:
                    # Look for patterns like "Accuracy: 82.34%" or "Test Accuracy: 82.34"
                    if "%" in line:
                        acc_str = line.split(":")[-1].replace("%", "").strip()
                        result["accuracy"] = float(acc_str)
                    else:
                        acc_str = line.split(":")[-1].strip()
                        result["accuracy"] = float(acc_str)
                except:
                    pass
            
            # Parse CGMO/CGO statistics
            if "CGMO Statistics" in line or "CGO Statistics" in line:
                try:
                    # Extract key metrics from statistics line
                    parts = line.split(" - ")
                    for part in parts:
                        if ":" in part:
                            key, value = part.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            try:
                                result["adaptation_stats"][key] = float(value)
                            except:
                                result["adaptation_stats"][key] = value
                except:
                    pass
        
        return result
    
    def run_comprehensive_tests(self, corruption_types: List[str] = None, 
                              severities: List[int] = None) -> pd.DataFrame:
        """Run comprehensive tests across multiple configurations."""
        
        if corruption_types is None:
            corruption_types = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", 
                              "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog"]
        
        if severities is None:
            severities = [3, 5]  # Medium and high severity
        
        results = []
        
        for corruption in corruption_types:
            for severity in severities:
                for config in self.test_configs:
                    result = self.run_single_test(config, corruption, severity)
                    results.append(result)
                    
                    # Save intermediate results
                    self.save_results(results)
                    
                    # Small delay between tests
                    time.sleep(2)
        
        return pd.DataFrame(results)
    
    def save_results(self, results: List[Dict]):
        """Save test results to JSON and CSV."""
        
        # Save as JSON
        with open(self.results_dir / "cgmo_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "cgmo_test_results.csv", index=False)
    
    def generate_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis report."""
        
        report = {
            "summary": {},
            "comparisons": {},
            "recommendations": {}
        }
        
        # Filter successful tests
        successful = results_df[results_df['success'] == True]
        
        if len(successful) == 0:
            report["summary"]["error"] = "No successful tests completed"
            return report
        
        # Calculate average accuracies per configuration
        config_accuracies = successful.groupby('config_name')['accuracy'].agg(['mean', 'std', 'count'])
        
        report["summary"]["config_accuracies"] = config_accuracies.to_dict()
        
        # Compare CGMO vs Vanilla TTA
        vanilla_acc = successful[successful['config_name'] == 'vanilla_tta']['accuracy'].mean()
        
        cgmo_configs = [c for c in successful['config_name'].unique() if 'cgmo' in c]
        
        report["comparisons"]["vanilla_baseline"] = vanilla_acc
        report["comparisons"]["cgmo_improvements"] = {}
        
        for cgmo_config in cgmo_configs:
            cgmo_acc = successful[successful['config_name'] == cgmo_config]['accuracy'].mean()
            improvement = cgmo_acc - vanilla_acc
            report["comparisons"]["cgmo_improvements"][cgmo_config] = {
                "accuracy": cgmo_acc,
                "improvement": improvement,
                "improvement_percent": (improvement / vanilla_acc * 100) if vanilla_acc > 0 else 0
            }
        
        # Find best configuration
        best_config = config_accuracies['mean'].idxmax()
        best_accuracy = config_accuracies.loc[best_config, 'mean']
        
        report["recommendations"]["best_config"] = best_config
        report["recommendations"]["best_accuracy"] = best_accuracy
        report["recommendations"]["improvement_over_vanilla"] = best_accuracy - vanilla_acc
        
        # Save report
        with open(self.results_dir / "cgmo_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, report: Dict):
        """Print human-readable summary."""
        
        print("\n" + "="*60)
        print("CGMO TEST SUMMARY")
        print("="*60)
        
        if "error" in report["summary"]:
            print(f"Error: {report['summary']['error']}")
            return
        
        print("\nConfiguration Accuracies:")
        for config, stats in report["summary"]["config_accuracies"]["mean"].items():
            std = report["summary"]["config_accuracies"]["std"][config]
            count = report["summary"]["config_accuracies"]["count"][config]
            print(f"  {config}: {stats:.2f}% (±{std:.2f}, n={count})")
        
        print(f"\nVanilla TTA Baseline: {report['comparisons']['vanilla_baseline']:.2f}%")
        
        print("\nCGMO Improvements:")
        for config, data in report["comparisons"]["cgmo_improvements"].items():
            print(f"  {config}: {data['accuracy']:.2f}% "
                  f"({data['improvement']:+.2f}% vs vanilla)")
        
        print(f"\nBest Configuration: {report['recommendations']['best_config']}")
        print(f"Best Accuracy: {report['recommendations']['best_accuracy']:.2f}%")
        print(f"Improvement over Vanilla: {report['recommendations']['improvement_over_vanilla']:+.2f}%")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test CGMO-enhanced ViTTA")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with fewer configurations")
    parser.add_argument("--corruption", type=str, default="gaussian_noise",
                       help="Corruption type to test")
    parser.add_argument("--severity", type=int, default=3,
                       help="Corruption severity level")
    
    args = parser.parse_args()
    
    tester = CGMOTester()
    
    if args.quick:
        # Quick test with single corruption/severity
        logger.info("Running quick test...")
        results = []
        for config in tester.test_configs:
            result = tester.run_single_test(config, args.corruption, args.severity)
            results.append(result)
        
        df = pd.DataFrame(results)
        report = tester.generate_report(df)
        tester.print_summary(report)
        
    else:
        # Comprehensive test
        logger.info("Running comprehensive tests...")
        results_df = tester.run_comprehensive_tests(
            corruption_types=[args.corruption],
            severities=[args.severity]
        )
        
        report = tester.generate_report(results_df)
        tester.print_summary(report)


if __name__ == "__main__":
    main()
