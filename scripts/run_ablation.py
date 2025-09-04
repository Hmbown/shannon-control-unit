#!/usr/bin/env python3
"""Run ablation studies for SCU."""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_fixed_lambda_grid(args):
    """Run training with fixed lambda values."""
    results = []
    lambdas = [0.3, 1.0, 3.0, 10.0]
    
    print("Running fixed-λ grid search...")
    print(f"Lambda values: {lambdas}")
    
    for lmbda in lambdas:
        print(f"\n{'='*60}")
        print(f"Training with fixed λ = {lmbda}")
        print('='*60)
        
        output_dir = f"ablations/fixed_lambda_{lmbda}"
        
        # Run training with fixed lambda (no PI control)
        cmd = [
            sys.executable, "scripts/train_scu.py",
            "--base_model", args.base_model,
            "--adapter_out", output_dir,
            "--steps", str(args.steps),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--lambda_init", str(lmbda),
            "--kp", "0.0",  # Disable PI control
            "--ki", "0.0",
            "--log_csv", f"{output_dir}/training.csv",
            "--train_data", args.train_data
        ]
        
        subprocess.run(cmd, check=True)
        
        # Evaluate
        eval_cmd = [
            sys.executable, "scripts/eval_bpt.py",
            "--base_model", args.base_model,
            "--adapter_path", output_dir,
            "--texts", args.val_data,
            "--output", f"{output_dir}/eval_results.json"
        ]
        
        subprocess.run(eval_cmd, check=True)
        
        # Load results
        with open(f"{output_dir}/eval_results.json") as f:
            eval_results = json.load(f)
        
        results.append({
            'lambda': lmbda,
            'val_bpt': eval_results['scu_bpt'],
            'delta_bpt': eval_results['delta_bpt']
        })
    
    # Compare with PI control
    print(f"\n{'='*60}")
    print("Training with PI control (baseline)")
    print('='*60)
    
    output_dir = "ablations/pi_control"
    
    cmd = [
        sys.executable, "scripts/train_scu.py",
        "--base_model", args.base_model,
        "--adapter_out", output_dir,
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--target_s", "0.01",
        "--kp", "0.8",
        "--ki", "0.15",
        "--log_csv", f"{output_dir}/training.csv",
        "--train_data", args.train_data
    ]
    
    subprocess.run(cmd, check=True)
    
    eval_cmd = [
        sys.executable, "scripts/eval_bpt.py",
        "--base_model", args.base_model,
        "--adapter_path", output_dir,
        "--texts", args.val_data,
        "--output", f"{output_dir}/eval_results.json"
    ]
    
    subprocess.run(eval_cmd, check=True)
    
    with open(f"{output_dir}/eval_results.json") as f:
        pi_results = json.load(f)
    
    # Create markdown table
    markdown = "## Fixed-λ Grid Search Results\n\n"
    markdown += "| Method | Val BPT | ΔBPT | Notes |\n"
    markdown += "|--------|---------|------|-------|\n"
    
    for r in results:
        markdown += f"| Fixed λ={r['lambda']} | {r['val_bpt']:.3f} | {r['delta_bpt']:.3f} | "
        if r['val_bpt'] == min(results, key=lambda x: x['val_bpt'])['val_bpt']:
            markdown += "Best fixed |"
        else:
            markdown += "|"
        markdown += "\n"
    
    markdown += f"| **PI Control** | **{pi_results['scu_bpt']:.3f}** | **{pi_results['delta_bpt']:.3f}** | "
    
    if pi_results['scu_bpt'] < min(results, key=lambda x: x['val_bpt'])['val_bpt']:
        markdown += "**Winner** |\n"
    else:
        markdown += "|\n"
    
    return markdown


def run_target_sweep(args):
    """Run training with different target S values."""
    results = []
    targets = [0.005, 0.01, 0.02, 0.03]
    
    print("Running target S sweep...")
    print(f"Target values: {[f'{t:.1%}' for t in targets]}")
    
    for target in targets:
        print(f"\n{'='*60}")
        print(f"Training with target S = {target:.1%}")
        print('='*60)
        
        output_dir = f"ablations/target_s_{target:.3f}"
        
        cmd = [
            sys.executable, "scripts/train_scu.py",
            "--base_model", args.base_model,
            "--adapter_out", output_dir,
            "--steps", str(args.steps),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--target_s", str(target),
            "--kp", "0.8",
            "--ki", "0.15",
            "--log_csv", f"{output_dir}/training.csv",
            "--train_data", args.train_data
        ]
        
        subprocess.run(cmd, check=True)
        
        # Evaluate
        eval_cmd = [
            sys.executable, "scripts/eval_bpt.py",
            "--base_model", args.base_model,
            "--adapter_path", output_dir,
            "--texts", args.val_data,
            "--output", f"{output_dir}/eval_results.json"
        ]
        
        subprocess.run(eval_cmd, check=True)
        
        # Load results
        with open(f"{output_dir}/eval_results.json") as f:
            eval_results = json.load(f)
        
        with open(f"{output_dir}/metadata.json") as f:
            metadata = json.load(f)
        
        results.append({
            'target_s': target,
            'final_s': metadata['final_s'],
            'final_lambda': metadata['final_lambda'],
            'val_bpt': eval_results['scu_bpt']
        })
    
    # Create markdown table
    markdown = "## Target S Sweep Results\n\n"
    markdown += "| Target S* | Final S | Final λ | Val BPT |\n"
    markdown += "|-----------|---------|---------|----------|\n"
    
    best_bpt = min(results, key=lambda x: x['val_bpt'])['val_bpt']
    
    for r in results:
        bold = "**" if r['val_bpt'] == best_bpt else ""
        markdown += f"| {bold}{r['target_s']:.1%}{bold} | {r['final_s']:.2%} | {r['final_lambda']:.2f} | {bold}{r['val_bpt']:.3f}{bold} |\n"
    
    markdown += "\nDistinct S plateaus confirm control works; optimal S* depends on task.\n"
    
    return markdown


def main(args):
    # Create ablations directory
    Path("ablations").mkdir(exist_ok=True)
    
    markdown = f"# SCU Ablation Results\n\n"
    markdown += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    markdown += f"Base Model: {args.base_model}\n"
    markdown += f"Steps: {args.steps}\n\n"
    
    if args.mode == "fixed-lambda":
        markdown += run_fixed_lambda_grid(args)
    elif args.mode == "target-sweep":
        markdown += run_target_sweep(args)
    elif args.mode == "all":
        markdown += run_fixed_lambda_grid(args)
        markdown += "\n\n"
        markdown += run_target_sweep(args)
    
    # Save markdown
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(markdown)
    
    print(f"\n{'='*60}")
    print(f"Ablation results saved to {output_path}")
    print('='*60)
    print(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCU ablations")
    
    parser.add_argument("--mode", choices=["fixed-lambda", "target-sweep", "all"],
                       default="fixed-lambda",
                       help="Ablation mode")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B",
                       help="Base model")
    parser.add_argument("--steps", type=int, default=200,
                       help="Training steps per run")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--train_data", default="data/train.txt",
                       help="Training data")
    parser.add_argument("--val_data", default="data/val.txt",
                       help="Validation data")
    parser.add_argument("--output", default="figures/ablations.md",
                       help="Output markdown file")
    
    args = parser.parse_args()
    main(args)