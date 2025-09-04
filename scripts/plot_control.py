#!/usr/bin/env python3
"""Plot S(t) and λ(t) control curves from training CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    # Read CSV
    df = pd.read_csv(args.csv)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot S(t) with target band
    ax1.plot(df['step'], df['S'] * 100, 'b-', linewidth=2, label='S(t)')
    
    # Add target band
    if args.target:
        target_pct = args.target * 100
        deadband_pct = args.deadband * 100
        ax1.axhspan(
            target_pct - deadband_pct,
            target_pct + deadband_pct,
            alpha=0.2, color='green',
            label=f'Target {target_pct:.1f}% ± {deadband_pct:.2f}pp'
        )
        ax1.axhline(target_pct, color='green', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('S (%)')
    ax1.set_title('Compression Ratio S = ParamBPT / (DataBPT + ParamBPT)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot λ(t) on log scale
    ax2.semilogy(df['step'], df['lambda'], 'r-', linewidth=2, label='λ(t)')
    
    # Add bounds
    if args.lambda_min:
        ax2.axhline(args.lambda_min, color='gray', linestyle=':', alpha=0.5, label=f'λ_min={args.lambda_min}')
    if args.lambda_max:
        ax2.axhline(args.lambda_max, color='gray', linestyle=':', alpha=0.5, label=f'λ_max={args.lambda_max}')
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('λ (log scale)')
    ax2.set_title('Regularization Strength λ(t)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figures
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined plot
    combined_path = output_dir / f"control_curves_{args.tag}.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot to {combined_path}")
    
    # Save individual plots
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.plot(df['step'], df['S'] * 100, 'b-', linewidth=2)
    if args.target:
        target_pct = args.target * 100
        deadband_pct = args.deadband * 100
        ax_s.axhspan(
            target_pct - deadband_pct,
            target_pct + deadband_pct,
            alpha=0.2, color='green'
        )
        ax_s.axhline(target_pct, color='green', linestyle='--', alpha=0.5)
    ax_s.set_xlabel('Training Step')
    ax_s.set_ylabel('S (%)')
    ax_s.set_title('Compression Ratio S(t)')
    ax_s.grid(True, alpha=0.3)
    
    s_path = output_dir / f"s_curve_{args.tag}.png"
    fig_s.savefig(s_path, dpi=150, bbox_inches='tight')
    print(f"Saved S curve to {s_path}")
    
    # Lambda plot
    fig_l, ax_l = plt.subplots(figsize=(10, 5))
    ax_l.semilogy(df['step'], df['lambda'], 'r-', linewidth=2)
    if args.lambda_min:
        ax_l.axhline(args.lambda_min, color='gray', linestyle=':', alpha=0.5)
    if args.lambda_max:
        ax_l.axhline(args.lambda_max, color='gray', linestyle=':', alpha=0.5)
    ax_l.set_xlabel('Training Step')
    ax_l.set_ylabel('λ (log scale)')
    ax_l.set_title('Regularization Strength λ(t)')
    ax_l.grid(True, alpha=0.3)
    
    lambda_path = output_dir / f"lambda_{args.tag}.png"
    fig_l.savefig(lambda_path, dpi=150, bbox_inches='tight')
    print(f"Saved λ curve to {lambda_path}")
    
    # Print statistics
    print("\nControl Statistics:")
    print(f"Final S: {df['S'].iloc[-1]:.1%}")
    print(f"Final λ: {df['lambda'].iloc[-1]:.3f}")
    print(f"S range: [{df['S'].min():.1%}, {df['S'].max():.1%}]")
    print(f"λ range: [{df['lambda'].min():.3f}, {df['lambda'].max():.3f}]")
    
    # Check if S converged to target
    if args.target:
        final_s = df['S'].iloc[-1]
        error = abs(final_s - args.target)
        if error <= args.deadband:
            print(f"✓ S converged to target within deadband")
        else:
            print(f"✗ S did not converge (error: {error:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot control curves from CSV")
    
    parser.add_argument("--csv", required=True,
                       help="Path to training CSV file")
    parser.add_argument("--tag", default="scu",
                       help="Tag for output filenames")
    parser.add_argument("--output_dir", default="figures",
                       help="Output directory for plots")
    parser.add_argument("--target", type=float, default=0.01,
                       help="Target S value for reference")
    parser.add_argument("--deadband", type=float, default=0.002,
                       help="Deadband for target")
    parser.add_argument("--lambda_min", type=float, default=1e-4,
                       help="Minimum lambda for reference")
    parser.add_argument("--lambda_max", type=float, default=10.0,
                       help="Maximum lambda for reference")
    
    args = parser.parse_args()
    main(args)