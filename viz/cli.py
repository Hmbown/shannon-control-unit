"""
SCU Visualization CLI
Command-line interface for generating investor-grade figures.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import plots


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate professional visualizations for SCU training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots
  python -m viz.cli --root outputs --out assets/figures --which all
  
  # Generate specific plots
  python -m viz.cli --root outputs --which s_curve,lambda,validation
  
  # List available plots
  python -m viz.cli --list
        """
    )
    
    parser.add_argument(
        '--root', 
        type=str, 
        default='outputs',
        help='Root directory containing PI and fixed_lambda_* subdirs (default: outputs)'
    )
    
    parser.add_argument(
        '--out', 
        type=str, 
        default='assets/figures',
        help='Output directory for figures (default: assets/figures)'
    )
    
    parser.add_argument(
        '--which',
        type=str,
        default='all',
        help='Comma-separated list of plots to generate, or "all" (default: all)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available plot types and exit'
    )
    
    parser.add_argument(
        '--eval-json',
        type=str,
        help='Path to evaluation JSON for validation plot'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        help='Specific run ID to use for single-run plots'
    )
    
    args = parser.parse_args()
    
    # Available plots
    available_plots = {
        's_curve': 'S-tracking curve with target band',
        'lambda': 'Lambda evolution on log scale', 
        'validation': 'Validation results table (Base vs SCU)',
        'grid': 'Fixed-lambda grid vs PI control',
        'sweep': 'Target sweep plots (2 figures)',
        'pulse': 'Pulse test showing negative plant gain',
        'data_bpt': 'Data BPT learning curve',
        'param_bpt': 'Parameter BPT curve',
        'step_time': 'Step time distribution histogram'
    }
    
    if args.list:
        print("\nAvailable plot types:")
        print("-" * 40)
        for name, desc in available_plots.items():
            print(f"  {name:12} - {desc}")
        print("\nUse --which to select specific plots or 'all' for everything")
        return 0
    
    # Parse which plots to generate
    if args.which == 'all':
        plots_to_generate = list(available_plots.keys())
    else:
        plots_to_generate = [p.strip() for p in args.which.split(',')]
        # Validate plot names
        invalid = [p for p in plots_to_generate if p not in available_plots]
        if invalid:
            print(f"Error: Unknown plot types: {', '.join(invalid)}")
            print("Use --list to see available plots")
            return 1
    
    print(f"\n{'='*60}")
    print("SCU Visualization Generator")
    print(f"{'='*60}")
    print(f"Root directory: {args.root}")
    print(f"Output directory: {args.out}")
    print(f"Generating: {', '.join(plots_to_generate)}")
    print(f"{'='*60}\n")
    
    # Load all runs
    print("Loading training runs...")
    runs = plots.load_runs(args.root)
    
    n_pi = sum(len(r) for r in runs.get('PI', {}).values()) if isinstance(runs.get('PI'), dict) else 0
    n_fixed = sum(len(runs.get('fixed_lambda', {}).get(k, {})) 
                  for k in runs.get('fixed_lambda', {}))
    
    print(f"Found {n_pi} PI runs and {n_fixed} fixed-λ runs\n")
    
    # Generate plots
    generated = []
    skipped = []
    
    # Single-run plots (use first PI run or specified run)
    single_run_plots = ['s_curve', 'lambda', 'pulse', 'data_bpt', 'param_bpt']
    
    run = None
    if any(p in plots_to_generate for p in single_run_plots):
        # Find a run to use
        if args.run_id:
            # Look for specific run
            for run_type in ['PI', 'fixed_lambda']:
                if run_type == 'PI' and args.run_id in runs.get('PI', {}):
                    run = runs['PI'][args.run_id]
                    break
                elif run_type == 'fixed_lambda':
                    for lambda_val, lambda_runs in runs.get('fixed_lambda', {}).items():
                        if args.run_id in lambda_runs:
                            run = lambda_runs[args.run_id]
                            break
        else:
            # Use first PI run if available
            if runs.get('PI'):
                run_id = list(runs['PI'].keys())[0]
                run = runs['PI'][run_id]
                print(f"Using PI run: {run_id}")
    
    # Generate each plot
    for plot_name in plots_to_generate:
        print(f"\nGenerating {plot_name}...")
        
        try:
            if plot_name == 's_curve':
                if run:
                    fig = plots.plot_s_curve(run, args.out)
                    generated.append('s_curve.png/svg')
                else:
                    print("  ⚠️  No PI runs found for S-curve")
                    skipped.append(plot_name)
                    
            elif plot_name == 'lambda':
                if run:
                    fig = plots.plot_lambda_curve(run, args.out)
                    generated.append('lambda_curve.png/svg')
                else:
                    print("  ⚠️  No PI runs found for lambda curve")
                    skipped.append(plot_name)
                    
            elif plot_name == 'validation':
                eval_path = args.eval_json or 'outputs/eval/latest.json'
                if Path(eval_path).exists():
                    fig = plots.plot_validation(eval_path, args.out)
                    generated.append('validation_delta.png/svg')
                else:
                    print(f"  ⚠️  Evaluation file not found: {eval_path}")
                    skipped.append(plot_name)
                    
            elif plot_name == 'grid':
                if runs.get('fixed_lambda') or runs.get('PI'):
                    fig = plots.plot_grid_vs_pi(runs, args.out)
                    generated.append('grid_vs_pi.png/svg')
                else:
                    print("  ⚠️  No fixed-lambda or PI runs found")
                    skipped.append(plot_name)
                    
            elif plot_name == 'sweep':
                if runs.get('PI'):
                    fig1, fig2 = plots.plot_sweep(runs, args.out)
                    generated.extend(['sweep_target_vs_achieved.png/svg',
                                    'sweep_target_vs_valbpt.png/svg'])
                else:
                    print("  ⚠️  No PI runs found for sweep plots")
                    skipped.append(plot_name)
                    
            elif plot_name == 'pulse':
                if run:
                    fig = plots.plot_pulse(run, args.out)
                    generated.append('pulse_test.png/svg')
                else:
                    print("  ⚠️  No runs found for pulse test")
                    skipped.append(plot_name)
                    
            elif plot_name == 'data_bpt':
                if run:
                    fig = plots.plot_data_curve(run, args.out)
                    generated.append('data_bpt_curve.png/svg')
                else:
                    print("  ⚠️  No runs found for data BPT curve")
                    skipped.append(plot_name)
                    
            elif plot_name == 'param_bpt':
                if run:
                    fig = plots.plot_param_curve(run, args.out)
                    generated.append('param_bpt_curve.png/svg')
                else:
                    print("  ⚠️  No runs found for param BPT curve")
                    skipped.append(plot_name)
                    
            elif plot_name == 'step_time':
                if runs.get('PI') or runs.get('fixed_lambda'):
                    fig = plots.plot_step_time(runs, args.out)
                    generated.append('step_time_hist.png/svg')
                else:
                    print("  ⚠️  No runs found for timing analysis")
                    skipped.append(plot_name)
                    
            print(f"  ✓ Generated {plot_name}")
            
        except Exception as e:
            print(f"  ✗ Error generating {plot_name}: {e}")
            skipped.append(plot_name)
    
    # Generate captions
    print("\nGenerating captions.json...")
    plots.generate_captions(args.out)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Generated {len(generated)} figures:")
    for fig in generated:
        print(f"  - {fig}")
    
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} plots:")
        for plot in skipped:
            print(f"  - {plot}")
    
    print(f"\nOutput directory: {args.out}/")
    print(f"{'='*60}\n")
    
    return 0 if not skipped else 1


if __name__ == '__main__':
    sys.exit(main())