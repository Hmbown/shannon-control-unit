# SCU Visualization Toolkit

Professional-grade plotting toolkit for Shannon Control Unit training logs, designed to generate investor-ready figures for websites and Hugging Face model cards.

## Features

- **10 plot types** covering control dynamics, validation results, and ablations
- **Retina-ready** exports (2000×1200px PNG @ 200 DPI)
- **Vector graphics** (SVG) for perfect scaling
- **Automatic captions** with alt-text for accessibility
- **Graceful handling** of missing data with clear warnings
- **Bootstrap CI** computation for statistical significance

## Installation

```bash
# Dependencies (should already be installed)
pip install matplotlib numpy pandas
```

## Quick Start

```bash
# Generate all plots from training logs
python -m viz.cli --root outputs --out assets/figures --which all

# Generate specific plots only
python -m viz.cli --root outputs --which s_curve,lambda,validation

# List available plot types
python -m viz.cli --list
```

## Available Plots

| Plot ID | Description | Output Files |
|---------|-------------|--------------|
| `s_curve` | S(t) tracking with target band | s_curve.png/svg |
| `lambda` | λ(t) evolution on log scale | lambda_curve.png/svg |
| `validation` | Base vs SCU comparison table | validation_delta.png/svg |
| `grid` | Fixed-λ grid vs PI control | grid_vs_pi.png/svg |
| `sweep` | Target sweep (2 plots) | sweep_target_vs_*.png/svg |
| `pulse` | Plant sign test (S vs ln λ) | pulse_test.png/svg |
| `data_bpt` | Data loss learning curve | data_bpt_curve.png/svg |
| `param_bpt` | Parameter cost curve | param_bpt_curve.png/svg |
| `step_time` | Training speed histogram | step_time_hist.png/svg |

## Expected Directory Structure

```
outputs/
├── PI/
│   └── <run_id>/
│       ├── train_log.csv      # Required: step,data_bpt,param_bpt,S,lambda
│       └── metadata.json       # Optional: target_s, kp, ki, etc.
├── fixed_lambda_<value>/
│   └── <run_id>/
│       ├── train_log.csv
│       └── metadata.json
└── eval/
    └── *.json                  # Validation results
```

## Output Structure

```
assets/figures/
├── s_curve.png                 # High-res PNG (200 DPI)
├── s_curve.svg                 # Vector graphics
├── lambda_curve.png
├── lambda_curve.svg
├── validation_delta.png
├── validation_delta.svg
├── grid_vs_pi.png
├── grid_vs_pi.svg
├── sweep_target_vs_achieved.png
├── sweep_target_vs_achieved.svg
├── sweep_target_vs_valbpt.png
├── sweep_target_vs_valbpt.svg
├── pulse_test.png
├── pulse_test.svg
├── data_bpt_curve.png         # Optional
├── data_bpt_curve.svg         # Optional
├── param_bpt_curve.png        # Optional
├── param_bpt_curve.svg        # Optional
├── step_time_hist.png         # Optional
├── step_time_hist.svg         # Optional
└── captions.json               # Alt-text and captions for all figures
```

## CLI Options

```bash
python -m viz.cli [OPTIONS]

Options:
  --root DIR        Root directory with training logs (default: outputs)
  --out DIR         Output directory for figures (default: assets/figures)
  --which PLOTS     Comma-separated list or 'all' (default: all)
  --eval-json FILE  Path to evaluation JSON for validation plot
  --run-id ID       Specific run ID for single-run plots
  --list           List available plot types and exit
```

## Examples

### Generate Core Plots Only
```bash
python -m viz.cli --which s_curve,lambda,validation,grid
```

### Use Specific Evaluation File
```bash
python -m viz.cli --which validation --eval-json results/best_run.json
```

### Generate Plots for Specific Run
```bash
python -m viz.cli --which s_curve,lambda --run-id run_20250903_123456
```

## Programmatic Usage

```python
from viz import plots

# Load all runs
runs = plots.load_runs('outputs')

# Generate S-curve for first PI run
if runs['PI']:
    run_id = list(runs['PI'].keys())[0]
    run = runs['PI'][run_id]
    fig = plots.plot_s_curve(run, out_dir='figures')

# Generate validation comparison
plots.plot_validation('outputs/eval/latest.json', out_dir='figures')

# Generate captions file
plots.generate_captions('figures')
```

## Key Computations

### Settling Time
First step where S stays within target band for ≥25 consecutive steps.

### Steady-State Error (SSE)
Mean absolute deviation from target over last 20% of training.

### Bootstrap CI
10,000 bootstrap samples for 95% confidence interval on ΔBPT.

### Perplexity Reduction
Calculated as `(1 - 2^(-ΔBPT)) × 100%`

## Troubleshooting

### Missing Data Warnings
The toolkit gracefully handles missing data:
- Skips plots with clear console messages
- Generates placeholder figures when appropriate
- Returns non-zero exit code if any plots were skipped

### Column Requirements
Training logs must have these columns:
- `step`: Training step number
- `data_bpt`: Data bits per token
- `param_bpt`: Parameter bits per token  
- `S`: Compression ratio (param_bpt / total_bpt)
- `lambda`: Regularization strength
- `wall_time_s`: (Optional) Wall clock time

### Performance Tips
- PNG generation takes ~1-2 seconds per plot
- SVG generation is nearly instant
- Bootstrap CI adds ~1 second for 10K samples
- Total time for all plots: ~15-20 seconds

## Citation

If you use these visualizations in publications:

```bibtex
@software{scu_viz2025,
  title={SCU Visualization Toolkit},
  author={Shannon Labs},
  year={2025},
  url={https://github.com/hunterbown/shannon-control-unit}
}
```

## License

Apache 2.0 - See LICENSE file for details.