#!/usr/bin/env python3
"""
Clean up unnecessary files from HuggingFace repo
"""
from huggingface_hub import HfApi, delete_file, list_repo_files

REPO = "hunterbown/shannon-control-unit"
api = HfApi()

# Files to remove from HuggingFace
files_to_remove = [
    # Duplicate/unnecessary in root
    "MODEL_CARD.md",  # Duplicate of README
    "control_plots.png",  # Old plot
    "control_plots_validated.png",  # Old plot  
    "ce_kl_auto_metrics.jsonl",  # Training artifact
    "train_log.csv",  # Training artifact
    
    # Old figures
    "figures/control_plots_validated.png",
    
    # SVG files (keep only PNGs for HF)
    "assets/figures/data_bpt_curve.svg",
    "assets/figures/lambda_curve.svg", 
    "assets/figures/param_bpt_curve.svg",
    "assets/figures/pulse_test.svg",
    "assets/figures/s_curve.svg",
    "assets/figures/sweep_target_vs_achieved.svg",
    "assets/figures/sweep_target_vs_valbpt.svg",
    "assets/figures/validation_delta.svg",
    
    # Captions file (not needed on HF)
    "assets/figures/captions.json",
    
    # Training script (not needed on HF)
    "scripts/train_scu.py",
]

print("Cleaning up HuggingFace repo...")
print(f"Target: {REPO}")
print("="*50)

# Get current files
current_files = list_repo_files(REPO, repo_type="model")

removed = []
for file_path in files_to_remove:
    if file_path in current_files:
        try:
            delete_file(
                path_in_repo=file_path,
                repo_id=REPO,
                repo_type="model",
                commit_message=f"cleanup: remove {file_path}"
            )
            print(f"✓ Removed: {file_path}")
            removed.append(file_path)
        except Exception as e:
            print(f"✗ Failed to remove {file_path}: {e}")

print("\n" + "="*50)
print(f"Removed {len(removed)} files from HuggingFace")
print("\nEssential files kept:")
print("- README.md (model card)")
print("- adapter_config.json, adapter_model.safetensors (1B adapter)")  
print("- 3b-scu/, 3b-fixed/ (3B model variants)")
print("- notebooks/SCU_Demo.ipynb (demo)")
print("- assets/figures/*.png (essential plots only)")
print("- figures/*_1b.png (HF display plots)")