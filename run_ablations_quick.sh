#!/bin/bash

# Quick ablation studies for SCU
# Based on the validated training configuration

set -e
source ../shannon_env/bin/activate

echo "============================================================"
echo "SCU Ablation Studies - Quick 30 steps each"
echo "Testing fixed λ vs PI control"
echo "============================================================"

# Create ablations directory
mkdir -p ablations

# Test different fixed lambda values (30 steps each for quick test)
for lambda in 0.5 1.0 2.0 5.0; do
    echo ""
    echo "Training with fixed λ=$lambda (no PI control)"
    python scripts/train_scu.py \
        --base_model "meta-llama/Llama-3.2-1B" \
        --adapter_out "ablations/fixed_lambda_$lambda" \
        --steps 30 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --lr 5e-4 \
        --lambda_init $lambda \
        --kp 0.0 \
        --ki 0.0 \
        --fp16 \
        --prior_sigma 0.1 \
        --target_s 0.01 \
        --log_csv "ablations/fixed_${lambda}.csv"
done

# Run with PI control
echo ""
echo "Training with PI control (adaptive λ)"
python scripts/train_scu.py \
    --base_model "meta-llama/Llama-3.2-1B" \
    --adapter_out "ablations/pi_control" \
    --steps 30 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 5e-4 \
    --lambda_init 1.0 \
    --kp 0.8 \
    --ki 0.15 \
    --fp16 \
    --prior_sigma 0.1 \
    --target_s 0.01 \
    --log_csv "ablations/pi_control.csv"

echo ""
echo "Ablation training complete! Now evaluating..."

# Evaluate all models
for dir in ablations/*/; do
    if [ -d "$dir" ]; then
        echo "Evaluating $dir"
        python scripts/eval_bpt.py \
            --base_model "meta-llama/Llama-3.2-1B" \
            --adapter_path "$dir" \
            --texts data/val.txt \
            --output "${dir}eval.json"
    fi
done

# Generate summary
python scripts/summarize_ablations.py

echo "Ablation studies complete!"