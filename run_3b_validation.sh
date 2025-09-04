#!/bin/bash
# Quick 3B validation script for SCU models
# Run this tonight to have data for tomorrow's outreach

echo "========================================="
echo "Shannon Control Unit - 3B Model Validation"
echo "Testing efficiency gains for investor outreach"
echo "========================================="

# Check if we're on Mac (MPS) or Linux (CUDA)
if [[ "$OSTYPE" == "darwin"* ]]; then
    DEVICE="mps"
    echo "Running on Mac (MPS)"
else
    DEVICE="cuda"
    echo "Running on Linux/CUDA"
fi

# Run the validation
python3 test_3b_models.py \
    --base-model "meta-llama/Llama-3.2-3B" \
    --device $DEVICE \
    --output "3b_validation_results_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "========================================="
echo "Validation complete! Check results above."
echo "Use the 'EMAIL-READY SUMMARY' for tomorrow's outreach."
echo "========================================="