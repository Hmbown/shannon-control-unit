#!/usr/bin/env python3
"""Test SCU model loading from local files."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys

def test_model_variant(base_id, adapter_path, variant_name):
    """Test loading a specific model variant."""
    print(f"\n{'='*60}")
    print(f"Testing {variant_name}")
    print(f"Base: {base_id}")
    print(f"Adapter: {adapter_path}")
    print('='*60)
    
    try:
        # Load base model
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_id, 
            device_map="cpu",  # Use CPU for testing
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base.config.pad_token_id = tokenizer.pad_token_id
        
        # Load adapter
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base, adapter_path)
        
        # Test generation
        print("\nTesting generation...")
        prompt = "The Shannon Control Unit is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        print(f"\n✅ {variant_name} loaded and tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing {variant_name}: {e}")
        return False

def main():
    """Test all model variants."""
    print("Shannon Control Unit - Model Loading Test")
    print("="*60)
    
    results = []
    
    # Test 1B model (main directory)
    if os.path.exists("adapter_config.json"):
        results.append(("1B Main", test_model_variant(
            "meta-llama/Llama-3.2-1B",
            ".",
            "Llama-3.2-1B SCU (main directory)"
        )))
    
    # Test 1B model (1b-scu directory)
    if os.path.exists("1b-scu/adapter_config.json"):
        results.append(("1B Subdirectory", test_model_variant(
            "meta-llama/Llama-3.2-1B",
            "./1b-scu",
            "Llama-3.2-1B SCU (1b-scu/)"
        )))
    
    # Test 3B SCU model
    if os.path.exists("3b-scu/adapter_config.json"):
        results.append(("3B SCU", test_model_variant(
            "meta-llama/Llama-3.2-3B",
            "./3b-scu",
            "Llama-3.2-3B SCU (3b-scu/)"
        )))
    
    # Test 3B Fixed model
    if os.path.exists("3b-fixed/adapter_config.json"):
        results.append(("3B Fixed", test_model_variant(
            "meta-llama/Llama-3.2-3B",
            "./3b-fixed",
            "Llama-3.2-3B Fixed λ (3b-fixed/)"
        )))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:20} {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All models loaded successfully!")
    else:
        print("\n⚠️ Some models failed to load. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()