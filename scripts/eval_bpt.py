#!/usr/bin/env python3
"""Evaluate base model vs SCU adapter on BPT and perplexity."""

import os
import sys
import argparse
import math
import json
import random
import statistics as stats
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))
from scu import data


def bpt_for_texts(model, tokenizer, texts, max_len=512, device=None):
    """Calculate BPT for each text.
    
    Returns list of BPT values (one per text).
    """
    model.eval()
    bpts = []
    
    for text in texts:
        # Tokenize
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False
        )
        
        # Move to device
        enc = {k: v.to(device or model.device) for k, v in enc.items()}
        
        # Labels are same as inputs
        labels = enc["input_ids"].clone()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**enc, labels=labels)
            # Convert from nats to bits
            bpt = outputs.loss.item() / math.log(2)
            bpts.append(bpt)
    
    return bpts


def bootstrap_ci(delta_list, iters=10000, seed=42):
    """Bootstrap confidence interval for mean difference.
    
    Returns (lower_95, mean, upper_95)
    """
    random.seed(seed)
    means = []
    n = len(delta_list)
    
    for _ in range(iters):
        # Resample with replacement
        sample = [delta_list[random.randrange(n)] for _ in range(n)]
        means.append(stats.mean(sample))
    
    means.sort()
    lower = means[int(0.025 * iters)]
    upper = means[int(0.975 * iters)]
    mean_val = stats.mean(delta_list)
    
    return lower, mean_val, upper


def main(args):
    # Suppress tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Setup device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        use_4bit = not args.no_4bit
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        use_4bit = False
    else:
        device = "cpu"
        dtype = torch.float32
        use_4bit = False
        print("WARNING: Using CPU - evaluation will be slow")
    
    # Quantization config
    quantization_config = None
    if use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation texts
    print(f"Loading validation texts from {args.texts}")
    val_texts = data.load_texts_from_file(args.texts, max_texts=args.max_texts)
    print(f"Loaded {len(val_texts)} texts")
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_bpts = bpt_for_texts(base_model, tokenizer, val_texts, max_len=args.max_length, device=device)
    base_mean_bpt = stats.mean(base_bpts)
    base_perplexity = 2 ** base_mean_bpt
    
    # Load adapter model if provided
    if args.adapter_path:
        print(f"\nLoading SCU adapter from {args.adapter_path}")
        scu_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        scu_model.eval()
        
        # Evaluate SCU model
        print("Evaluating SCU model...")
        scu_bpts = bpt_for_texts(scu_model, tokenizer, val_texts, max_len=args.max_length, device=device)
        scu_mean_bpt = stats.mean(scu_bpts)
        scu_perplexity = 2 ** scu_mean_bpt
        
        # Calculate differences
        delta_bpts = [b - s for b, s in zip(base_bpts, scu_bpts)]
        delta_mean = stats.mean(delta_bpts)
        
        # Bootstrap CI
        if args.bootstrap:
            print("\nCalculating bootstrap confidence interval...")
            ci_lower, ci_mean, ci_upper = bootstrap_ci(delta_bpts, iters=args.bootstrap_iters)
        else:
            ci_lower = ci_mean = ci_upper = delta_mean
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Base Model:    {base_mean_bpt:.3f} BPT (ppl {base_perplexity:.2f})")
        print(f"SCU Model:     {scu_mean_bpt:.3f} BPT (ppl {scu_perplexity:.2f})")
        print(f"Improvement:   {delta_mean:.3f} BPT ({100*delta_mean/base_mean_bpt:.1f}%)")
        print(f"Perplexity:    -{100*(1 - scu_perplexity/base_perplexity):.1f}%")
        
        if args.bootstrap:
            print(f"\nBootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            if ci_lower > 0:
                print("✓ CI excludes zero - improvement is statistically significant")
            else:
                print("✗ CI includes zero - improvement not statistically significant")
        
        # Save results if requested
        if args.output:
            results = {
                'base_model': args.base_model,
                'adapter_path': args.adapter_path,
                'num_texts': len(val_texts),
                'base_bpt': base_mean_bpt,
                'scu_bpt': scu_mean_bpt,
                'delta_bpt': delta_mean,
                'delta_bpt_percent': 100 * delta_mean / base_mean_bpt,
                'base_perplexity': base_perplexity,
                'scu_perplexity': scu_perplexity,
                'perplexity_reduction': 100 * (1 - scu_perplexity/base_perplexity),
                'ci_lower': ci_lower,
                'ci_mean': ci_mean,
                'ci_upper': ci_upper,
                'individual_base_bpts': base_bpts,
                'individual_scu_bpts': scu_bpts
            }
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {args.output}")
    
    else:
        # Base model only
        print("\n" + "="*60)
        print("BASE MODEL RESULTS")
        print("="*60)
        print(f"BPT:        {base_mean_bpt:.3f}")
        print(f"Perplexity: {base_perplexity:.2f}")
        print(f"Texts:      {len(val_texts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BPT and perplexity")
    
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B",
                       help="Base model name")
    parser.add_argument("--adapter_path", default=None,
                       help="Path to SCU adapter (optional)")
    parser.add_argument("--texts", default="data/val.txt",
                       help="Validation texts file")
    parser.add_argument("--max_texts", type=int, default=None,
                       help="Maximum texts to evaluate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--no_4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Calculate bootstrap CI")
    parser.add_argument("--bootstrap_iters", type=int, default=10000,
                       help="Bootstrap iterations")
    parser.add_argument("--output", default=None,
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    main(args)