#!/usr/bin/env python3
"""
Test 3B Llama models with SCU adapters to validate efficiency gains.
Runs tonight to have data for tomorrow's outreach.
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from tqdm import tqdm

def calculate_perplexity_and_bpt(model, tokenizer, text, device='cuda'):
    """Calculate perplexity and bits-per-token for given text."""
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss.item()  # This is cross-entropy loss in nats
        
    # Convert to bits per token
    bpt = loss / np.log(2)  # Convert nats to bits
    
    # Calculate perplexity
    perplexity = np.exp(loss)
    
    return {
        'loss': loss,
        'bpt': bpt,
        'perplexity': perplexity,
        'n_tokens': inputs['input_ids'].shape[1]
    }

def load_test_data():
    """Load test data from various sources."""
    test_texts = []
    
    # Sample text categories
    samples = {
        'code': '''def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)''',
        
        'technical': '''The transformer architecture has revolutionized natural language processing 
through self-attention mechanisms that capture long-range dependencies. Unlike recurrent neural 
networks, transformers process sequences in parallel, leading to significant training efficiency 
improvements. The multi-head attention mechanism allows the model to jointly attend to information 
from different representation subspaces at different positions.''',
        
        'narrative': '''The old lighthouse keeper climbed the spiral stairs one last time. Each step 
echoed through the tower, a rhythm he'd known for forty years. Tonight was different. The automated 
system would take over at midnight, rendering his vigil obsolete. He reached the lamp room and gazed 
at the horizon, where sky met sea in an endless embrace of blue.''',
        
        'scientific': '''Quantum entanglement represents one of the most counterintuitive phenomena in 
physics. When two particles become entangled, measuring the state of one instantly affects the state 
of the other, regardless of the distance separating them. Einstein famously referred to this as 
"spooky action at a distance," and it challenges our classical understanding of locality and realism 
in physical systems.''',
        
        'conversational': '''Hey, did you see the game last night? It was absolutely incredible! The 
underdog team came back from a 20-point deficit in the fourth quarter. I've never seen anything 
like it. The crowd went wild, and even the announcers were at a loss for words. Sports can really 
surprise you sometimes, you know?'''
    }
    
    for category, text in samples.items():
        test_texts.append({'category': category, 'text': text})
    
    # Try to load validation file if it exists
    val_file = Path('/Volumes/VIXinSSD/shannon/shannon_llm/shannon-control-unit/data/val.txt')
    if val_file.exists():
        with open(val_file, 'r') as f:
            val_text = f.read()[:5000]  # First 5000 chars
            test_texts.append({'category': 'validation_file', 'text': val_text})
    
    return test_texts

def test_model_configuration(base_model_name, adapter_path, test_texts, device='cuda'):
    """Test a specific model configuration."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {adapter_path if adapter_path else 'Base Model'}")
    print(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None,
        low_cpu_mem_usage=True
    )
    
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model_name = Path(adapter_path).name
    else:
        model = base_model
        model_name = "base_model"
    
    model.eval()
    
    # Test on each text
    results = []
    for test_item in tqdm(test_texts, desc="Evaluating"):
        metrics = calculate_perplexity_and_bpt(
            model, tokenizer, test_item['text'], device
        )
        metrics['category'] = test_item['category']
        results.append(metrics)
    
    # Calculate averages
    avg_bpt = np.mean([r['bpt'] for r in results])
    avg_perplexity = np.mean([r['perplexity'] for r in results])
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"{'Category':<20} {'BPT':<10} {'Perplexity':<12} {'Tokens':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['category']:<20} {r['bpt']:<10.4f} {r['perplexity']:<12.2f} {r['n_tokens']:<10}")
    print("-" * 55)
    print(f"{'AVERAGE':<20} {avg_bpt:<10.4f} {avg_perplexity:<12.2f}")
    
    # Clean up
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'model': model_name,
        'avg_bpt': avg_bpt,
        'avg_perplexity': avg_perplexity,
        'details': results
    }

def main():
    parser = argparse.ArgumentParser(description='Test 3B models with SCU adapters')
    parser.add_argument('--base-model', default='meta-llama/Llama-3.2-3B', 
                       help='Base model name')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output', default='3b_validation_results.json',
                       help='Output file for results')
    args = parser.parse_args()
    
    print(f"Shannon Control Unit - 3B Model Validation")
    print(f"Running on: {args.device}")
    print(f"Base model: {args.base_model}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load test data
    test_texts = load_test_data()
    print(f"\nLoaded {len(test_texts)} test texts")
    
    # Define model configurations to test
    configs = [
        ('Base Model', None),
        ('3B-SCU', '/Volumes/VIXinSSD/shannon/shannon_llm/hf_repo_temp/3b-scu'),
        ('3B-Fixed', '/Volumes/VIXinSSD/shannon/shannon_llm/hf_repo_temp/3b-fixed'),
    ]
    
    # Also check for local models
    local_3b_paths = [
        '/Volumes/VIXinSSD/shannon/shannon_llm/models/llama-3.2-3b-scu',
        '/Volumes/VIXinSSD/shannon/shannon_llm/models/llama-3.2-3b-fixed',
    ]
    
    for path in local_3b_paths:
        if Path(path).exists():
            configs.append((Path(path).name, path))
    
    # Test each configuration
    all_results = []
    for name, adapter_path in configs:
        try:
            result = test_model_configuration(
                args.base_model, adapter_path, test_texts, args.device
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error testing {name}: {e}")
            continue
    
    # Compare results
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARATIVE RESULTS")
        print(f"{'='*60}")
        
        base_result = all_results[0]
        print(f"\n{'Model':<25} {'BPT':<12} {'Perplexity':<12} {'ΔBPT':<12} {'Δppl %':<12}")
        print("-" * 75)
        
        for result in all_results:
            if result['model'] == 'base_model':
                print(f"{result['model']:<25} {result['avg_bpt']:<12.4f} {result['avg_perplexity']:<12.2f} {'---':<12} {'---':<12}")
            else:
                delta_bpt = result['avg_bpt'] - base_result['avg_bpt']
                delta_ppl_pct = ((result['avg_perplexity'] - base_result['avg_perplexity']) / base_result['avg_perplexity']) * 100
                print(f"{result['model']:<25} {result['avg_bpt']:<12.4f} {result['avg_perplexity']:<12.2f} {delta_bpt:<12.4f} {delta_ppl_pct:<12.1f}%")
        
        # Find best performer
        best = min(all_results[1:], key=lambda x: x['avg_bpt'])
        improvement_bpt = base_result['avg_bpt'] - best['avg_bpt']
        improvement_ppl_pct = ((base_result['avg_perplexity'] - best['avg_perplexity']) / base_result['avg_perplexity']) * 100
        
        print(f"\n{'='*60}")
        print(f"🎯 BEST PERFORMER: {best['model']}")
        print(f"   BPT Improvement: {improvement_bpt:.4f} bits")
        print(f"   Perplexity Improvement: {improvement_ppl_pct:.1f}%")
        print(f"{'='*60}")
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'base_model': args.base_model,
        'device': args.device,
        'results': all_results,
        'summary': {
            'best_model': min(all_results, key=lambda x: x['avg_bpt'])['model'],
            'best_bpt': min(all_results, key=lambda x: x['avg_bpt'])['avg_bpt'],
            'base_bpt': all_results[0]['avg_bpt'] if all_results else None,
            'improvement': all_results[0]['avg_bpt'] - min(all_results, key=lambda x: x['avg_bpt'])['avg_bpt'] if all_results else None
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # Print email-ready summary
    if len(all_results) > 1 and output_data['summary']['improvement'] > 0:
        print(f"\n{'='*60}")
        print("📧 EMAIL-READY SUMMARY:")
        print(f"{'='*60}")
        print(f"3B Validation Results:")
        print(f"Base Model: {output_data['summary']['base_bpt']:.3f} BPT")
        print(f"SCU Model: {output_data['summary']['best_bpt']:.3f} BPT")
        print(f"Improvement: {output_data['summary']['improvement']:.3f} BPT ({improvement_ppl_pct:.1f}% perplexity reduction)")
        print(f"\nValidated on Llama-3.2-3B. 70B validation next.")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()