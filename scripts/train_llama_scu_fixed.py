#!/usr/bin/env python3
"""
FIXED Llama SCU Training - With Differentiable MDL Penalty
Fixes the critical bug where parameter penalty was not affecting gradients.
"""

import math
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Import the FIXED SCU with differentiable penalty
from scu_llama_fixed import ShannonControlUnit

LN2 = math.log(2.0)

class TextDataset(Dataset):
    """Simple text dataset for MDL training - matching original setup."""
    def __init__(self, texts, tokenizer, block_size=1024, overlap=128):
        self.examples = []
        self.block_size = block_size
        self.overlap = overlap
        
        # Tokenize all texts and combine
        all_tokens = []
        for text in texts:
            # Process in chunks to avoid memory issues
            chunk_size = 100000  # characters
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                # Tokenize chunk
                chunk_tokens = tokenizer.encode(
                    chunk,
                    truncation=True,
                    max_length=131072,
                    add_special_tokens=(i == 0 and len(all_tokens) == 0)
                )
                all_tokens.extend(chunk_tokens)
        
        tokens = all_tokens
        self.N = len(tokens)  # Store total tokens N
        print(f"Tokenized {len(tokens)} tokens")
        print(f"Total tokens loaded (N): {self.N:,}")
        
        # Create overlapping blocks (matching original)
        stride = block_size - overlap
        for i in range(0, len(tokens) - block_size + 1, stride):
            self.examples.append(torch.tensor(tokens[i:i + block_size]))
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def load_training_data(data_path: str, tokenizer, sample_size: int = None, 
                      block_size: int = 1024, overlap: int = 128):
    """Load and tokenize training data."""
    data = []
    
    # Check if data_path is a file or directory
    data_path = Path(data_path)
    if data_path.is_file():
        # Load single file
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if sample_size and len(text) > sample_size:
                text = text[:sample_size]
            data.append(text)
    else:
        # Load all text files from directory
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if sample_size and len(text) > sample_size:
                    text = text[:sample_size]
                data.append(text)
    
    if not data:
        raise ValueError(f"No text data found in {data_path}")
    
    # Create dataset with proper block_size and overlap
    dataset = TextDataset(data, tokenizer, block_size=block_size, overlap=overlap)
    
    # Use the N from the dataset (total tokens tokenized)
    total_tokens = dataset.N
    
    return dataset, total_tokens

def train_llama_scu(args):
    """Main training function with fixed differentiable MDL penalty."""
    
    # Initialize accelerator
    # Check if we're on MPS (Apple Silicon) which doesn't support fp16
    import platform
    is_mps = platform.system() == 'Darwin' and platform.processor() == 'arm'
    
    if is_mps and args.fp16:
        print("Note: fp16 not supported on Apple Silicon MPS, using fp32 instead")
        mixed_precision = None
        torch_dtype = torch.float32
    else:
        mixed_precision = 'fp16' if args.fp16 else None
        torch_dtype = torch.float16 if args.fp16 else torch.float32
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("Creating dataset...")
    dataset, N = load_training_data(args.data_path, tokenizer, args.sample_size)
    print(f"Total tokens loaded (N): {N:,}")
    
    # Create dataloader
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded = []
        for x in batch:
            padding = torch.full((max_len - len(x),), tokenizer.pad_token_id)
            padded.append(torch.cat([x, padding]))
        return torch.stack(padded)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"Created {len(dataset)} training examples")
    
    # Calculate steps based on epochs if specified
    if args.num_epochs:
        steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
        args.max_steps = args.num_epochs * steps_per_epoch
        print(f"Training for {args.num_epochs} epochs = {args.max_steps} steps")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=0.0  # We handle regularization ourselves
    )
    
    # Initialize SCU
    print(f"Using dataset token count for N (assumes 1 epoch): {N:,}")
    
    # Create SCU with target S
    if args.mode in ["ce_kl_fixed", "ce_kl_auto"]:
        band_width = args.target_S * 0.2  # 20% tolerance
        scu = ShannonControlUnit(
            prior_sigma=args.prior_sigma,
            q_sigma=args.prior_sigma,  # Equal sigmas = L2
            target_S=args.target_S,
            band=(args.target_S - band_width, args.target_S + band_width),
            kp=2.0,
            ki=0.5,
            ema=0.90,
            lam_init=args.lam_init,
            lam_min=1e-6,
            lam_max=1e2,
            log_path=str(args.output_dir / "mdl_ledger.jsonl")
        )
        lam = args.lam_init
    
    # Prepare for training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    print(f"\nStarting training with mode: {args.mode}")
    if args.mode == "ce_kl_auto":
        print(f"Target S: {args.target_S:.1%} ± {band_width:.1%}")
    
    global_step = 0
    progress_bar = tqdm(range(args.max_steps), desc=f"Training ({args.mode})")
    
    # Metrics tracking
    metrics = {
        'steps': [],
        'data_bpt': [],
        'param_bpt': [],
        'total_bpt': [],
        'S_ratio': [],
        'lambda': [],
        'grad_norm': []
    }
    
    for epoch in range(args.max_epochs):
        for batch in dataloader:
            if global_step >= args.max_steps:
                break
            
            # Forward pass
            outputs = model(batch, labels=batch)
            ce_loss = outputs.loss  # This is averaged per token
            
            # Convert to nats per token
            ce_nats_per_token = ce_loss * LN2
            batch_tokens = (batch != tokenizer.pad_token_id).sum().item()
            
            # CRITICAL FIX: Compute both monitoring and optimization penalties
            param_bpt_monitor = 0.0  # Default for modes without MDL
            if args.mode in ["ce_kl_fixed", "ce_kl_auto"]:
                # 1. Monitoring value (non-differentiable) for SCU updates
                kl_bits_monitor = scu.compute_param_bits_monitoring(model, lora_only=True)
                param_bpt_monitor = kl_bits_monitor / max(N, 1)
                
                # 2. Differentiable penalty for optimization
                kl_nats_diff = scu.compute_differentiable_penalty_nats(model, lora_only=True)
                param_penalty_nats = kl_nats_diff / max(N, 1)  # Amortized over N
            
            # Compute data BPT
            data_bpt = ce_nats_per_token.detach().item() / LN2
            
            # Update lambda if auto-tuning
            if args.mode == "ce_kl_auto" and accelerator.sync_gradients:
                lam, S, data_bpt_smooth, param_bpt_smooth = scu.update_lambda(
                    data_bpt, param_bpt_monitor
                )
                scu.tokens_seen += batch_tokens
            
            # Compose total loss based on mode
            if args.mode == "ce":
                total_loss = ce_nats_per_token
            elif args.mode == "ce_l2":
                # Classical L2 regularization on LoRA weights
                l2_sum = torch.tensor(0.0, device=ce_loss.device)
                for name, param in model.named_parameters():
                    if param.requires_grad and "lora_" in name:
                        l2_sum = l2_sum + (param.float() ** 2).sum()
                total_loss = ce_nats_per_token + args.l2_weight * l2_sum / batch_tokens
            elif args.mode == "ce_kl_fixed":
                # Fixed lambda with DIFFERENTIABLE penalty
                total_loss = ce_nats_per_token + args.lam_init * param_penalty_nats
            else:  # ce_kl_auto
                # Auto-tuned lambda with DIFFERENTIABLE penalty
                total_loss = ce_nats_per_token + lam * param_penalty_nats
            
            # Backward pass
            accelerator.backward(total_loss)
            
            # Gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Calculate optimization loss BPT
                optimization_loss_bpt = total_loss.detach().item() / LN2
                
                # Calculate MDL BPT (λ=1) for monitoring
                mdl_bpt = data_bpt + param_bpt_monitor
                
                # Track metrics
                metrics['steps'].append(global_step)
                metrics['data_bpt'].append(data_bpt)
                metrics['param_bpt'].append(param_bpt_monitor)
                metrics['total_bpt'].append(mdl_bpt)
                metrics['grad_norm'].append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
                
                if args.mode in ["ce_kl_fixed", "ce_kl_auto"]:
                    S_current = param_bpt_monitor / max(mdl_bpt, 1e-6)
                    metrics['S_ratio'].append(S_current)
                    metrics['lambda'].append(lam)
                
                # Update progress bar with rich metrics
                desc = f"Training ({args.mode})"
                if global_step % 2 == 0:  # Update more frequently
                    if args.mode in ["ce_kl_fixed", "ce_kl_auto"]:
                        # Show trend arrows for key metrics
                        if len(metrics['lambda']) > 1:
                            λ_trend = "↓" if lam < metrics['lambda'][-2] else "↑" if lam > metrics['lambda'][-2] else "→"
                        else:
                            λ_trend = ""
                        
                        if len(metrics['S_ratio']) > 1:
                            S_trend = "↑" if S_current > metrics['S_ratio'][-2] else "↓" if S_current < metrics['S_ratio'][-2] else "→"
                        else:
                            S_trend = ""
                        
                        desc = f"Data_BPT={data_bpt:.3f} | Param_BPT={param_bpt_monitor:.4f} | S={S_current*100:.2f}%{S_trend} (target:1%) | λ={lam:.2f}{λ_trend} | Total_BPT={mdl_bpt:.3f}"
                    else:
                        desc = f"BPT={optimization_loss_bpt:.3f} | Grad={grad_norm:.3f}"
                
                progress_bar.set_description(desc)
                progress_bar.update(1)
                
                # Check if we're in a good state (for early stopping indication)
                if args.mode == "ce_kl_auto" and abs(S_current - args.target_S) < 0.002:
                    print(f"\n✓ S is near target! S={S_current*100:.2f}% (target={args.target_S*100:.1f}%)")
                
                # Periodic logging
                if global_step % 10 == 0:
                    log_entry = {
                        'step': global_step,
                        'mode': args.mode,
                        'data_bpt': round(data_bpt, 4),
                        'optimization_loss_bpt': round(optimization_loss_bpt, 4),
                        'mdl_bpt': round(mdl_bpt, 4),
                        'grad_norm': round(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, 4)
                    }
                    
                    if args.mode in ["ce_kl_fixed", "ce_kl_auto"]:
                        log_entry.update({
                            'param_bpt': round(param_bpt_monitor, 6),
                            'S_ratio': round(S_current, 4),
                            'lambda': round(lam, 6),
                        })
                    
                    # Save incremental metrics
                    with open(args.output_dir / f"{args.mode}_metrics.jsonl", 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')
        
        if global_step >= args.max_steps:
            break
    
    progress_bar.close()
    
    # Save final model
    print(f"\nSaving model to {args.output_dir}")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save metrics summary
    summary = {
        'mode': args.mode,
        'model': args.model_name,
        'steps': args.max_steps,
        'N_tokens': N,
        'target_S': args.target_S if args.mode == "ce_kl_auto" else None,
        'final_metrics': {
            'data_bpt': metrics['data_bpt'][-1] if metrics['data_bpt'] else None,
            'param_bpt': metrics['param_bpt'][-1] if metrics['param_bpt'] else None,
            'total_bpt': metrics['total_bpt'][-1] if metrics['total_bpt'] else None,
            'S_ratio': metrics['S_ratio'][-1] if metrics['S_ratio'] else None,
            'lambda': metrics['lambda'][-1] if metrics['lambda'] else None,
        },
        'avg_last_100': {
            'data_bpt': np.mean(metrics['data_bpt'][-100:]) if len(metrics['data_bpt']) > 100 else np.mean(metrics['data_bpt']),
            'total_bpt': np.mean(metrics['total_bpt'][-100:]) if len(metrics['total_bpt']) > 100 else np.mean(metrics['total_bpt']),
            'S_ratio': np.mean(metrics['S_ratio'][-100:]) if len(metrics['S_ratio']) > 100 else np.mean(metrics['S_ratio']) if metrics['S_ratio'] else None,
        }
    }
    
    with open(args.output_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final MDL BPT: {summary['final_metrics']['total_bpt']:.4f}")
    if args.mode in ["ce_kl_fixed", "ce_kl_auto"]:
        print(f"Final S ratio: {summary['final_metrics']['S_ratio']:.2%}")
        print(f"Final lambda: {summary['final_metrics']['lambda']:.6f}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Train Llama with FIXED MDL/SCU")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model to train")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for model and logs")
    
    # Training arguments
    parser.add_argument("--mode", type=str, required=True,
                        choices=["ce", "ce_l2", "ce_kl_fixed", "ce_kl_auto"],
                        help="Training mode")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum training steps")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Maximum epochs")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Target number of epochs (overrides max-steps if set)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 mixed precision")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, default="training_data",
                        help="Path to training data directory")
    parser.add_argument("--sample-size", type=int, default=2000000,
                        help="Sample size per file in characters")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    
    # MDL/SCU arguments
    parser.add_argument("--prior-sigma", type=float, default=0.1,
                        help="Prior sigma for KL divergence")
    parser.add_argument("--target-S", type=float, default=0.01,
                        help="Target S ratio for SCU (1% default)")
    parser.add_argument("--lam-init", type=float, default=1.0,
                        help="Initial lambda for fixed mode")
    parser.add_argument("--l2-weight", type=float, default=0.1,
                        help="L2 weight for ce_l2 mode")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()  # Make a copy to avoid modifying args
    # Convert Path objects to strings for JSON serialization
    config['output_dir'] = str(args.output_dir)
    with open(args.output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    train_llama_scu(args)

if __name__ == "__main__":
    main()