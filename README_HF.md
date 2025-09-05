---
license: llama3.2
library_name: peft
pipeline_tag: text-generation
base_model:
  - meta-llama/Llama-3.2-1B
  - meta-llama/Llama-3.2-3B
tags:
  - lora
  - peft
  - control-theory
  - regularization
  - information-theory
  - llama
  - adapter
language:
  - en
inference: false
---

# Shannon Control Unit (SCU) — Cruise Control for LLM Training

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](https://shannonlabs.dev)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/hunterbown/shannon-control-unit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hmbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb)
[![Website](https://img.shields.io/badge/Website-shannonlabs.dev-green)](https://shannonlabs.dev)

**Model Weights:** Llama 3.2 Community License | **Code:** Apache-2.0 ([GitHub](https://github.com/Hmbown/shannon-control-unit))

**Like cruise control maintains your speed regardless of hills, SCU maintains optimal regularization regardless of data complexity.**

Set your target information ratio \( S^* \), and our PI controller automatically adjusts \( \lambda \) to maintain it throughout training. No manual hyperparameter tuning required.

**Validated Results:**

| Model | Metric | Baseline | SCU | Improvement |
|-------|--------|----------|-----|-------------|
| **Llama-3.2-1B** | BPT | 3.920 | 3.676 | **-6.2%** |
| | Perplexity | 15.14 | 12.78 | **-15.6%** |
| **Llama-3.2-3B** 🎯 | BPT | 1.830 | 1.635 | **-10.6%** |
| | Perplexity | 3.56 | 3.11 | **-12.6%** |

**Status:** Validated at 1B/3B scales | Seeking partners for 7B+ external validation

[View validation artifacts](./3b_validation_results.json) | [Evaluation protocol](./scripts/eval_bpt.py)

## Available Models

| Directory | Model | S* Target | λ Control | Notes |
|-----------|-------|-----------|-----------|-------|
| **main** | Llama-3.2-1B | 1.0% | Adaptive PI | Primary validated model |
| **1b-scu/** | Llama-3.2-1B | 1.0% | Adaptive PI | Same as main |
| **3b-scu/** | Llama-3.2-3B | 2.88% | Adaptive (λ=2.61) | Best 3B performance |
| **3b-fixed/** | Llama-3.2-3B | 3.35% | Fixed λ=0.5 | Ablation study |

**Note:** HuggingFace UI shows only the root 1B model. Load 3B models using `subfolder="3b-scu"` parameter in code.

![Validation: Base vs SCU](assets/figures/validation_delta.png)

---

## Control telemetry

**S(t) tracking 1.0% ± 0.2pp**  
![S curve](assets/figures/s_curve.png)

**λ(t) bounded (log scale)**  
![Lambda curve](assets/figures/lambda_curve.png)

<details>
<summary><b>Training curves (details)</b></summary>

**DataBPT (bits/token)**  
![DataBPT curve](assets/figures/data_bpt_curve.png)

**ParamBPT (bits/token)**  
![ParamBPT curve](assets/figures/param_bpt_curve.png)

</details>

---

## Quick start (adapters)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# For 1B model (validated with 6.2% BPT improvement)
base_id = "meta-llama/Llama-3.2-1B"  # accept terms on HF first
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
tok  = AutoTokenizer.from_pretrained(base_id)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base.config.pad_token_id = tok.pad_token_id

# Load the validated 1B adapter (main directory or 1b-scu/)
model = PeftModel.from_pretrained(base, "hunterbown/shannon-control-unit")  

# Or for 3B models, use:
# base_id = "meta-llama/Llama-3.2-3B"
# model = PeftModel.from_pretrained(base, "hunterbown/shannon-control-unit", subfolder="3b-scu")
```

**Demo notebook:** [Open in Colab](https://colab.research.google.com/github/Hmbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb)

---

## How It Works (Cruise Control Analogy)

Just like cruise control in your car:
- **You set the target:** Choose your information ratio $S^*$  
- **SCU maintains it automatically:** PI controller adjusts $\lambda$ in real-time
- **No manual intervention:** Works across data distribution shifts and training dynamics

**Technical Details:**
- **Control variable:** $S=\frac{\text{ParamBPT}}{\text{DataBPT}+\text{ParamBPT}}$
- **Control law:** $\lambda \leftarrow \lambda \cdot \exp(-(K_p \cdot \text{error} + K_i \cdot I))$
- **Result:** Automatic regularization without hyperparameter sweeps

**Key Research Question:** 
Optimal $S^*$ scaling laws are still being discovered. We found 1.0% works for 1B models and 2.88% for 3B models. The relationship between model size, training data, and optimal $S^*$ is an active area of research.

---

## Licensing & IP

* **Model weights:** Meta Llama 3.2 Community License (inherited from base model)
* **SCU training code:** Apache-2.0 License ([GitHub repository](https://github.com/Hmbown/shannon-control-unit))
* **IP status:** U.S. patent pending (provisional filed September 2025)

> Repro tips: block size 1024, batch 1, grad-accum 4, gradient checkpointing on, `use_cache=False`.
