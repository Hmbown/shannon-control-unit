---
license: llama3.2
library_name: transformers
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - control-theory
  - regularization
  - information-theory
  - llama
language:
  - en
---

# Shannon Control Unit (SCU) — Dial-in LLM regularization

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](https://shannonlabs.dev)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/hunterbown/shannon-control-unit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hmbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb)
[![Website](https://img.shields.io/badge/Website-shannonlabs.dev-green)](https://shannonlabs.dev)

**Idea.** Hold an MDL-motivated **information budget** during training:  
\( S = \frac{\text{ParamBPT}}{\text{DataBPT}+\text{ParamBPT}} \).  
A simple **PI controller** adjusts \( \lambda \) so \( S \) tracks a target \( S^* \).

**Result (held-out, Llama-3.2-1B):** Base **3.920 BPT** (ppl **15.14**) → **SCU** **3.676 BPT** (ppl **12.78**),  
**Δ = −0.244 BPT** (≈ **−15.6%** perplexity).

## Available Models

- **Main directory**: Llama-3.2-1B SCU adapter (validated, S=1.0%)
- **1b-scu/**: Same as main (Llama-3.2-1B SCU, S=1.0%, λ adaptive)
- **3b-scu/**: Llama-3.2-3B SCU adapter (S=2.88%, λ=2.61) 
- **3b-fixed/**: Llama-3.2-3B fixed λ=0.5 (S=3.35%)

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

# For 1B model (recommended - validated with 6.2% improvement)
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

**Demo notebook:** [Open in Colab](https://huggingface.co/hunterbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb) (hosted on HuggingFace)

---

## Method (one screen)

* **Target:** $S=\frac{\text{ParamBPT}}{\text{DataBPT}+\text{ParamBPT}}$
* **Update:** $\lambda \leftarrow \lambda \cdot \exp(-(K_p\,\text{error}+K_i\,I))$, with $\text{error}=\hat S-S^*$
* **ParamBPT:** quadratic term vs $\mathcal N(0,\sigma^2)$, **nats→bits**, normalized by fixed $N$ (per epoch/report window)

**Why it helps:** You **dial a capacity share** $S^*$ and the loop enforces it across model size/data drift—no λ grid search.

---

## Licensing & IP

* **Adapters/models:** Meta **Llama 3.2** Community License
* **SCU training code:** **Apache-2.0**
* **IP status:** U.S. **patent pending** (provisional filed September 2025)

> Repro tips: block size 1024, batch 1, grad-accum 4, gradient checkpointing on, `use_cache=False`.