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

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](https://shannonlabs.dev)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/hunterbown/shannon-control-unit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hmbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb)
[![Website](https://img.shields.io/badge/Website-shannonlabs.dev-green)](https://shannonlabs.dev)

**Like cruise control maintains your speed regardless of hills, SCU maintains optimal regularization regardless of data complexity.**

Set your target information ratio \( S^* \), and our PI controller automatically adjusts \( \lambda \) to maintain it throughout training. No manual hyperparameter tuning required.

**Validated Results:**
- **Llama-3.2-1B:** Base 3.920 BPT → SCU 3.676 BPT (15.6% lower perplexity, 6.2% lower BPT)
- **🎯 Llama-3.2-3B:** Base 1.8295 BPT → SCU 1.6351 BPT (10.6% lower BPT)
- **Production ready:** Seeking partnerships for 7B+ scale validation

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

## How It Works (Cruise Control Analogy)

Just like cruise control in your car:
- **You set the target:** Choose your information ratio $S^*$ (typically 1.0%)  
- **SCU maintains it automatically:** PI controller adjusts $\lambda$ in real-time
- **No manual intervention:** Works across data distribution shifts and training dynamics

**Technical Details:**
- **Control variable:** $S=\frac{\text{ParamBPT}}{\text{DataBPT}+\text{ParamBPT}}$
- **Control law:** $\lambda \leftarrow \lambda \cdot \exp(-(K_p\,\text{error}+K_i\,I))$
- **Result:** Automatic regularization without hyperparameter sweeps

---

## Licensing & IP

* **Adapters/models:** Meta **Llama 3.2** Community License
* **SCU training code:** **Apache-2.0**
* **IP status:** U.S. **patent pending** (provisional filed September 2025)

> Repro tips: block size 1024, batch 1, grad-accum 4, gradient checkpointing on, `use_cache=False`.