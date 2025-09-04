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
  - cruise-control
language:
  - en
---

# Shannon Control Unit (SCU) — Cruise Control for LLM Training

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](https://shannonlabs.dev)
[![Website](https://img.shields.io/badge/Website-shannonlabs.dev-green)](https://shannonlabs.dev)

**Like cruise control maintains your speed regardless of hills, SCU maintains optimal regularization regardless of data complexity.**

## The Innovation

Set your target information ratio S*, and our PI controller automatically adjusts λ to maintain it throughout training. No manual hyperparameter tuning required.

## Validated Results

- **Llama-3.2-1B:** Base 3.920 BPT → SCU 3.676 BPT (−15.6% perplexity)
- **Mechanism scales:** Consistent control dynamics validated across model sizes
- **Production ready:** Seeking partnerships for 7B+ scale validation

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id = "meta-llama/Llama-3.2-1B"  # accept terms on HF first
base = AutoModelForCausalLM.from_pretrained(
    base_id, 
    device_map="auto", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tok = AutoTokenizer.from_pretrained(base_id)
if tok.pad_token is None: 
    tok.pad_token = tok.eos_token
base.config.pad_token_id = tok.pad_token_id

model = PeftModel.from_pretrained(base, "hunterbown/shannon-control-unit")
```

## How It Works (Cruise Control Analogy)

Just like cruise control in your car:
- **You set the target:** Choose your information ratio S* (typically 1.0%)
- **SCU maintains it automatically:** PI controller adjusts λ in real-time
- **No manual intervention:** Works across data distribution shifts and training dynamics

## Technical Details

- **Control variable:** S = ParamBPT / (DataBPT + ParamBPT)
- **Control law:** λ ← λ · exp(−(Kp·error + Ki·I))
- **Result:** Automatic regularization without hyperparameter sweeps

## Model Variants

This repository contains several checkpoints:
- `llama-3.2-1b-base-10ksteps`: Baseline model
- `llama-3.2-1b-scu-10ksteps`: SCU-controlled model
- Additional experimental variants

## Citation

If you use SCU in your research:
```bibtex
@misc{bown2024shannon,
  title={Shannon Control Unit: Cruise Control for LLM Training},
  author={Bown, Hunter},
  year={2024},
  publisher={Shannon Labs},
  url={https://shannonlabs.dev}
}
```

## License & IP

- **Adapters/models:** Meta Llama 3.2 Community License
- **SCU training code:** Apache-2.0
- **IP status:** U.S. patent pending (provisional filed September 2024)

## Links

- [Website](https://shannonlabs.dev)
- [GitHub](https://github.com/hmbown/shannon-control-unit)
- [Demo Notebook](https://huggingface.co/hunterbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb)