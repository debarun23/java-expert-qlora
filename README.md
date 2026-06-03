# java-expert-qlora

> Fine-tuned Qwen2.5-3B on an 8GB consumer GPU using QLoRA to build a domain-locked Java QA assistant · Val perplexity 2.40 · 7,921 QA pairs

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-JavaExpert--Qwen2.5--3B-blue)](https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What This Is

Most LLM fine-tuning work happens on cloud A100s with unlimited VRAM budgets. This project did it differently: a production-ready Java programming assistant, built on a single 8GB consumer GPU, that answers Java questions accurately and refuses everything else — no cloud, no shortcuts.

**The model:**
- Answers Java questions with accurate, grounded explanations
- Refuses non-Java queries without external guardrails
- Runs inference at under 2GB VRAM
- Ships as a single merged model — no adapter loading at runtime

---

## Results

| Metric | Target | Achieved |
|---|---|---|
| Validation Perplexity | < 10 | **2.40** |
| Peak Training VRAM | ≤ 8 GB | **< 7 GB** |
| Inference VRAM | — | **< 2 GB** |
| Java Correctness | — | **8.5 / 10** |
| Domain Restriction | — | **8.5 / 10** |
| Hallucination Control | — | **8.0 / 10** |

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Debarun12/JavaExpert-Qwen2.5-3B")
tokenizer = AutoTokenizer.from_pretrained("Debarun12/JavaExpert-Qwen2.5-3B")

def ask(question):
    messages = [
        {"role": "system", "content": "You are a Java programming assistant."},
        {"role": "user", "content": question}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=256)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

ask("What is polymorphism in Java?")
```

---

## How It Works

### 1. Data Engineering
- Started with a raw 42,000-line Java documentation corpus (PDF-extracted, noisy)
- Built a custom preprocessing pipeline: sliding window chunking (150 words, 30 overlap) + `is_valid_chunk()` filter rejecting OCR artifacts, TOC noise, and short stubs
- Generated **7,921 QA pairs** using `qwen2.5:7b` via `ollama` across 4 question types: *what*, *how*, *why*, *unanswerable*
- The `unanswerable` category is the key to domain restriction — it teaches the model when not to answer

### 2. QLoRA Fine-Tuning on 8GB GPU
Every config decision was driven by explicit VRAM accounting:

| Component | Initial | Final | VRAM Saved |
|---|---|---|---|
| LoRA rank | 32 across 7 modules | 16 on q_proj, v_proj | −1.5 GB |
| Batch size | 4 | 1 + grad accum ×8 | −2.0 GB |
| Optimizer | AdamW | Adafactor | −1.0 GB |
| Gradient checkpointing | Enabled | Disabled | −1.0 GB |

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 8   # effective batch = 8
optim                       = "adafactor"
bf16                        = True
learning_rate               = 2e-4
num_train_epochs            = 3
lora_r                      = 16
lora_target_modules         = ["q_proj", "v_proj"]
```

### 3. Training Results

| Epoch | Train Loss | Val Loss | Val Perplexity |
|---|---|---|---|
| 1 | 0.8511 | 0.9343 | 2.55 |
| 2 | 0.8367 | 0.8839 | 2.42 |
| 3 | 0.7780 | 0.8772 | **2.40** |

No overfitting across all 3 epochs. Validation loss decreased monotonically.

---

## Domain Restriction in Practice

| Query | Response |
|---|---|
| *What is the capital of France?* | I don't have enough information in my knowledge base to answer this. |
| *Who won the FIFA World Cup?* | I don't have enough information in my knowledge base to answer this. |
| *What is polymorphism in Java?* | Polymorphism allows objects of different classes to be treated as objects of a common superclass... |

---

## Bugs Fixed During Training

**SFTTrainer step-count explosion** — `packing=True` with pre-tokenized data caused 2,139 steps on 90 samples (24× expected). Fixed by switching to standard `Trainer` + `DataCollatorForLanguageModeling`. Steps normalised to 891/epoch.

**PyTorch 2.6 checkpoint error** — `UnpicklingError` on resume due to `weights_only=True` default in PyTorch 2.6. Fixed by removing `rng_state.pth` from checkpoint directory before resuming.

---

## Repository Structure
