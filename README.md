# JavaExpert-Qwen2.5-3B: Domain-Locked Java QA via QLoRA on Consumer Hardware

> Fine-tuned Qwen2.5-3B on an 8GB consumer GPU using QLoRA to build a domain-locked Java QA assistant · Val Perplexity 2.40 · 7,921 QA Pairs

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-JavaExpert--Qwen2.5--3B-blue)](https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

Most LLM fine-tuning work happens on cloud A100s with large VRAM budgets. This project demonstrates a different approach: building a deployable Java programming assistant on a single 8GB consumer GPU using QLoRA.

The model was fine-tuned to:

* Answer Java programming questions accurately
* Refuse non-Java questions without external guardrails
* Run inference using less than 2GB VRAM
* Ship as a single merged model without adapter loading

---

## Key Results

| Metric                | Target | Achieved     |
| --------------------- | ------ | ------------ |
| Validation Perplexity | < 10   | **2.40**     |
| Peak Training VRAM    | ≤ 8 GB | **< 7 GB**   |
| Inference VRAM        | —      | **< 2 GB**   |
| Java Correctness      | —      | **8.5 / 10** |
| Domain Restriction    | —      | **8.5 / 10** |
| Hallucination Control | —      | **8.0 / 10** |

---

## Model

**Hugging Face Model**

https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B

### Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Debarun12/JavaExpert-Qwen2.5-3B"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Debarun12/JavaExpert-Qwen2.5-3B"
)
```

### Example Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Debarun12/JavaExpert-Qwen2.5-3B"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Debarun12/JavaExpert-Qwen2.5-3B"
)

def ask(question):

    messages = [
        {
            "role": "system",
            "content": "You are a Java programming assistant."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    outputs = model.generate(
        inputs,
        max_new_tokens=256
    )

    print(
        tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
    )

ask("What is polymorphism in Java?")
```

---

## Data Engineering Pipeline

### Source Corpus

* 42,000+ lines of Java documentation extracted from PDFs
* Significant PDF parsing noise and OCR artifacts
* Custom preprocessing pipeline built specifically for Java documentation

### Chunking Strategy

* Sliding window chunk size: 150 words
* Overlap: 30 words
* Preserves context across chunk boundaries

### Quality Filtering

Each chunk passes through `is_valid_chunk()` and is rejected if:

* Mean word length > 12 characters
* Missing Java-specific keywords
* Chunk length < 30 words
* Contains 20+ consecutive dots
* Matches chapter or section marker patterns

### QA Generation

Valid chunks were processed using Qwen2.5:7B through Ollama.

Four question archetypes were generated:

1. What questions
2. How questions
3. Why questions
4. Unanswerable questions

The unanswerable category provides the negative training signal that teaches the model when not to answer.

### Final Dataset

| Property              | Value            |
| --------------------- | ---------------- |
| Total QA Pairs        | 7,921            |
| Format                | Chat-style JSONL |
| Deduplication         | MD5 Hashing      |
| Estimated Cleanliness | 95%+             |

---

## QLoRA Fine-Tuning

### Why QLoRA

Full fp16 fine-tuning of a 3B parameter model requires roughly 14GB VRAM.

QLoRA enables training on consumer hardware by:

* Quantizing the base model to 4-bit NF4
* Keeping trainable adapter weights in bf16
* Preserving model quality while reducing memory usage

---

## Memory Optimization Decisions

| Component              | Initial             | Final                | VRAM Impact     |
| ---------------------- | ------------------- | -------------------- | --------------- |
| LoRA Rank              | 32 across 7 modules | 16 on q_proj, v_proj | -1.5 GB         |
| Batch Size             | 4                   | 1 + Grad Accum ×8    | -2.0 GB         |
| Optimizer              | AdamW               | Adafactor            | -1.0 GB         |
| Gradient Checkpointing | Enabled             | Disabled             | -1.0 GB         |
| Compute Type           | fp16                | bf16                 | Stable Training |

### Training Configuration

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

optim = "adafactor"

learning_rate = 2e-4

num_train_epochs = 3

bf16 = True

lora_r = 16

lora_target_modules = [
    "q_proj",
    "v_proj"
]
```

---

## Training Results

| Epoch | Train Loss | Validation Loss | Validation Perplexity |
| ----- | ---------- | --------------- | --------------------- |
| 1     | 0.8511     | 0.9343          | 2.55                  |
| 2     | 0.8367     | 0.8839          | 2.42                  |
| 3     | 0.7780     | 0.8772          | **2.40**              |

Validation loss decreased consistently across all epochs.

The narrow train-validation gap suggests stable generalisation rather than memorisation.

---

## Domain Restriction Examples

| Query                                      | Model Response                                                       |
| ------------------------------------------ | -------------------------------------------------------------------- |
| What is the capital of France?             | I don't have enough information in my knowledge base to answer this. |
| Who won the FIFA World Cup?                | I don't have enough information in my knowledge base to answer this. |
| What is polymorphism in Java?              | Correct Java explanation                                             |
| How does ArrayList differ from LinkedList? | Correct Java explanation                                             |

---

## Engineering Bugs Found and Fixed

### SFTTrainer Step Count Explosion

**Problem**

Using:

```python
packing=True
```

with pre-tokenized data produced:

* 2,139 training steps on 90 samples
* Approximately 24× the expected count

**Root Cause**

Packed sequence lengths were being interpreted incorrectly by SFTTrainer.

**Fix**

Replaced:

```python
SFTTrainer
```

with:

```python
Trainer
```

plus:

```python
DataCollatorForLanguageModeling
```

Result:

* 891 steps per epoch
* Expected training behavior restored

---

### PyTorch 2.6 Checkpoint Issue

**Problem**

Checkpoint resumption raised:

```python
UnpicklingError
```

after the PyTorch 2.6 default change:

```python
weights_only=True
```

**Fix**

Automatically removed:

```python
rng_state.pth
```

before checkpoint restoration.

---

## Limitations

### Domain Boundary Leakage

Domain-adjacent questions involving:

* Python
* Machine Learning
* Algorithms

may occasionally receive partial answers before refusal.

### No Retrieval Layer

The model can generalise from learned concepts but cannot retrieve exact facts from source documents.

---

## Future Work

### Retrieval-Augmented Generation (RAG)

Add corpus retrieval for stronger factual grounding.

### Direct Preference Optimisation (DPO)

Improve refusal quality on adjacent technical topics.

### More Negative Examples

Expand unanswerable examples focused on:

* Python
* Machine Learning
* General Computer Science

---

## Repository Structure

```text
.
├── data
│   ├── java_corpus.txt
│   └── qa_pairs_chat.jsonl
│
├── notebooks
│   ├── QNA_dataset_generation.ipynb
│   └── FineTuneQLoRA_final.ipynb
│
├── checkpoints
│
├── adapters
│
└── merged_model
```

---

## Installation

```bash
git clone https://github.com/debarun23/java-expert-qlora.git

cd java-expert-qlora

pip install -r requirements.txt
```

### Core Dependencies

```text
transformers>=4.40
peft>=0.10
bitsandbytes>=0.43
trl>=0.8
torch>=2.6
datasets>=2.18
accelerate>=0.28
```

---

## Hardware

Training Hardware:

* NVIDIA GeForce RTX 5050 Laptop GPU
* 8GB VRAM

No cloud compute was used for:

* Dataset generation
* Fine-tuning
* Evaluation
* Inference testing

---

## Author

### Debarun Das

**Model**

https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B

**GitHub Repository**

https://github.com/debarun23/java-expert-qlora

---

## Citation

```bibtex
@misc{das2026javaexpert,
  author       = {Debarun Das},
  title        = {JavaExpert-Qwen2.5-3B: Domain-Locked Java QA via QLoRA on Consumer Hardware},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Debarun12/JavaExpert-Qwen2.5-3B}}
}
```
