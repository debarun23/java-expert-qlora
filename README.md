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
