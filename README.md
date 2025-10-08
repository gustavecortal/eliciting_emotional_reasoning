## Lightweight preference optimization using ORPO and LoRA

This repo fine-tunes Hugging Face models for preference optimization using ORPO + LoRA.

If you want the cheapest way to align an LLM without a reference model, you are in the right place.
Using LoRA with a small rank, **if you have enough compute for inference, then you probably have enough for fine-tuning**.

From my experiments, ORPO + LoRA works well and benefits from model souping (averaging checkpoints).

* `prepare_dataset.py` — downloads a preference dataset, wraps it in chat format, filters by length, and saves to disk.
* `train_orpo.py` — fine-tunes the model with ORPO + LoRA.
* `model_soup.py` — merges multiple LoRA checkpoints into a full model.

Usage: `python train_orpo.py --config configs/config_0.6b.yaml`


### What are ORPO and LoRA?

**ORPO (Odds Ratio Preference Optimization).** A reference-model-free preference objective: for each `(x, y⁺, y⁻)` pair, it adds a log-odds term that boosts the likelihood of the chosen response and penalizes the rejected one, so alignment happens in a single SFT-style stage (no PPO/DPO or separate ref model). ([arXiv][1])

**LoRA (Low-Rank Adaptation).** Keeps the pretrained weights frozen and learns tiny low-rank matrices on selected layers. These adapters are small, swappable, and can be merged into the base model for export. ([Hugging Face][2])

[1]: https://arxiv.org/abs/2403.07691 "ORPO: Monolithic Preference Optimization without Reference Model"
[2]: https://huggingface.co/docs/peft/main/en/developer_guides/lora

