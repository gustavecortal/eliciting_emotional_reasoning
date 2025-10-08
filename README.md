## Simple ORPO training with LoRA

This workspace fine-tunes HuggingFace chat models using ORPO and LoRA. ORPO training pairs preferred vs. dispreferred completions and optimizes the odds ratio. LoRA training adapts large base models by learning low-rank adapter weights that can be merged or swapped without touching the full model.

- `prepare_dataset.py` — downloads a preference dataset, wraps it in chat format, filters by length, and saves it to disk based on settings in `config.yaml`.  
  Usage: `python prepare_dataset.py`

- `train_orpo.py` — fine-tunes the base model with LoRA adapters using the prepared dataset and ORPO training hyperparameters from a YAML config.  
  Usage: `python train_orpo.py --config configs/config_0.6b.yaml`

- `model_soup.py` — merges multiple LoRA checkpoints into a single adapter or full model (linear or TIES soup) as configured in the YAML file.  
  Usage: `python model_soup.py --config configs/config_0.6b.yaml`
