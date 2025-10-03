"""Prepare and save the ORPO training dataset in chat format."""
from pathlib import Path
from typing import Dict, Any

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import apply_chat_template


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_role(example: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Wrap fields as chat messages with roles."""
    prompt_text = f"{example['prompt']}{suffix}"
    example["prompt"] = [{"role": "user", "content": prompt_text}]
    example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
    example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]
    return example


def filter_max_length(example: Dict[str, Any], tokenizer, max_len: int) -> bool:
    """Return True if prompt/chosen/rejected lengths are within max_len."""
    prompt_len = len(tokenizer(example["prompt"][0]["content"])["input_ids"])
    chosen_len = len(tokenizer(example["chosen"][0]["content"])["input_ids"])
    rejected_len = len(tokenizer(example["rejected"][0]["content"])["input_ids"])
    return prompt_len <= max_len and chosen_len <= max_len and rejected_len <= max_len


def main() -> None:
    """Prepare dataset using YAML-configured hyperparameters."""
    config_path = Path("config.yaml")
    cfg = load_config(str(config_path))

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    base_model_path: str = model_cfg["base_path"]
    trust_remote_code: bool = bool(model_cfg.get("tokenizer_trust_remote_code", True))

    dataset_name: str = dataset_cfg["name"]
    save_path: str = dataset_cfg["save_path"]
    add_suffix: str = dataset_cfg.get("add_no_think_suffix", "")
    max_length: int = int(dataset_cfg["max_length"])
    num_proc: int = int(dataset_cfg.get("num_proc", 1))
    limit_train = dataset_cfg.get("limit_train")
    limit_test = dataset_cfg.get("limit_test")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True, trust_remote_code=trust_remote_code
    )

    dataset = load_dataset(dataset_name)
    if "question" in dataset["train"].column_names:
        dataset = dataset.rename_column("question", "prompt")

    if limit_train is not None:
        dataset["train"] = dataset["train"].select(range(int(limit_train)))
    if "test" in dataset and limit_test is not None:
        dataset["test"] = dataset["test"].select(range(int(limit_test)))

    print(len(dataset["train"]))

    # Add chat roles
    dataset = dataset.map(
        add_role,
        fn_kwargs={"suffix": add_suffix},
        num_proc=num_proc,
    )

    # Filter by tokenized length
    dataset = dataset.filter(
        filter_max_length,
        fn_kwargs={"tokenizer": tokenizer, "max_len": max_length},
        num_proc=num_proc,
    )

    print(len(dataset["train"]))
    print(dataset)

    # Apply chat template using TRL helper (previous correct implementation)
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=num_proc,
    )

    print(dataset)

    # Save to disk
    save_dir = Path(save_path)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_dir))


if __name__ == "__main__":
    main()
