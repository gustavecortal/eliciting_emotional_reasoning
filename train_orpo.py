""""Train ORPO with LoRA using a YAML-configured setup."""
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ORPOConfig, ORPOTrainer

import os

os.environ["WANDB_MODE"] = "offline"


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def enable_tf32() -> None:
    """Enable TF32 on matmul and cudnn if available."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ORPO with LoRA using a YAML-configured setup."
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    """Run ORPO training using hyperparameters from a YAML config."""
    args = parse_args()
    config_path: Path = args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    #config_path = "config.yaml"
    cfg = load_config(str(config_path))
    print(f"Launch {cfg}...")

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    dataset_path: str = dataset_cfg["save_path"]

    base_model_path: str = model_cfg["base_path"]
    trust_remote_code: bool = bool(model_cfg.get("tokenizer_trust_remote_code", True))
    resume_from_checkpoint: bool = bool(training_cfg.get("resume_from_checkpoint", False))
    torch_dtype_str: str = training_cfg.get("torch_dtype", "bfloat16")
    #device_map: str = str(training_cfg.get("device_map", "auto"))
    allow_tf32: bool = bool(training_cfg.get("allow_tf32", True))

    lora_params: Dict[str, Any] = training_cfg["lora"]
    orpo_params: Dict[str, Any] = training_cfg["orpo"]
    save_model_path: str = training_cfg["output_dir"]

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True, trust_remote_code=trust_remote_code
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=False,          # set True only if needed
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,          # <-- add this
        device_map=None,           # <-- change from None to CPU-only
    )

    if allow_tf32:
        enable_tf32()

    dataset = load_from_disk(dataset_path)
    print(dataset)

    lora_config = LoraConfig(
        r=int(lora_params["r"]),
        lora_alpha=int(lora_params["lora_alpha"]),
        lora_dropout=float(lora_params["lora_dropout"]),
        bias=str(lora_params["bias"]),
        task_type=str(lora_params["task_type"]),
        target_modules=list(lora_params["target_modules"]),
    )

    orpo_config = ORPOConfig(
        output_dir=save_model_path,
        num_train_epochs=int(orpo_params["num_train_epochs"]),
        learning_rate=float(orpo_params["learning_rate"]),
        per_device_train_batch_size=int(orpo_params["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(orpo_params["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(orpo_params["gradient_accumulation_steps"]),
        logging_steps=int(orpo_params["logging_steps"]),
        eval_strategy=str(orpo_params.get("eval_strategy", "steps")),
        eval_steps=int(orpo_params["eval_steps"]),
        save_steps=int(orpo_params["save_steps"]),
        max_length=int(orpo_params["max_length"]),
        max_prompt_length=int(orpo_params["max_prompt_length"]),
        beta=float(orpo_params["beta"]),
        bf16=bool(orpo_params["bf16"]),
        remove_unused_columns=bool(orpo_params["remove_unused_columns"]),
        gradient_checkpointing=bool(orpo_params["gradient_checkpointing"]),
        report_to=list(orpo_params["report_to"]),
        warmup_ratio=float(orpo_params["warmup_ratio"]),
    )

    eval_dataset: Optional[object] = dataset["test"] if "test" in dataset else None

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)


if __name__ == "__main__":
    main()
