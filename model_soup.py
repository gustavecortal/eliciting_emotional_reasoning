"""Merge LoRA checkpoints via linear or TIES soup as configured."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Any, List
import argparse

import torch
import yaml
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_adapter_dirs(output_dir: str) -> List[str]:
    """Return sorted adapter dirs like .../checkpoint-XXXX (all checkpoints)."""
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output dir not found: {output_dir}")
    pattern = re.compile(r"^checkpoint-(\d+)$")
    items = []
    for d in os.listdir(output_dir):
        m = pattern.match(d)
        if m:
            step = int(m.group(1))
            items.append((step, os.path.join(output_dir, d)))
    if not items:
        raise FileNotFoundError(f"No checkpoint-* folders under {output_dir}")
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def dtype_from_str(name: str) -> torch.dtype:
    """Map dtype string to torch dtype."""
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return mapping.get(name, torch.bfloat16)


def build_model_with_adapters(base_dir: str, adapter_dirs: List[str], device_map: str, torch_dtype: torch.dtype, trust_remote_code: bool) -> PeftModel:
    """Load base and mount all adapters by name."""
    base = AutoModelForCausalLM.from_pretrained(
        base_dir, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
    )
    model = PeftModel.from_pretrained(base, adapter_dirs[0], adapter_name=os.path.basename(adapter_dirs[0]))
    for d in adapter_dirs[1:]:
        model.load_adapter(d, adapter_name=os.path.basename(d))
    model.eval()
    return model


def save_merged(model, tokenizer, save_dir: str) -> None:
    """Save merged model and tokenizer."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

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
    """Run model soup merges as configured in config.yaml."""

    args = parse_args()
    config_path: Path = args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    #config_path = "config.yaml"
    cfg = load_config(str(config_path))
    print(f"Launch {cfg}...")
    
    #cfg = load_config("config.yaml")
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    soup_cfg = training_cfg.get("soup", {})
    output_dir = training_cfg.get("output_dir") or training_cfg["orpo"]["output_dir"]
    combination_types: List[str] = list(soup_cfg.get("combination_types", ["linear"]))
    weights_cfg = soup_cfg.get("weights")
    ties_density = float(soup_cfg.get("density", 0.2))
    save_dir_root = output_dir
    device_map = 'cuda:0' #str(training_cfg.get("device_map", "auto"))
    torch_dtype = dtype_from_str(training_cfg.get("torch_dtype", "bfloat16"))
    trust_remote_code = bool(model_cfg.get("tokenizer_trust_remote_code", True))

    adapter_dirs = find_adapter_dirs(output_dir)
    print(f"Using adapters: {adapter_dirs}")

    # Infer base from the first adapter
    peft_cfg = PeftConfig.from_pretrained(adapter_dirs[0])
    base_dir = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True, trust_remote_code=trust_remote_code)

    # Weights
    adapter_names = [os.path.basename(d) for d in adapter_dirs]
    if weights_cfg is None:
        weights = [1.0 for _ in adapter_names]
    else:
        if len(weights_cfg) != len(adapter_names):
            raise ValueError("weights length must match number of adapters")
        weights = [float(w) for w in weights_cfg]

    # Do each requested merge independently
    for comb in combination_types:
        comb = str(comb).lower().strip()
        if comb not in {"linear", "dare_ties"}:
            print(f"Skipping unsupported combination_type: {comb}")
            continue

        model = build_model_with_adapters(base_dir, adapter_dirs, device_map, torch_dtype, trust_remote_code)

        merged_adapter_name = f"merge_{comb}"
        add_kwargs: Dict[str, Any] = {}
        if comb == "dare_ties":
            add_kwargs["density"] = ties_density

        model.add_weighted_adapter(
            adapters=adapter_names,
            weights=weights,
            adapter_name=merged_adapter_name,
            combination_type=comb,
            **add_kwargs,
        )
        model.set_adapter(merged_adapter_name)

        merged = model.merge_and_unload()
        # Choose save path
        target = Path(save_dir_root) / comb
        save_merged(merged, tokenizer, str(target))
        print(f"Saved merged model to: {target}")


if __name__ == "__main__":
    main()
