"""
src/train.py
------------
Fine-tunes the pretrained BLIP VQA model on a configurable VQAv2 subset.

This is genuine fine-tuning of existing pretrained weights (loaded via
src/model.py -> load_model()), NOT training an architecture from scratch.

Run this as a script (`python src/train.py`) or import `run_training(cfg)`
from a notebook (see notebooks/03_finetuning.ipynb).

Everything that affects the run (batch size, LR, epochs, subset size, ...)
comes from configs/config.yaml - nothing here is hard-coded.
"""

import csv
import os
import random
from functools import partial

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.model import load_model, save_model, get_device
from src.dataset import load_vqa_subset, VQADataset, collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(cfg: dict):
    set_seed(cfg["reproducibility"]["seed"])
    device = get_device()
    print(f"[train.py] Using device: {device}")

    # --- Load pretrained model (this is the ONLY place weights come from) ---
    processor, model, device, _ = load_model(model_name=cfg["model"]["name"])

    # --- Data ---
    print("[train.py] Loading VQAv2 subset (train)...")
    train_split = load_vqa_subset(
        cfg["dataset"]["hf_dataset_name"], "train",
        cfg["dataset"]["train_subset_size"], cfg["dataset"]["seed"],
    )
    print("[train.py] Loading VQAv2 subset (validation)...")
    val_split = load_vqa_subset(
        cfg["dataset"]["hf_dataset_name"], "validation",
        cfg["dataset"]["val_subset_size"], cfg["dataset"]["seed"],
    )

    train_ds = VQADataset(train_split)
    val_ds = VQADataset(val_split)

    collate = partial(
        collate_fn,
        processor=processor,
        max_question_length=cfg["model"]["max_question_length"],
        max_answer_length=cfg["model"]["max_answer_length"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate
    )

    # --- Optimizer / scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    total_steps = (len(train_loader) // cfg["training"]["gradient_accumulation_steps"]) * cfg["training"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["fp16"] and device.type == "cuda")

    os.makedirs(os.path.dirname(cfg["training"]["log_file"]), exist_ok=True)
    log_rows = []

    for epoch in range(cfg["training"]["num_epochs"]):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=cfg["training"]["fp16"] and device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss / cfg["training"]["gradient_accumulation_steps"]

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg["training"]["gradient_accumulation_steps"]

            if (step + 1) % cfg["training"]["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = running_loss / max(len(train_loader), 1)

        # ---- Validate ----
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_running_loss += outputs.loss.item()
        val_loss = val_running_loss / max(len(val_loader), 1)

        print(f"[train.py] Epoch {epoch + 1}/{cfg['training']['num_epochs']} "
              f"- train_loss={train_loss:.4f} - val_loss={val_loss:.4f}")
        log_rows.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if cfg["training"]["save_every_epoch"]:
            save_model(model, processor, cfg["training"]["output_dir"])

    # Always save the final model, even if save_every_epoch is False
    save_model(model, processor, cfg["training"]["output_dir"])

    # Write the loss log as a plain CSV so it's easy to plot later / paste
    # into EXPERIMENT_LOG.md
    with open(cfg["training"]["log_file"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train.py] Loss log written to {cfg['training']['log_file']}")

    return log_rows


if __name__ == "__main__":
    cfg = load_config()
    run_training(cfg)
