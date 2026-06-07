#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup


IGNORE_INDEX = -100


class JsonlSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int = 1536):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt = row["prompt"].strip()
        target = row["target"].strip()
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        input_ids = enc["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
        return {"input_ids": input_ids, "labels": labels, "id": row.get("id")}


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        n = len(item["input_ids"])
        pad = max_len - n
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [IGNORE_INDEX] * pad)
        attention_mask.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


@torch.no_grad()
def eval_loss(model: Any, loader: DataLoader, device: str, max_batches: int = 30) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=25)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = JsonlSFTDataset(args.train, tokenizer, args.max_length)
    val_ds = JsonlSFTDataset(args.val, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, int(math.ceil(args.epochs * steps_per_epoch)))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    start = time.time()
    global_step = 0
    running = 0.0
    opt.zero_grad(set_to_none=True)

    num_update_target = total_steps
    epoch = 0
    while global_step < num_update_target:
        epoch += 1
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss.detach().cpu()) * args.grad_accum
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 5 == 0:
                    elapsed = time.time() - start
                    avg = running / 5
                    running = 0.0
                    print(json.dumps({"step": global_step, "epoch": epoch, "train_loss": avg, "lr": sched.get_last_lr()[0], "elapsed_s": round(elapsed, 1)}, ensure_ascii=False), flush=True)
                if global_step % args.eval_every == 0:
                    vl = eval_loss(model, val_loader, device)
                    print(json.dumps({"step": global_step, "val_loss": vl, "val_ppl": math.exp(min(20, vl))}, ensure_ascii=False), flush=True)
                if global_step % args.save_every == 0:
                    ckpt = args.outdir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                if global_step >= num_update_target:
                    break

    final = args.outdir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    vl = eval_loss(model, val_loader, device, max_batches=999)
    summary = {"global_step": global_step, "final_val_loss": vl, "final_val_ppl": math.exp(min(20, vl)), "train_rows": len(train_ds), "val_rows": len(val_ds)}
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
