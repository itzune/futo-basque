"""
Phase 3: pretrain the 36M-param Llama base model on Basque (eu).

Architecture matches the reference English FUTO model (only vocab_size differs):
  - vocab_size=4096 (reference uses 15008; we use 4096 — see RESEARCH.md §11.3.1)
  - hidden=512, ffn=1024, layers=8, heads=8, head_dim=64
  - max_position=2048, rms_norm_eps=1e-6, rope_theta=10000, MHA (no GQA)
  - tie_word_embeddings=False (output and token_embd are separate tensors)

Designed to run on the RTX 3090 (24 GB) inside the Unraid Docker container.
The 5070 Ti (16 GB) can also run this with reduced micro-batch.

Usage on gpu-train host (3090 container):
  cd /workspace
  source env/bin/activate
  uv run python -m scripts.pretrain.train \\
      --tokenizer tokenizer/spm_eu.model \\
      --corpus corpora/clean \\
      --out pretrain \\
      --total-steps 150000 \\
      --micro-batch 16 --grad-accum 16 \\
      --wandb-project futo-eu
"""
from __future__ import annotations
import argparse
import glob
import os
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader
import sentencepiece as spm
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from scripts.lib.progress import ProgressCallback
from scripts.lib.runconfig import load_config, pick


# Verified architecture (notes/reference_metadata.txt + futo_model_schema memory).
# vocab_size is passed in (not hardcoded) so it matches the trained tokenizer.
# Reference English model uses 15008; we use 4096 for Basque (see RESEARCH.md §11.3.1
# — morpheus ablation: 4K vocab → 66.7% MorphAcc vs 28.6% at 32K).
def build_model(vocab_size: int, arch: dict | None = None) -> LlamaForCausalLM:
    """Build the Llama base model.

    Architecture defaults match the reference English FUTO model (only
    vocab_size differs). Pass ``arch`` from the config to override any field.
    See configs/phase3_pretrain.yaml for the canonical architecture spec.
    """
    arch = arch or {}
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=arch.get("hidden_size", 512),
        intermediate_size=arch.get("intermediate_size", 1024),
        num_hidden_layers=arch.get("num_hidden_layers", 8),
        num_attention_heads=arch.get("num_attention_heads", 8),
        num_key_value_heads=arch.get("num_key_value_heads", 8),           # MHA, no GQA
        max_position_embeddings=arch.get("max_position_embeddings", 2048),
        rms_norm_eps=arch.get("rms_norm_eps", 1e-6),               # NOT 1e-5
        rope_theta=arch.get("rope_theta", 10000.0),
        tie_word_embeddings=arch.get("tie_word_embeddings", False),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n/1e6:.2f}M (vocab_size={vocab_size})")
    return model


class EuShardStreamer(IterableDataset):
    """
    Streams from `shard_*.txt` files, tokenizes on the fly, packs sequences
    to a fixed length. Worker-aware: each DataLoader worker picks a disjoint
    subset of shards by index modulo, so workers don't read the same data.
    """
    def __init__(self, shard_paths: list[str], sp_model_path: str, seq_len: int = 1024,
                 shuffle_buffer: int = 1024, seed: int = 1337):
        self.shard_paths = sorted(shard_paths)
        self.sp_model_path = sp_model_path
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _iter_shards(self, worker_id: int, num_workers: int):
        # Each worker gets shards indexed [worker_id::num_workers]
        for i, path in enumerate(self.shard_paths):
            if i % num_workers != worker_id:
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id)

        sp = spm.SentencePieceProcessor()
        sp.load(self.sp_model_path)
        bos, eos = sp.bos_id(), sp.eos_id()

        buffer: list[int] = []
        line_buffer: list[str] = []

        for line in self._iter_shards(worker_id, num_workers):
            line_buffer.append(line)
            if len(line_buffer) >= self.shuffle_buffer:
                rng.shuffle(line_buffer)
                for ln in line_buffer:
                    buffer.append(bos)
                    buffer.extend(sp.encode(ln, out_type=int))
                    buffer.append(eos)
                    while len(buffer) >= self.seq_len:
                        ids = buffer[: self.seq_len]
                        del buffer[: self.seq_len]
                        # input_ids and labels are the SAME sequence; HF
                        # LlamaForCausalLM.forward applies the causal shift
                        # internally when `labels` are passed. Pre-shifting here
                        # (ids[:-1]/ids[1:]) caused a DOUBLE shift: the model
                        # learned P(token[i+2] | token[i]) (skip-1), making it
                        # functionally random for next-token prediction (inference
                        # loss 8.0 vs 4.33 skip-1). Fixed to match finetune
                        # datasets (fulltext.py / datasets.py).
                        yield {
                            "input_ids": torch.tensor(ids, dtype=torch.long),
                            "labels": torch.tensor(ids, dtype=torch.long),
                        }
                line_buffer.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase3_pretrain.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--tokenizer", required=True, help="Path to spm_eu.model")
    ap.add_argument("--corpus", required=True, nargs="+",
                    help="One or more directories of shard_*.txt files. "
                         "Pass both tiers to mix clean + conversational: "
                         "--corpus corpora/clean corpora/conversational")
    ap.add_argument("--out", default="pretrain", help="Output directory for checkpoints")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None,
                    help="Training sequence length. <=2048 (architectural max).")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wandb-project", type=str, default="futo-eu")
    ap.add_argument("--resume-from", type=str, default=None)
    ap.add_argument("--progress-log", type=str, default=None,
                    help="File to mirror compact [progress] lines into (default: <out>/progress.log)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)

    # Resolve all params: CLI > config > hard default.
    total_steps = pick(args.total_steps, cfg, "total_steps", 150_000)
    seq_len = pick(args.seq_len, cfg, "seq_len", 1024)
    micro_batch = pick(args.micro_batch, cfg, "micro_batch", 16)
    grad_accum = pick(args.grad_accum, cfg, "grad_accum", 16)
    lr = pick(args.lr, cfg, "lr", 3e-4)
    warmup = pick(args.warmup, cfg, "warmup_steps", 2000)
    weight_decay = pick(args.weight_decay, cfg, "weight_decay", 0.1)
    save_every = pick(args.save_every, cfg, "save_every", 5000)
    num_workers = pick(args.num_workers, cfg, "num_workers", 4)
    seed = pick(args.seed, cfg, "seed", 1337)
    arch = cfg.get("model", {})  # architecture overrides (empty = use defaults)

    set_seed(seed)

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Glob shard_*.txt across all provided corpus dirs (e.g. clean + conversational).
    shards = []
    for c in args.corpus:
        shards += glob.glob(str(Path(c) / "shard_*.txt"))
    shards = sorted(shards)
    if not shards:
        raise SystemExit(f"No shards found in {args.corpus}")
    print(f"Found {len(shards)} corpus shards across {len(args.corpus)} dir(s): {args.corpus}")

    train_ds = EuShardStreamer(
        shard_paths=shards,
        sp_model_path=args.tokenizer,
        seq_len=seq_len,
        seed=seed,
    )

    # Read vocab size from the tokenizer so model embeddings match exactly.
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.vocab_size()
    print(f"Tokenizer vocab_size: {vocab_size}")
    assert vocab_size >= 560, f"vocab too small ({vocab_size}); need >= 560 for structural+byte slots"

    model = build_model(vocab_size=vocab_size, arch=arch)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        max_steps=total_steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=False,    # 25M params + 24GB = no need
        logging_steps=50,
        save_steps=save_every,
        save_total_limit=5,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,  # avoid worker re-init at epoch boundary
        report_to=["wandb"] if args.wandb_project else [],
        seed=seed,
        disable_tqdm=False,
        # Streaming dataset so eval split is awkward; skip eval during pretrain
        # and rely on perplexity logging plus manual sample-generation in eval script.
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        callbacks=[ProgressCallback(phase="pretrain", seq_len=seq_len, log_path=progress_log)],
    )

    print(f"Starting pretrain: {total_steps} steps, "
          f"global batch = {micro_batch}*{grad_accum} = "
          f"{micro_batch * grad_accum}, seq_len={seq_len}")
    print(f"Progress log: {progress_log}")

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(str(out / "base"))
    print(f"Saved final checkpoint to {out}/base/")


if __name__ == "__main__":
    main()
