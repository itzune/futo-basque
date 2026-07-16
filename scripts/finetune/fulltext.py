"""
Phase 4b: In-context autocorrect fine-tune.

Takes real Basque sentences from the corpus, randomly replaces ~33% of words
with `<XBU>typo<XBC>correct<XEC>` triples (the wiki target), and continues
training from the Stage 4a checkpoint.

Loss formulation: weighted cross-entropy via Prompt Loss Weight (PLW).
  --plw 1.0   full-sequence loss (every token contributes equally — the
              v1/v2 historical behavior that mode-collapsed)
  --plw 0.05  (Phase 1 B1 recommended) per arxiv 2401.13586 for short
              completions. Correction spans get full weight, clean tokens
              get 0.05. Diagnosed fix for stage_b/c mode collapse.

Off-by-one fix: previous version emitted input_ids=ids[:-1] and labels=ids[1:]
of equal length, but HF LlamaForCausalLM does internal shift on labels —
double-shift bug. Now input_ids and labels are the same sequence; HF shifts.
"""
from __future__ import annotations
import argparse
import glob
import os
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
import sentencepiece as spm
from transformers import (
    LlamaForCausalLM,
    TrainingArguments,
    set_seed,
)

from scripts.lib.typo_synthesis import make_inline_corrected
from scripts.lib.progress import ProgressCallback
from scripts.lib.plw_trainer import PLWTrainer, SAMPLWTrainer, build_loss_weights_for_xbu
from scripts.lib.real_eval_callback import RealTypoEvalCallback
from scripts.lib.runconfig import load_config, pick


class InlineCorruptedDataset(IterableDataset):
    """
    Streams from corpus shards, injects typos at ~typo_rate, packs into
    fixed-length sequences. Emits loss_weights so PLWTrainer can weight
    XBU-span tokens (the correction signal) higher than clean text.
    """
    def __init__(self, shard_paths: list[str], sp_model_path: str,
                 seq_len: int = 512, typo_rate: float = 0.33,
                 plw_clean: float = 1.0, seed: int = 1337,
                 shuffle_buffer: int = 1024):
        self.shard_paths = sorted(shard_paths)
        self.sp_model_path = sp_model_path
        self.seq_len = seq_len
        self.typo_rate = typo_rate
        self.plw_clean = plw_clean
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _iter_shards(self, worker_id: int, num_workers: int):
        for i, path in enumerate(self.shard_paths):
            if i % num_workers != worker_id:
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id * 9973)

        sp = spm.SentencePieceProcessor()
        sp.load(self.sp_model_path)
        bos = sp.bos_id()
        eos = sp.eos_id()
        xbu_id = sp.piece_to_id("<XBU>")
        xec_id = sp.piece_to_id("<XEC>")

        buffer: list[int] = []
        line_buffer: list[str] = []

        for line in self._iter_shards(worker_id, num_workers):
            line_buffer.append(line)
            if len(line_buffer) >= self.shuffle_buffer:
                rng.shuffle(line_buffer)
                for raw in line_buffer:
                    corrupted = make_inline_corrected(raw, rng, typo_rate=self.typo_rate)
                    buffer.append(bos)
                    buffer.extend(sp.encode(corrupted, out_type=int))
                    buffer.append(eos)
                    while len(buffer) >= self.seq_len:
                        ids = buffer[: self.seq_len]
                        del buffer[: self.seq_len]
                        # input_ids and labels are the SAME sequence; HF shifts
                        # internally. (Previous code pre-shifted, causing a
                        # double-shift / off-by-one bug.)
                        loss_weights = build_loss_weights_for_xbu(
                            ids, xbu_id=xbu_id, xec_id=xec_id,
                            plw_clean=self.plw_clean, in_span_weight=1.0,
                        )
                        yield {
                            "input_ids": torch.tensor(ids, dtype=torch.long),
                            "labels": torch.tensor(ids, dtype=torch.long),
                            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
                        }
                line_buffer.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase4b_fulltext.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--base", required=True, help="Phase 4a final checkpoint dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", default=None, help="Dir of shard_*.txt")
    ap.add_argument("--out", default="finetune/stage_b")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--typo-rate", type=float, default=None)
    ap.add_argument("--plw", type=float, default=None,
                    help="Prompt Loss Weight for clean (non-XBU-span) tokens. "
                         "1.0 = full-sequence loss (v2 default). "
                         "0.05 = recommended fix per arxiv 2401.13586.")
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--save-total-limit", type=int, default=None,
                    help="HF Trainer save_total_limit (keeps last N checkpoints).")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wandb-project", type=str, default="futo-eu")
    ap.add_argument("--progress-log", type=str, default=None)
    ap.add_argument("--eval-jsonl", type=str, default=None,
                    help="Path to notes/real_typos_eval.json. If set, runs real-typo "
                         "eval every --eval-every steps and writes a CSV.")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--use-sam", action="store_true",
                    help="Enable Sharpness-Aware Minimization (doubles per-step cost).")
    ap.add_argument("--sam-rho", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)

    # Resolve all params: CLI > config > hard default.
    corpus = pick(args.corpus, cfg, "corpus", "corpora/clean")
    total_steps = pick(args.total_steps, cfg, "total_steps", 40000)
    seq_len = pick(args.seq_len, cfg, "seq_len", 512)
    micro_batch = pick(args.micro_batch, cfg, "micro_batch", 24)
    grad_accum = pick(args.grad_accum, cfg, "grad_accum", 8)
    lr = pick(args.lr, cfg, "lr", 5e-5)
    warmup = pick(args.warmup, cfg, "warmup_steps", 1000)
    typo_rate = pick(args.typo_rate, cfg, "typo_rate", 0.33)
    plw = pick(args.plw, cfg, "plw", 1.0)
    save_every = pick(args.save_every, cfg, "save_every", 2000)
    save_total_limit = pick(args.save_total_limit, cfg, "save_total_limit", 3)
    num_workers = pick(args.num_workers, cfg, "num_workers", 4)
    seed = pick(args.seed, cfg, "seed", 1337)

    set_seed(seed)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading base checkpoint: {args.base}")
    model = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    shards = sorted(glob.glob(str(Path(corpus) / "shard_*.txt")))
    if not shards:
        raise SystemExit(f"No shards in {corpus}")
    print(f"Found {len(shards)} shards")

    train_ds = InlineCorruptedDataset(
        shard_paths=shards,
        sp_model_path=args.tokenizer,
        seq_len=seq_len,
        typo_rate=typo_rate,
        plw_clean=plw,
        seed=seed,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        max_steps=total_steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        save_steps=save_every,
        save_total_limit=save_total_limit,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if args.wandb_project else [],
        seed=seed,
        disable_tqdm=False,
        run_name="phase4b_fulltext",
        remove_unused_columns=False,  # keep loss_weights
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [ProgressCallback(phase="stage_b", seq_len=seq_len, log_path=progress_log)]
    if args.eval_jsonl:
        callbacks.append(RealTypoEvalCallback(
            eval_jsonl=args.eval_jsonl,
            sp_model_path=args.tokenizer,
            eval_every=args.eval_every,
            csv_path=str(out / "real_typo_eval.csv"),
        ))

    trainer_cls = SAMPLWTrainer if args.use_sam else PLWTrainer
    trainer_kwargs = {"sam_rho": args.sam_rho} if args.use_sam else {}
    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=train_ds,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    print(f"Starting Phase 4b: {total_steps} steps, plw={plw}, "
          f"typo_rate={typo_rate}, global batch {micro_batch * grad_accum}, "
          f"seq_len {seq_len}")
    print(f"Progress log: {progress_log}")
    if args.eval_jsonl:
        print(f"Real-typo eval: every {args.eval_every} steps → {out}/real_typo_eval.csv")

    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"Saved Phase 4b final checkpoint to {out}/final/")


if __name__ == "__main__":
    main()
