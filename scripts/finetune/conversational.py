"""
Phase 4c: conversational adaptation fine-tune  (NEW — was missing from v1).

FUTO's own wiki (GitLab Keyboard-LM-docs) describes this stage:

  "the model is finetuned on a much smaller corpus representing Internet
   first-person speech with the same 1/3 misspelling augmentation. This has
   appeared to be an important step for the model to comprehend that
   sentences can start with 'I'm', certain slang/lingo, etc."

Without it, the model suggests Wikipedia-register continuations ("Furthermore,
...") instead of chat-register ("yeah, ..."). This is the stage the Portuguese
community guide includes as Phase 4c and we were missing (see RESEARCH.md
§11.6.1).

The dataset is BERnaT BSM (corpora/conversational/), aggressively cleaned by
Phase 1b. The typo augmentation stays at 1/3 (same as Phase 4b per FUTO wiki)
so the model still does autocorrect — just now on conversational text. The
learning rate is lower (2e-5 vs 5e-5) because this is adaptation, not
retraining.

Reuses ``InlineCorruptedDataset`` from Phase 4b (same inline-typo-injection
logic, different corpus + hyperparams).

INPUT:  finetune/stage_b/final/ + corpora/conversational (BERnaT BSM)
OUTPUT: finetune/stage_c/final/

Usage:
  uv run python -m scripts.finetune.conversational \\
      --config configs/phase4c_conversational.yaml --mode full \\
      --base finetune/stage_b/final \\
      --tokenizer tokenizer/spm_eu.model
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path

import torch
from transformers import (
    LlamaForCausalLM,
    TrainingArguments,
    set_seed,
)

from scripts.lib.runconfig import load_config, pick
from scripts.lib.typo_synthesis import make_inline_corrected  # noqa: F401 (re-exported by fulltext import)
from scripts.lib.progress import ProgressCallback
from scripts.lib.plw_trainer import PLWTrainer, SAMPLWTrainer, build_loss_weights_for_xbu
from scripts.lib.real_eval_callback import RealTypoEvalCallback
# Reuse the exact same dataset class as Phase 4b (inline typo injection).
from scripts.finetune.fulltext import InlineCorruptedDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase4c_conversational.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--base", required=True, help="Phase 4b final checkpoint dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", default=None,
                    help="Dir of shard_*.txt (default: corpora/conversational)")
    ap.add_argument("--out", default="finetune/stage_c")
    # All training params default to None → resolved from config or hard default.
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--typo-rate", type=float, default=None)
    ap.add_argument("--plw", type=float, default=None,
                    help="Prompt Loss Weight for clean tokens. 1.0 = full-sequence loss.")
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--save-total-limit", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wandb-project", type=str, default="futo-eu")
    ap.add_argument("--progress-log", type=str, default=None)
    ap.add_argument("--eval-jsonl", type=str, default=None,
                    help="Path to real_typos_eval.json. If set, runs real-typo eval.")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--use-sam", action="store_true",
                    help="Enable Sharpness-Aware Minimization (doubles per-step cost).")
    ap.add_argument("--sam-rho", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)

    # Resolve all params: CLI > config > hard default.
    corpus = pick(args.corpus, cfg, "corpus", "corpora/conversational")
    total_steps = pick(args.total_steps, cfg, "total_steps", 5000)
    seq_len = pick(args.seq_len, cfg, "seq_len", 512)
    micro_batch = pick(args.micro_batch, cfg, "micro_batch", 24)
    grad_accum = pick(args.grad_accum, cfg, "grad_accum", 8)
    lr = pick(args.lr, cfg, "lr", 2.0e-5)
    warmup = pick(args.warmup, cfg, "warmup_steps", 200)
    typo_rate = pick(args.typo_rate, cfg, "typo_rate", 0.33)
    plw = pick(args.plw, cfg, "plw", 1.0)
    save_every = pick(args.save_every, cfg, "save_every", 1000)
    save_total_limit = pick(args.save_total_limit, cfg, "save_total_limit", 3)
    num_workers = pick(args.num_workers, cfg, "num_workers", 4)
    seed = pick(args.seed, cfg, "seed", 1337)

    set_seed(seed)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading Phase 4b checkpoint: {args.base}")
    model = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    shards = sorted(glob.glob(str(Path(corpus) / "shard_*.txt")))
    if not shards:
        raise SystemExit(f"No shards in {corpus} — run Phase 1b first.")
    print(f"Found {len(shards)} conversational shards in {corpus}")

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
        run_name="phase4c_conversational",
        remove_unused_columns=False,  # keep loss_weights
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [ProgressCallback(phase="stage_c", seq_len=seq_len, log_path=progress_log)]
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

    print(f"Starting Phase 4c (conversational adaptation): {total_steps} steps, "
          f"lr={lr}, typo_rate={typo_rate}, plw={plw}, "
          f"global batch {micro_batch * grad_accum}, seq_len {seq_len}")
    print(f"  corpus: {corpus} (BERnaT BSM)")
    print(f"  base:   {args.base} (Phase 4b checkpoint)")
    print(f"Progress log: {progress_log}")
    if args.eval_jsonl:
        print(f"Real-typo eval: every {args.eval_every} steps → {out}/real_typo_eval.csv")

    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"Saved Phase 4c final checkpoint to {out}/final/")


if __name__ == "__main__":
    main()
