"""
Phase 4 (unified): Multi-task finetune from the pretrain checkpoint.

This REPLACES the old sequential 4a → 4b → 4c pipeline, which caused
catastrophic format contamination (the model always emitted ``<XBU>`` as
top-1 after plain text, making next-word prediction 0%).

STRATEGY (per senior-ML-engineger review):
  Start from the **Phase 3 pretrain checkpoint** (loss 4.33 — healthy
  plain-text model) and train on a single balanced mixture:

    60% pure plain text (Latxa v2 / BSM)  → PLW = 1.0 on every token
    40% isolated autocorrect triples       → mask prompt, PLW = 1.0 on
                                             correction span + <XEC> only

The two streams are strictly segregated at the sequence level (never
inline-mixed). Because the model receives zero gradient for spontaneously
generating ``<XBU>`` (the triple prompt is masked), the transition
``[plain text] → <XBU>`` is never rewarded — which is correct, because the
FUTO C++ inference layer *injects* ``<XBU>`` itself when a keypress
correction is needed.

This balanced pressure dedicates a subset of attention capacity to the
structural autocorrect format while leaving the lexical next-word pathways
intact.

HYPERPARAMETERS:
  - LR: 5e-5 cosine → 5e-6  (moderate; decaying)
  - Steps: 18K       (~660M plain + ~440M triple tokens)
  - Mix: 60% plain / 40% triples
  - seq_len: 512

INPUT:  pretrain/base/  + corpora/clean (Latxa v2) + notes/{synth,real}.json
OUTPUT: finetune/stage_m/final/

Usage:
  uv run python -m scripts.finetune.multitask \\
      --config configs/phase4_multitask.yaml --mode full \\
      --base pretrain/base \\
      --tokenizer tokenizer/spm_eu.model \\
      --corpus corpora/clean \\
      --synth-jsonl notes/synth.json \\
      --real-jsonl notes/real.json
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

from scripts.lib.datasets import MultiTaskFinetuneDataset
from scripts.lib.progress import ProgressCallback
from scripts.lib.plw_trainer import PLWTrainer, SAMPLWTrainer
from scripts.lib.real_eval_callback import RealTypoEvalCallback
from scripts.lib.runconfig import load_config, pick


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase4_multitask.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full", "recover"],
                    help="Which mode section to load from the config")
    ap.add_argument("--base", required=True,
                    help="Base checkpoint dir (pretrain/base — the healthy plain-text model)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", default=None,
                    help="Dir of shard_*.txt (default: corpora/clean — Latxa v2)")
    ap.add_argument("--synth-jsonl", default=None,
                    help="synth.json from generate_triples.py")
    ap.add_argument("--real-jsonl", default=None,
                    help="real.json from generate_triples.py")
    ap.add_argument("--out", default="finetune/stage_m")
    # Mix parameters
    ap.add_argument("--plain-ratio", type=float, default=None,
                    help="Fraction of samples from plain text (default 0.60)")
    ap.add_argument("--real-mix-ratio", type=float, default=None,
                    help="Fraction of triple samples from real typos (default 0.25)")
    # Training params
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--save-total-limit", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-grad-norm", type=float, default=None,
                    help="Gradient clipping norm (default 1.0)")
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
    corpus = pick(args.corpus, cfg, "corpus", "corpora/clean")
    synth_jsonl = pick(args.synth_jsonl, cfg, "synth_jsonl", "notes/synth.json")
    real_jsonl = pick(args.real_jsonl, cfg, "real_jsonl", "notes/real.json")
    plain_ratio = pick(args.plain_ratio, cfg, "plain_ratio", 0.60)
    real_mix_ratio = pick(args.real_mix_ratio, cfg, "real_mix_ratio", 0.25)
    total_steps = pick(args.total_steps, cfg, "total_steps", 18000)
    seq_len = pick(args.seq_len, cfg, "seq_len", 512)
    micro_batch = pick(args.micro_batch, cfg, "micro_batch", 24)
    grad_accum = pick(args.grad_accum, cfg, "grad_accum", 8)
    lr = pick(args.lr, cfg, "lr", 5.0e-5)
    warmup = pick(args.warmup, cfg, "warmup_steps", 200)
    save_every = pick(args.save_every, cfg, "save_every", 2000)
    save_total_limit = pick(args.save_total_limit, cfg, "save_total_limit", 3)
    num_workers = pick(args.num_workers, cfg, "num_workers", 4)
    max_grad_norm = pick(args.max_grad_norm, cfg, "max_grad_norm", 1.0)
    seed = pick(args.seed, cfg, "seed", 1337)

    set_seed(seed)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading base checkpoint: {args.base}")
    model = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    shards = sorted(glob.glob(str(Path(corpus) / "shard_*.txt")))
    if not shards:
        raise SystemExit(f"No shards in {corpus} — run Phase 1 first.")
    print(f"Found {len(shards)} clean-text shards in {corpus}")

    train_ds = MultiTaskFinetuneDataset(
        shard_paths=shards,
        synth_jsonl=synth_jsonl,
        real_jsonl=real_jsonl,
        sp_model_path=args.tokenizer,
        seq_len=seq_len,
        plain_ratio=plain_ratio,
        real_mix_ratio=real_mix_ratio,
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
        max_grad_norm=max_grad_norm,
        bf16=True,
        logging_steps=50,
        save_steps=save_every,
        save_total_limit=save_total_limit,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if args.wandb_project else [],
        seed=seed,
        disable_tqdm=False,
        run_name="phase4_multitask",
        remove_unused_columns=False,  # keep loss_weights
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [ProgressCallback(phase="stage_m", seq_len=seq_len, log_path=progress_log)]
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

    print(f"Starting Phase 4 (unified multi-task): {total_steps} steps, "
          f"lr={lr} (cosine), plain_ratio={plain_ratio}, "
          f"global batch {micro_batch * grad_accum}, seq_len {seq_len}")
    print(f"  plain text:  {corpus} (PLW=1.0 — next-word prediction)")
    print(f"  triples:     {synth_jsonl} + {real_jsonl} (isolated, prompt masked)")
    print(f"  base:        {args.base}")
    print(f"  grad clip:   max_grad_norm={max_grad_norm}")
    print(f"Progress log: {progress_log}")
    if args.eval_jsonl:
        print(f"Real-typo eval: every {args.eval_every} steps → {out}/real_typo_eval.csv")

    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"Saved Phase 4 multi-task final checkpoint to {out}/final/")


if __name__ == "__main__":
    main()
