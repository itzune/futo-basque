"""
Phase 4d: Clean-text recovery finetune.

PROBLEM: After Phases 4a/4b/4c, the model suffers 100% format contamination —
it ALWAYS emits ``<XBU>`` as top-1 after plain text, making next-word
prediction completely non-functional (0% top-1 in real inference testing).

ROOT CAUSE: Phase 4b (fulltext) trains on corpus where ~33% of words are
replaced with ``<XBU>typo<XBC>correct<XEC>`` triples INLINE. This creates a
strong association: plain text → ``<XBU>``. The PLW=0.05 mitigation was not
enough — the model still predicts ``<XBU>`` as top-1 after any plain text.

SOLUTION: Train on a MIX of two data types, kept SEPARATE (not inline-mixed):

  1. **Pure plain text** (75%): Clean Latxa v2 corpus, NO typo injection,
     NO ``<XBU>`` tokens at all. Full loss on every token (PLW=1.0).
     Teaches: "plain context → plain word."

  2. **Isolated autocorrect triples** (25%): Same Phase 4a data
     (``<XBU><CHAR_*>...<XBC>correct<XEC>``). Loss on correction span only
     (PLW=0.0). Preserves: "autocorrect context → format."

By keeping these SEPARATE (not inline-mixed like 4b), we break the
"plain text → ``<XBU>``" association. The model sees:
  - Plain text sequences that NEVER contain ``<XBU>`` → learns word continuation
  - Isolated triples that START with ``<XBU>`` → retains autocorrect format

These are conditionally independent given context, so the model can learn both.

HYPERPARAMETERS (gentle recovery):
  - LR: 1e-5 (lowest of all phases — don't catastrophically forget autocorrect)
  - Steps: 3000 (enough to shift next-word distribution, not enough to forget)
  - Mix: 75% plain / 25% triples
  - seq_len: 512 (same as 4b/4c)

INPUT:  finetune/stage_c/final/ + corpora/clean (Latxa v2) + notes/{synth,real}.json
OUTPUT: finetune/stage_d/final/

Usage:
  uv run python -m scripts.finetune.recovery \\
      --config configs/phase4d_recovery.yaml --mode full \\
      --base finetune/stage_c/final \\
      --tokenizer tokenizer/spm_eu.model \\
      --corpus corpora/clean \\
      --synth-jsonl notes/synth.json \\
      --real-jsonl notes/real.json
"""
from __future__ import annotations
import argparse
import glob
import json
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

from scripts.lib.typo_synthesis import make_xbu_triple
from scripts.lib.progress import ProgressCallback
from scripts.lib.plw_trainer import (
    PLWTrainer,
    SAMPLWTrainer,
    build_loss_weights_for_correction_only,
)
from scripts.lib.real_eval_callback import RealTypoEvalCallback
from scripts.lib.runconfig import load_config, pick


class RecoveryDataset(IterableDataset):
    """Interleaves pure plain text (majority) with isolated autocorrect triples.

    For each sample, draws from plain text with probability ``plain_ratio``
    or from isolated triples with probability ``1 - plain_ratio``.

    Plain text: streamed from corpus shards, packed into ``seq_len`` sequences,
    ALL loss weights = 1.0 (we want the model to learn next-word prediction).

    Triples: random draw from synth/real JSONL, padded to ``seq_len``,
    loss weights = correction-span-only (PLW=0.0, same as Phase 4a).
    """

    def __init__(
        self,
        shard_paths: list[str],
        synth_jsonl: str,
        real_jsonl: str,
        sp_model_path: str,
        seq_len: int = 512,
        plain_ratio: float = 0.75,
        real_mix_ratio: float = 0.25,
        seed: int = 1337,
        shuffle_buffer: int = 1024,
    ):
        self.shard_paths = sorted(shard_paths)
        self.synth_pairs = json.loads(Path(synth_jsonl).read_text())
        self.real_pairs = json.loads(Path(real_jsonl).read_text())
        self.sp_model_path = sp_model_path
        self.seq_len = seq_len
        self.plain_ratio = plain_ratio
        self.real_mix_ratio = real_mix_ratio
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        print(
            f"[dataset] shards={len(self.shard_paths)} "
            f"synth={len(self.synth_pairs)} real={len(self.real_pairs)} "
            f"plain_ratio={plain_ratio:.2f} real_mix_ratio={real_mix_ratio:.2f}"
        )

    def _iter_plain_shards(self, worker_id: int, num_workers: int):
        """Yield raw text lines from corpus shards (worker-sharded)."""
        for i, path in enumerate(self.shard_paths):
            if i % num_workers != worker_id:
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def _iter_plain_sequences(self, sp, bos, eos, worker_id, num_workers, rng):
        """Pack plain text into seq_len chunks (no <XBU>, no typo injection).

        All loss_weights = 1.0 — we want the model to learn next-word on
        every token. attention_mask is all-1 (fully packed, no padding).
        """
        buffer: list[int] = []
        line_buffer: list[str] = []

        for line in self._iter_plain_shards(worker_id, num_workers):
            line_buffer.append(line)
            if len(line_buffer) >= self.shuffle_buffer:
                rng.shuffle(line_buffer)
                for raw in line_buffer:
                    buffer.append(bos)
                    buffer.extend(sp.encode(raw, out_type=int))
                    buffer.append(eos)
                    while len(buffer) >= self.seq_len:
                        ids = buffer[: self.seq_len]
                        del buffer[: self.seq_len]
                        yield {
                            "input_ids": torch.tensor(ids, dtype=torch.long),
                            "labels": torch.tensor(ids, dtype=torch.long),
                            "loss_weights": torch.ones(
                                self.seq_len, dtype=torch.float32
                            ),
                            "attention_mask": torch.ones(
                                self.seq_len, dtype=torch.long
                            ),
                        }
                line_buffer.clear()

    def _iter_triples(self, sp, bos, eos, pad, xbc_id, xec_id, rng):
        """Yield isolated <XBU>typo<XBC>correct<XEC> triples, padded to seq_len.

        loss_weights: 1.0 for correction span (XBC..XEC), 0.0 elsewhere
        (same as Phase 4a with PLW=0.0).
        """
        while True:
            if rng.random() < self.real_mix_ratio and self.real_pairs:
                pair = rng.choice(self.real_pairs)
            else:
                pair = rng.choice(self.synth_pairs)
            typo, correct = pair["typed"], pair["committed"]
            if not typo or not correct or typo == correct:
                continue
            triple = make_xbu_triple(typo, correct)
            ids = [bos] + sp.encode(triple, out_type=int) + [eos]
            if len(ids) > self.seq_len:
                continue
            input_ids = ids + [pad] * (self.seq_len - len(ids))
            labels = [t if t != pad else -100 for t in input_ids]
            loss_weights = build_loss_weights_for_correction_only(
                input_ids,
                xbc_id=xbc_id,
                xec_id=xec_id,
                plw_clean=0.0,
                in_span_weight=1.0,
            )
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
                "attention_mask": torch.tensor(
                    [1 if t != pad else 0 for t in input_ids], dtype=torch.long
                ),
            }

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id * 9973)

        sp = spm.SentencePieceProcessor()
        sp.load(self.sp_model_path)
        bos = sp.bos_id()
        eos = sp.eos_id()
        pad = sp.pad_id()
        xbc_id = sp.piece_to_id("<XBC>")
        xec_id = sp.piece_to_id("<XEC>")

        plain_iter = self._iter_plain_sequences(
            sp, bos, eos, worker_id, num_workers, rng
        )
        triple_iter = self._iter_triples(sp, bos, eos, pad, xbc_id, xec_id, rng)

        while True:
            if rng.random() < self.plain_ratio:
                try:
                    yield next(plain_iter)
                except StopIteration:
                    # Plain corpus exhausted — fall through to triples only.
                    # (In practice with 3B-token corpus and 3K steps, this
                    # won't happen, but handle it gracefully.)
                    yield next(triple_iter)
            else:
                yield next(triple_iter)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase4d_recovery.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--base", required=True,
                    help="Phase 4c final checkpoint dir (finetune/stage_c/final)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", default=None,
                    help="Dir of shard_*.txt (default: corpora/clean — Latxa v2)")
    ap.add_argument("--synth-jsonl", default=None,
                    help="synth.json from generate_triples.py")
    ap.add_argument("--real-jsonl", default=None,
                    help="real.json from generate_triples.py")
    ap.add_argument("--out", default="finetune/stage_d")
    # Mix parameters
    ap.add_argument("--plain-ratio", type=float, default=None,
                    help="Fraction of samples from plain text (default 0.75)")
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
    plain_ratio = pick(args.plain_ratio, cfg, "plain_ratio", 0.75)
    real_mix_ratio = pick(args.real_mix_ratio, cfg, "real_mix_ratio", 0.25)
    total_steps = pick(args.total_steps, cfg, "total_steps", 3000)
    seq_len = pick(args.seq_len, cfg, "seq_len", 512)
    micro_batch = pick(args.micro_batch, cfg, "micro_batch", 24)
    grad_accum = pick(args.grad_accum, cfg, "grad_accum", 8)
    lr = pick(args.lr, cfg, "lr", 1.0e-5)
    warmup = pick(args.warmup, cfg, "warmup_steps", 200)
    save_every = pick(args.save_every, cfg, "save_every", 1000)
    save_total_limit = pick(args.save_total_limit, cfg, "save_total_limit", 3)
    num_workers = pick(args.num_workers, cfg, "num_workers", 4)
    seed = pick(args.seed, cfg, "seed", 1337)

    set_seed(seed)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading Phase 4c checkpoint: {args.base}")
    model = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    shards = sorted(glob.glob(str(Path(corpus) / "shard_*.txt")))
    if not shards:
        raise SystemExit(f"No shards in {corpus} — run Phase 1 first.")
    print(f"Found {len(shards)} clean-text shards in {corpus}")

    train_ds = RecoveryDataset(
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
        bf16=True,
        logging_steps=50,
        save_steps=save_every,
        save_total_limit=save_total_limit,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if args.wandb_project else [],
        seed=seed,
        disable_tqdm=False,
        run_name="phase4d_recovery",
        remove_unused_columns=False,  # keep loss_weights
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [ProgressCallback(phase="stage_d", seq_len=seq_len, log_path=progress_log)]
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

    print(f"Starting Phase 4d (clean-text recovery): {total_steps} steps, "
          f"lr={lr}, plain_ratio={plain_ratio}, "
          f"global batch {micro_batch * grad_accum}, seq_len {seq_len}")
    print(f"  corpus:      {corpus} (Latxa v2 — pure plain text, no <XBU>)")
    print(f"  triples:     {synth_jsonl} + {real_jsonl} (isolated, PLW=0.0)")
    print(f"  base:        {args.base} (Phase 4c checkpoint)")
    print(f"Progress log: {progress_log}")
    if args.eval_jsonl:
        print(f"Real-typo eval: every {args.eval_every} steps → {out}/real_typo_eval.csv")

    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"Saved Phase 4d final checkpoint to {out}/final/")


if __name__ == "__main__":
    main()
