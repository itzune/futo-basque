"""Validate the pre-tokenization optimization.

Pre-tokenizes the corpus shard once (SentencePiece -> flat uint16 bin, memmap'd),
then runs the REAL HF Trainer with a trivial streaming dataset (no per-step
tokenization, no Python packing). If throughput jumps from ~160k tok/s toward
the 247k tok/s pure-compute ceiling, the on-the-fly tokenization was the
bottleneck and pre-tokenizing is a free ~1.5x speedup for the real training.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset
import sentencepiece as spm
from transformers import Trainer, TrainingArguments, set_seed

from scripts.pretrain.train import build_model
from scripts.lib.progress import ProgressCallback


def pretokenize(shard_paths: list[str], sp_path: str, out_bin: str):
    if Path(out_bin).exists():
        n = os.path.getsize(out_bin) // 2
        print(f"[pretok] cached {out_bin} ({n} tokens)")
        return
    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)
    bos, eos = sp.bos_id(), sp.eos_id()
    all_ids: list[np.ndarray] = []
    total = 0
    for p in shard_paths:
        print(f"[pretok] tokenizing {p}")
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = np.array([bos] + sp.encode(line, out_type=int) + [eos],
                               dtype=np.uint16)
                all_ids.append(ids)
                total += len(ids)
    arr = np.concatenate(all_ids)
    m = np.memmap(out_bin, dtype=np.uint16, mode="w+", shape=arr.shape)
    m[:] = arr[:]
    m.flush()
    print(f"[pretok] wrote {out_bin}: {len(arr)} tokens")


class PreTokDataset(IterableDataset):
    """Stream fixed-length slices from a memmap'd uint16 token bin."""
    def __init__(self, bin_path: str, seq_len: int = 1024, seed: int = 1337):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        arr = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        sl = self.seq_len
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        nw = wi.num_workers if wi else 1
        chunk = len(arr) // nw
        start = wid * chunk
        end = (wid + 1) * chunk if wid < nw - 1 else len(arr)
        i = start
        while i + sl + 1 <= end:
            ids = arr[i:i + sl + 1].astype(np.int64)
            yield {
                "input_ids": torch.from_numpy(ids[:sl]),
                "labels": torch.from_numpy(ids[1:sl + 1]),
            }
            i += sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="/tmp/tokens.bin")
    ap.add_argument("--tokenizer", default="reference_model/extracted_spm.model")
    ap.add_argument("--corpus", default="corpora/clean")
    ap.add_argument("--total-steps", type=int, default=100)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    import glob
    shards = sorted(glob.glob(str(Path(args.corpus) / "shard_*.txt")))
    pretokenize(shards, args.tokenizer, args.bin)

    set_seed(1337)
    ds = PreTokDataset(args.bin, seq_len=args.seq_len)
    model = build_model().to(torch.bfloat16)

    targs = TrainingArguments(
        output_dir="bench_pretok",
        max_steps=args.total_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=3e-4,
        warmup_steps=2000,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        save_steps=999999,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=[],
        seed=1337,
        disable_tqdm=False,
    )
    Path("bench_pretok").mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        callbacks=[ProgressCallback(phase="pretok", seq_len=args.seq_len,
                                    log_path="bench_pretok/progress.log")],
    )
    print(f"Starting pretok bench: {args.total_steps} steps, "
          f"global {args.micro_batch * args.grad_accum}, seq {args.seq_len}")
    trainer.train()
    print("DONE")


if __name__ == "__main__":
    main()
