"""Pure-compute throughput sweep for the 36M FUTO Llama on a single GPU.

Mirrors scripts.pretrain.train.build_model exactly, runs real AdamW
forward+backward+step loops in bf16 with gradient_accumulation, and reports
sec/step, steps/sec, tokens/sec, and peak VRAM for several micro-batch sizes.

No dataloader, no CPU tokenization — isolates GPU compute (the dominant cost
for a 36M model). Cross-check against the real Trainer run for overhead.
"""
from __future__ import annotations
import time
import torch

from scripts.pretrain.train import build_model

SEQ = 1024
VOCAB = 4096
STEPS = 20      # measured optimizer steps (after warmup)
WARMUP = 5


def bench(micro: int, accum: int):
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model = build_model(vocab_size=VOCAB).to(torch.bfloat16).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    gbatch = micro * accum

    def run(n):
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                x = torch.randint(0, VOCAB, (micro, SEQ), device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, labels=x)
                (out.loss / accum).backward()
            opt.step()

    run(WARMUP)
    torch.cuda.synchronize()
    t0 = time.time()
    run(STEPS)
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = STEPS / dt
    tps = sps * gbatch * SEQ
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return dt / STEPS, sps, tps, mem


def main():
    configs = [(16, 16), (32, 8), (64, 4), (96, 3), (128, 2), (192, 1), (256, 1)]
    print(f"{'mb':>4} {'ga':>4} {'gb':>4} {'s/step':>9} {'step/s':>9} "
          f"{'tok/s':>11} {'VRAM_GB':>8}")
    print("-" * 56)
    for micro, accum in configs:
        try:
            sps_dt, sps, tps, mem = bench(micro, accum)
            print(f"{micro:>4} {accum:>4} {micro*accum:>4} {sps_dt:>9.3f} "
                  f"{sps:>9.3f} {tps:>11,.0f} {mem:>8.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{micro:>4} {accum:>4} {micro*accum:>4}   OOM")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
