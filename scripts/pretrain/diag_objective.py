#!/usr/bin/env python3
"""Definitive diagnostic: is a checkpoint learning next-token or skip-1?

Background
----------
The pretrain double-shift bug trained the model on the SKIP-1 objective
P(t[i+2] | t[0..i]) instead of next-token P(t[i+1] | t[0..i]). Both objectives
make loss go down, so a falling loss curve is NOT proof the model is healthy.

This script computes BOTH losses on held-out Basque text:
  - next-token loss = model(input_ids=ids, labels=ids).loss
        HF shifts internally -> measures P(t[i+1] | t[0..i]).  [CORRECT]
  - skip-1 loss     = model(input_ids=ids[:-1], labels=ids[1:]).loss
        the exact broken objective -> measures P(t[i+2] | t[0..i]).  [BUG]

Decision rule
-------------
  next-token < skip-1  ->  HEALTHY (learned the right objective)
  skip-1     < next-token  ->  BROKEN (still the double-shift bug)

Runs on CPU so it does not disturb a live GPU training run.

Usage
-----
  python diag_objective.py <checkpoint_dir> [n_lines]
  python diag_objective.py pretrain/checkpoint-24000
  python diag_objective.py pretrain/checkpoint-5000 5
"""
import sys, os, random, torch
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

CKPT = sys.argv[1]
N_LINES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tokenizer", "spm_eu.model")
CORPUS = os.path.join(os.path.dirname(__file__), "..", "..", "corpora", "clean", "shard_00003.txt")

print(f"checkpoint : {CKPT}")
print(f"tokenizer  : {SP_PATH}")
print(f"device     : cpu  (float32)")
print()

sp = spm.SentencePieceProcessor()
sp.load(SP_PATH)
bos, eos = sp.bos_id(), sp.eos_id()

cfg = LlamaConfig.from_pretrained(CKPT)
model = LlamaForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32)
model.eval()
model.to("cpu")

# grab real corpus lines (reproducible seed)
lines = []
if os.path.exists(CORPUS):
    with open(CORPUS, encoding="utf-8") as f:
        pool = [ln.strip() for ln in f if len(ln.strip()) > 40]
    random.seed(1337)
    random.shuffle(pool)
    lines = pool[:N_LINES]
if len(lines) < N_LINES:
    # fallback hardcoded Basque
    fb = [
        "Egunon guztiei eta ongi etorri etxera gaur.",
        "Euskara Eurobiguneren hizkuntza ofiziala izango da aurrerantzean.",
        "Nire izena Mikel da eta orain Bilbon bizi naiz.",
        "Gaur eguzkia atera da eta eguraldia oso ona da.",
        "Bihar goizean goiz abiatuko gara mendira ibilaldi batera.",
    ]
    lines = (lines + fb)[:N_LINES]

@torch.no_grad()
def both_losses(ids):
    t = torch.tensor([ids], dtype=torch.long)
    nt = model(input_ids=t, labels=t).loss.item()                 # correct
    sk = model(input_ids=t[:, :-1], labels=t[:, 1:]).loss.item()  # broken obj
    return nt, sk

print(f"{'line':<6}{'next-token':>12}{'skip-1':>10}   verdict")
print("-" * 50)
healthy = 0
for i, ln in enumerate(lines):
    ids = [bos] + sp.encode(ln, out_type=int) + [eos]
    nt, sk = both_losses(ids)
    ok = nt < sk
    healthy += ok
    flag = "HEALTHY (next-token)" if ok else "BROKEN (skip-1)"
    print(f"line{i:<4}{nt:>12.3f}{sk:>10.3f}   {flag}")

print("-" * 50)
verdict = "HEALTHY" if healthy == len(lines) else ("BROKEN" if healthy == 0 else "MIXED")
print(f"\nVERDICT: {verdict}  ({healthy}/{len(lines)} lines next-token < skip-1)")
print("random baseline (uniform over vocab) = ln(vocab) = "
      f"{torch.log(torch.tensor(float(cfg.vocab_size))):.3f}")
