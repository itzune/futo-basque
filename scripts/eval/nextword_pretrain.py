#!/usr/bin/env python3
"""Pretrain-level next-word evaluation.

WHY THIS EXISTS
---------------
diag_objective.py proves the pretrain model learned the right *objective*
(next-token loss low, skip-1 loss high). But the previous catastrophe was
specifically next-WORD prediction at 8.3% top-1. "Low token loss" and
"predicts the right next word" are related but not identical, so this script
gives the direct empirical answer: does the model actually predict sensible
next words?

This runs on a RAW pretrain checkpoint (no FUTO control tokens needed), so it
works BEFORE finetuning. It is the right gate to pass before investing in
phase 4m.

WHAT IT MEASURES
----------------
  1. Hand-curated next-word tests (config/eu.py NEXT_WORD_TESTS):
       12 novel Basque prefixes, each with a set of plausible next words.
       Greedy-decode the next word; check top-1 + top-5 hit.
       (These are NOT memorized sentences -> honest generalization signal.)
  2. Corpus-based next-word accuracy:
       ~50 held-out-ish lines from the clean corpus. For each, take the prefix
       up to word N, greedy-decode the next word, compare to the gold word.
       top-1 exact + top-5 exact (top-5 via beam search).

BOS: prepended (sp.bos_id()), matching how the model was trained and how FUTO
     infers (LanguageModel.kt inserts BOS=1 at context start).

Runs on CPU (float32) so it does not disturb a live GPU training run.

Usage
-----
  python eval_nextword_pretrain.py <checkpoint> [tokenizer] [n_corpus]
  python eval_nextword_pretrain.py pretrain/checkpoint-5000
  python eval_nextword_pretrain.py pretrain/checkpoint-5000 tokenizer/spm_eu.model 50
"""
import sys, os, random, torch
from transformers import LlamaForCausalLM
import sentencepiece as spm

CKPT = sys.argv[1]
SP_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(__file__), "..", "..", "tokenizer", "spm_eu.model")
N_CORPUS = int(sys.argv[3]) if len(sys.argv) > 3 else 50

# import the hand-curated test set
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.eu import NEXT_WORD_TESTS

CORPUS = os.path.join(os.path.dirname(__file__), "..", "..", "corpora", "clean", "shard_00003.txt")

print(f"checkpoint : {CKPT}")
print(f"tokenizer  : {SP_PATH}")
print(f"device     : cpu  (float32)")
print()

sp = spm.SentencePieceProcessor()
sp.load(SP_PATH)
bos, eos, pad = sp.bos_id(), sp.eos_id(), sp.pad_id()
print(f"  bos={bos} eos={eos} pad={pad} vocab={sp.get_piece_size()}")

model = LlamaForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32)
model.eval().to("cpu")
print(f"  model loaded, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n")


@torch.no_grad()
def greedy_next_word(prefix: str, max_new_tokens: int = 6) -> str:
    """Greedy-decode from prefix (with BOS) until a word boundary (space/EOS)."""
    ids = [bos] + sp.encode(prefix, out_type=int)
    # ensure the prefix ends with a space so the first generated token starts a word
    if not prefix.endswith(" "):
        ids += [sp.piece_to_id("▁")] if sp.piece_to_id("▁") >= 0 else []
    t = torch.tensor([ids], dtype=torch.long)
    out = model.generate(t, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, pad_token_id=pad, eos_token_id=eos)
    gen = out[0, len(ids):].tolist()
    txt = sp.decode(gen)
    # first word only
    w = txt.strip().split()
    return w[0].strip(",.;:!?") if w else ""


@torch.no_grad()
def beam_next_words(prefix: str, k: int = 5, max_new_tokens: int = 6) -> list[str]:
    """Beam-search k continuations, return the first word of each."""
    ids = [bos] + sp.encode(prefix, out_type=int)
    if not prefix.endswith(" "):
        p = sp.piece_to_id("▁")
        if p >= 0:
            ids += [p]
    t = torch.tensor([ids], dtype=torch.long)
    beams = model.generate(t, max_new_tokens=max_new_tokens, num_beams=k,
                           num_return_sequences=k, do_sample=False,
                           pad_token_id=pad, eos_token_id=eos, early_stopping=True)
    words = []
    for i in range(beams.shape[0]):
        gen = beams[i, len(ids):].tolist()
        txt = sp.decode(gen)
        w = txt.strip().split()
        if w:
            words.append(w[0].strip(",.;:!?").lower())
    return words


# ── Part 1: hand-curated next-word tests ────────────────────────────────── #
print("=" * 64)
print("PART 1 — Hand-curated next-word tests (novel prefixes)")
print("=" * 64)
t1, t5 = 0, 0
for prefix, plausible in NEXT_WORD_TESTS:
    top1 = greedy_next_word(prefix)
    topk = beam_next_words(prefix, 5)
    plaus = {p.lower() for p in plausible}
    h1 = top1.lower() in plaus
    h5 = bool(set(topk) & plaus)
    t1 += h1; t5 += h5
    flag = "✓" if h1 else ("◯" if h5 else "✗")
    print(f"  {flag} {prefix!r:<22} → top1={top1!r:<14} top5={topk}")
    print(f"      plausible: {plausible}")
n = len(NEXT_WORD_TESTS)
print(f"\n  top-1: {t1}/{n} = {100*t1/n:.1f}%   top-5: {t5}/{n} = {100*t5/n:.1f}%\n")

# ── Part 2: corpus-based next-word accuracy ─────────────────────────────── #
print("=" * 64)
print(f"PART 2 — Corpus-based next-word accuracy ({N_CORPUS} lines)")
print("=" * 64)
lines = []
if os.path.exists(CORPUS):
    with open(CORPUS, encoding="utf-8") as f:
        pool = [ln.strip() for ln in f if len(ln.strip().split()) >= 6]
    random.seed(1337)
    random.shuffle(pool)
    lines = pool[:N_CORPUS]

c1, c5, total = 0, 0, 0
shown = 0
for ln in lines:
    words = ln.split()
    # pick a split point ~60% into the sentence
    idx = max(2, int(len(words) * 0.6))
    prefix = " ".join(words[:idx]) + " "
    gold = words[idx].strip(",.;:!?").lower()
    if not gold:
        continue
    total += 1
    top1 = greedy_next_word(prefix)
    topk = beam_next_words(prefix, 5)
    h1 = top1.lower() == gold
    h5 = gold in [w.lower() for w in topk]
    c1 += h1; c5 += h5
    if shown < 8 or h1:  # show first 8 + all hits
        flag = "✓" if h1 else ("◯" if h5 else "✗")
        print(f"  {flag} …{prefix[-30:]!r:<32} → {top1!r:<14} (gold {gold!r})")
        shown += 1
print(f"\n  top-1 exact: {c1}/{total} = {100*c1/total:.1f}%")
print(f"  top-5 exact: {c5}/{total} = {100*c5/total:.1f}%")
print(f"  (note: corpus lines are training data, so this is a lower-bound /")
print(f"   memorization-friendly signal; Part 1 is the honest generalization test.)")
