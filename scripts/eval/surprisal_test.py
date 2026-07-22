#!/usr/bin/env python3
"""
Surprisal-reduction test for Tier 2 viability (clean version).

Uses llm.reset() between evals — the proven pattern from autocorrect_diag.py.

Compares 5 scoring methods. The key question: does surprisal reduction
(log P(word|ctx) − log P(word)) beat naive logprob for contextual re-ranking?
"""
import llama_cpp
import numpy as np
import math

MODEL = "gguf/eu_futo_v2.gguf"
llm = llama_cpp.Llama(model_path=MODEL, n_ctx=2048, n_gpu_layers=0,
                      verbose=False, logits_all=True)

def token_logprobs(tokens, start_idx):
    """Evaluate tokens, return list of logprobs for tokens[start_idx:]."""
    llm.reset()
    llm.eval(tokens)
    logits = llm.eval_logits
    lps = []
    for i in range(start_idx, len(tokens)):
        if i == 0:
            continue
        row = np.array(logits[i - 1], dtype=np.float64)
        lp = row[tokens[i]] - math.log(np.exp(row - row.max()).sum())
        lps.append(lp)
    return lps

def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n

def score_word(context, word, use_bos):
    """Return (ctx_sum, ctx_n, base_sum, base_n)."""
    # Baseline: " word" (leading space matches in-context tokenization)
    base_tokens = llm.tokenize(f" {word}".encode("utf-8"), add_bos=use_bos)
    base_start = 1 + (1 if use_bos else 0)
    base_lps = token_logprobs(base_tokens, base_start)

    # In-context
    ctx_tokens = llm.tokenize(context.encode("utf-8"), add_bos=use_bos)
    full_tokens = llm.tokenize(f"{context} {word}".encode("utf-8"), add_bos=use_bos)
    cpl = common_prefix_len(ctx_tokens, full_tokens)
    cand_start = max(cpl, 1)
    ctx_lps = token_logprobs(full_tokens, cand_start)

    return (sum(ctx_lps), len(ctx_lps), sum(base_lps), len(base_lps))

cases = [
    ("Poliziak atxilotu egin du", "mutila", ["musika", "mutiko"]),
    ("Kalean zebilen", "mutila", ["musika", "mutiko"]),
    ("Nire anaia", "mutila", ["musika", "mutiko"]),
    ("Haurrak eta", "mutila", ["musika", "mutiko"]),
    ("Goizean goiz esan dio", "kaixo", ["musika", "mutila"]),
    ("Familiaren", "etxea", ["musika", "mutila"]),
    ("Euskal", "herria", ["musika", "mutila"]),
    ("Ongi", "etorri", ["musika", "mutila"]),
    ("Gose naiz, janari", "nahi", ["musika", "mutila"]),
    ("Euria ari", "du", ["musika", "mutila"]),
]

for use_bos in [False, True]:
    label = "WITH BOS (wllama default)" if use_bos else "NO BOS (correct for this model)"
    print(f"\n{'#'*72}\n#  {label}\n{'#'*72}")
    methods = [
        ("sum_logprob",    lambda c, b, cn, bn: c),
        ("per_token",      lambda c, b, cn, bn: c/cn if cn else -999),
        ("surprisal_sum",  lambda c, b, cn, bn: c - b),
        ("surprisal_ptok", lambda c, b, cn, bn: (c-b)/cn if cn else -999),
        ("surprisal_ratio",lambda c, b, cn, bn: (c/cn if cn else 0)-(b/bn if bn else 0)),
    ]
    for mname, mfn in methods:
        correct = 0
        detail = []
        for context, right, wrongs in cases:
            scored = {}
            for c in [right] + wrongs:
                cs, cn, bs, bn = score_word(context, c, use_bos)
                scored[c] = mfn(cs, bs, cn, bn)
            winner = max(scored, key=lambda w: scored[w])
            ok = winner == right
            correct += ok
            mark = "✓" if ok else "✗"
            detail.append(f"    {mark} {context[:24]:<26} → {winner:<8} (want {right})  " +
                          "  ".join(f"{c}={scored[c]:+.3f}" for c in [right]+wrongs))
        print(f"\n  [{mname}]  {correct}/{len(cases)} correct")
        for line in detail:
            print(line)
