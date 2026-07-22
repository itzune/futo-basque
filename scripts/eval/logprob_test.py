#!/usr/bin/env python3
"""
Cleaner logprob test: futo model on raw text, with and without BOS.
Tests the agent's specific claim: "0/5 correct, raw text is OOD."
"""
import llama_cpp
import numpy as np
import math

MODEL = "gguf/eu_futo_v2.gguf"
llm = llama_cpp.Llama(model_path=MODEL, n_ctx=2048, n_gpu_layers=0,
                      verbose=False, logits_all=True)

def score(context, candidate, use_bos):
    """Score P(candidate | context) via per-token logprob. Proper tokenization."""
    # Tokenize the FULL string, find candidate boundary by re-tokenizing
    full = f"{context} {candidate}"
    tokens = llm.tokenize(full.encode("utf-8"), add_bos=use_bos)
    # Tokenize context-only to find boundary
    ctx_tok = llm.tokenize(context.encode("utf-8"), add_bos=use_bos)
    # The candidate starts somewhere after ctx_tok. Find it by decoding.
    # Simple approach: candidate tokens = tokens after the context portion.
    # Account for the space — tokenize "context " (with trailing space)
    ctx_plus_space = llm.tokenize(f"{context} ".encode("utf-8"), add_bos=use_bos)
    cand_start = len(ctx_plus_space)
    if cand_start >= len(tokens):
        # Fallback: tokenization merged differently, use ctx_tok length
        cand_start = len(ctx_tok)

    llm.eval(tokens)
    logits = llm.eval_logits

    total_lp = 0.0
    n = 0
    for i in range(cand_start, len(tokens)):
        if i == 0:
            continue
        row = np.array(logits[i - 1], dtype=np.float64)
        logprob = row[tokens[i]] - math.log(np.exp(row - row.max()).sum())
        total_lp += logprob
        n += 1
    return total_lp, (total_lp / n if n > 0 else -999), n

cases = [
    ("Poliziak atxilotu egin du", "mutila", ["musika", "mutiko"]),
    ("Kalean zebilen", "mutila", ["musika", "mutiko"]),
    ("Nire anaia", "mutila", ["musika", "mutiko"]),
    ("Goizean goiz esan dio", "kaixo", ["musika", "mutila"]),
    ("Familiaren", "etxea", ["musika", "mutila"]),
    ("Euskal", "herria", ["musika", "mutila"]),
    ("Ongi", "etorri", ["musika", "mutila"]),
]

for use_bos in [False, True]:
    label = "WITH BOS (wllama default)" if use_bos else "NO BOS (correct)"
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    correct = 0
    for context, right, wrongs in cases:
        cands = [right] + wrongs
        scored = {c: score(context, c, use_bos) for c in cands}
        winner = max(scored, key=lambda w: scored[w][1])  # per-token
        ok = "✓" if winner == right else "✗"
        if winner == right:
            correct += 1
        for c in cands:
            s, pt, n = scored[c]
            tag = " ←correct" if c == right else ""
            win = " ★WIN" if c == winner else ""
            print(f"  {ok if c==winner else ' '} {context:<28} {c:<10} sum={s:8.2f} ptok={pt:7.3f} ({n}tok){tag}{win}")
        print()
    print(f"  Result: {correct}/{len(cases)} correct")
