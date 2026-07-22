#!/usr/bin/env python3
"""
Ikasbil fill-the-gap test for LM surprisal re-ranking.

Uses the same 8 test cases as lm-test.html (from Ikasbil exercise
"Euskaldun berriari akatsak zuzentzen") to validate surprisal scoring
in Python (llama-cpp-python) as a reference for the wllama JS implementation.

Scoring method: surprisal_sum = log P(candidate|context) - log P(candidate)
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

def score_surprisal(context, candidate, use_bos=False):
    """Return surprisal_sum = log P(candidate|context) - log P(candidate)."""
    # Baseline: " candidate" (leading space)
    base_tokens = llm.tokenize(f" {candidate}".encode("utf-8"), add_bos=use_bos)
    base_start = 1 + (1 if use_bos else 0)
    base_lps = token_logprobs(base_tokens, base_start)

    # In-context: find candidate token boundary
    ctx_tokens = llm.tokenize(context.encode("utf-8"), add_bos=use_bos)
    full_tokens = llm.tokenize(f"{context} {candidate}".encode("utf-8"), add_bos=use_bos)
    cpl = common_prefix_len(ctx_tokens, full_tokens)
    cand_start = max(cpl, 1)
    ctx_lps = token_logprobs(full_tokens, cand_start)

    ctx_sum = sum(ctx_lps)
    base_sum = sum(base_lps)
    return ctx_sum - base_sum


# Ikasbil exercise: "Gertatutakoa kontatzen" (fill-the-gap, 3 options each)
# Source: https://www.ikasbil.eus/documents/20928/15491438/1534219.pdf
cases = [
    ("Ezetz asmatu gaur", ["zer gertatu den", "zer gertatu da", "zer gertatu dion"], "zer gertatu den"),
    ("Badakizu, batzuetan zalantzak izaten ditu", ["neska horrek", "neska hori", "neska horri"], "neska horrek"),
    ("Mikel zuzentzen hasi zaio, eta", ["neskak aurpegi arraro oso", "neskak oso aurpegi arraroa", "neska aurpegi oso arraro"], "neskak oso aurpegi arraroa"),
    ("berak lasai-lasai esan dio", ["eskerrak ematen dizkion", "eskerrak ematen dizkiola", "eskerrak eman dizkio"], "eskerrak ematen dizkiola"),
    ("Miren asko haserretu da eta eskatu dio jendearen aurrean", ["ez zuzentzea", "ez zuzentzeko", "ez diola zuzendu"], "ez zuzentzeko"),
    ("Nire ustez,", ["Mikelek arrazoi duela", "Mikelek arrazoi du", "Mikel arrazoi duela"], "Mikelek arrazoi du"),
    ("bere neska-laguna", ["asko saiatzen da", "asko saiatzen du", "asko ahalegintzen du"], "asko saiatzen da"),
    ("abisatu dio Aneri hurrengoan", ["ez diola ezer zuzenduko", "ez omen diola ezer zuzenduko", "ez dio ezer zuzenduko"], "ez diola ezer zuzenduko"),
]

print(f"\n{'#'*72}\n#  Ikasbil fill-the-gap test (no BOS)\n{'#'*72}")
correct = 0
for context, options, right in cases:
    scored = {c: score_surprisal(context, c) for c in options}
    winner = max(scored, key=lambda w: scored[w])
    ok = winner == right
    if ok:
        correct += 1
    mark = "✓" if ok else "✗"
    print(f"  {mark} {context[:40]:<42} → {winner[:25]:<27} (want {right[:25]})")
    print(f"    " + "  ".join(f"{c[:20]}={scored[c]:+.3f}" for c in options))

print(f"\nResult: {correct}/{len(cases)} correct")
