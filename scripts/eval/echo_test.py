"""Validate echo-based fast path against the token-by-token reference.

Uses the EXACT same 10 cases and scoring logic as surprisal_test.py, but
replaces the iterative token_logprobs() with a single echo completion call.

If results match 9/10, the wllama implementation can use 1 call/candidate.
"""
import llama_cpp
import math

GGUF = 'gguf/eu_futo_v2_nobos.gguf'
llm = llama_cpp.Llama(model_path=GGUF, n_ctx=2048, n_gpu_layers=0,
                      verbose=False, logits_all=True)

def echo_logprobs(text):
    """Single echo call. Returns (tokens, logprobs) for the prompt portion."""
    r = llm.create_completion(prompt=text, max_tokens=1, logprobs=0,
                              echo=True, temperature=0)
    lp = r['choices'][0]['logprobs']
    return lp['tokens'], lp['token_logprobs']

def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n

def score_word(context, word):
    """Return (ctx_sum, ctx_n, base_sum, base_n) via echo fast path."""
    # Baseline: " word" (leading space matches in-context tokenization)
    base_tokens = llm.tokenize(f" {word}".encode("utf-8"), add_bos=False)
    _, base_lps = echo_logprobs(f" {word}")
    # base_start = 1 (skip first token's logprob which is None)
    base_start = 1
    base_vals = [x for x in base_lps[base_start:] if x is not None and math.isfinite(x)]

    # In-context
    ctx_tokens = llm.tokenize(context.encode("utf-8"), add_bos=False)
    full_tokens = llm.tokenize(f"{context} {word}".encode("utf-8"), add_bos=False)
    cpl = common_prefix_len(ctx_tokens, full_tokens)
    cand_start = max(cpl, 1)
    _, full_lps = echo_logprobs(f"{context} {word}")
    ctx_vals = [x for x in full_lps[cand_start:] if x is not None and math.isfinite(x)]

    return (sum(ctx_vals), len(ctx_vals), sum(base_vals), len(base_vals))

# EXACT same cases as surprisal_test.py
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

print("=== surprisal_sum via echo (fast path, no BOS) ===")
correct = 0
for context, right, wrongs in cases:
    scored = {}
    for c in [right] + wrongs:
        cs, cn, bs, bn = score_word(context, c)
        scored[c] = cs - bs  # surprisal_sum
    winner = max(scored, key=lambda w: scored[w])
    ok = winner == right
    correct += ok
    mark = "✓" if ok else "✗"
    print(f"  {mark} {context[:24]:<26} → {winner:<8} (want {right})  " +
          "  ".join(f"{c}={scored[c]:+.3f}" for c in [right]+wrongs))
print(f"\n  Result: {correct}/{len(cases)} correct")
print(f"  Reference (token-by-token, no BOS): 9/10")
