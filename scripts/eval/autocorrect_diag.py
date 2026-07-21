"""
Diagnostic: run AUTOCORRECT_TESTS against the shipped GGUF via llama.cpp,
mirroring keyboard.py::evaluate_autocorrect but with FULL token-level visibility.

Feeds RAW TOKEN IDS (not string tokenization) — exactly what keyboard.py does.
Resets KV cache between cases. Also runs a next-word sanity probe to validate
the llama.cpp setup independently (model supposedly scores ~50% top-1 there).
"""
from __future__ import annotations
import argparse
import unicodedata
import numpy as np
from llama_cpp import Llama
from config.eu import AUTOCORRECT_TESTS, NEXT_WORD_TESTS


def to_keypress_chars(typed: str) -> list[str]:
    out = []
    for ch in typed:
        base = "".join(c for c in unicodedata.normalize("NFD", ch)
                       if not unicodedata.combining(c))
        for c in base.upper():
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out


def char_id(c: str) -> int:
    """<CHAR_A>=182 .. <CHAR_Z>=207 (verified from GGUF)."""
    return 182 + (ord(c.upper()) - ord("A"))


def typo_to_char_ids(typed: str) -> list[int]:
    """Mirror to_keypress_chars but return token IDs (NOT a joined string).
    Strips diacritics (ñ→N) + case via NFD, one <CHAR_X> id per keystroke."""
    ids = []
    for ch in typed:
        base = "".join(c for c in unicodedata.normalize("NFD", ch)
                       if not unicodedata.combining(c))
        for c in base.upper():
            if "A" <= c <= "Z":
                ids.append(char_id(c))
    return ids


def greedy_decode(llm, prompt_ids, max_tokens, stop_id):
    llm.reset()                          # clear KV cache — critical between cases
    llm.eval(prompt_ids)
    out = []
    for _ in range(max_tokens):
        last = llm.eval_logits[-1]
        nxt = int(np.argmax(last))
        out.append(nxt)
        if nxt == stop_id:
            break
        llm.eval([nxt])
    return out


def run_autocorrect(llm, bos, max_tokens):
    XBU, XBC, XEC = 174, 175, 176
    top1 = 0
    n = len(AUTOCORRECT_TESTS)
    print(f"\n=== AUTOCORRECT (bos={bos}) ===")
    print(f"{'typo':<11}{'want':<12}{'got':<24}{'token ids'}")
    print("-" * 95)
    for typo, correct in AUTOCORRECT_TESTS:
        # build prompt as RAW TOKEN IDS — exactly what keyboard.py does
        # (sp.piece_to_id per <CHAR_X> piece; NEVER string-tokenize)
        prompt_ids = ([1] if bos else []) + [XBU] + \
                     typo_to_char_ids(typo) + [XBC]
        gen = greedy_decode(llm, prompt_ids, max_tokens, XEC)
        gen_clean = [t for t in gen if t != XEC]
        text = llm.detokenize(gen_clean).decode("utf-8", errors="replace").strip()
        hit = text.lower() == correct.lower()
        top1 += hit
        flag = "✓" if hit else "✗"
        print(f"{flag} {typo:<10}{correct:<12}{text!r:<24}{gen_clean[:12]}")
    print("-" * 95)
    print(f"top-1: {top1}/{n} = {100*top1/n:.1f}%")
    return top1, n


def run_nextword(llm, bos, max_tokens=6):
    """Sanity probe: feed a Basque sentence prefix, print the greedy continuation."""
    print(f"\n=== NEXT-WORD sanity probe (bos={bos}) ===")
    for prefix, plausible in NEXT_WORD_TESTS[:6]:
        # tokenize the prefix string; special tokens not needed here
        ids = llm.tokenize(prefix.encode("utf-8"), add_bos=bos, special=False)
        gen = greedy_decode(llm, ids, max_tokens, stop_id=-1)
        text = llm.detokenize(gen).decode("utf-8", errors="replace").strip()
        plaus = ", ".join(plausible)
        print(f"  {prefix!r:<28} → {text!r:<22} (plausible: {plaus})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="gguf/eu_futo_v2.gguf")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--bos", action="store_true", default=True,
                    help="prepend BOS (id=1). DEFAULT ON now.")
    ap.add_argument("--no-bos", dest="bos", action="store_false")
    ap.add_argument("--nextword-only", action="store_true")
    args = ap.parse_args()

    path = "gguf/eu_futo_mini_v2.gguf" if args.mini else args.gguf
    print(f"Loading {path}  (bos={args.bos}) ...")
    llm = Llama(model_path=path, n_ctx=1024, n_ubatch=512, n_threads=8,
                verbose=False, logits_all=True)

    if not args.nextword_only:
        run_autocorrect(llm, args.bos, args.max_tokens)
    run_nextword(llm, args.bos)


if __name__ == "__main__":
    main()
