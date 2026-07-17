#!/usr/bin/env python3
"""
Keystrokes-saved eval: simulate typing realistic messaging messages and measure
how many keystrokes the model saves via next-word prediction.

Methodology:
  - Take a realistic Basque chat message (WhatsApp/Telegram style)
  - Simulate typing it word by word
  - After each completed word + space, get the model's top-1 and top-3 next-word
    suggestions (greedy + top-K first-token logprobs)
  - If the intended next word is suggested, count len(word) as keystrokes saved
  - Report per-message and aggregate savings ratio

This measures the ACTUAL user experience: "how much typing does this model save?"
instead of the artificial "does greedy top-1 match our hand-picked gold answer?"

Usage:
  python keystrokes.py --gguf <model.gguf> --tokenizer <spm.model> [--add-bos] [--name <label>]
"""
import argparse, json, time, re
import sentencepiece as spm
import llama_cpp

# ── Realistic Basque messaging messages ─────────────────────────────────────
# These are natural one-person messages as typed in WhatsApp/Telegram/social
# media. Not textbook sentences — real chat register. Each is something a person
# would actually send. Varying length, register, and topic.
MESSAGES = [
    # Casual planning / coordination
    "Bihar goizean zurekin egongo naiz ez dut ezer programaturik",
    "Gero ikusiko gara ez ahaztu bilera sei eta erdietan",
    "Gaur ezin naiz etorriko dut beste plan bat",
    "Zer egingo dugu gaur arratsaldean eguraldia ona dago eta",
    "Atera behar dut txakurra paseatzera gero itzuliko naiz",

    # Gratitude / social
    "Eskerrik asko denagatik oso ondo pasa nuen",
    "Barkatu atzerapena erantzuteko lanpetuta egon naiz egun guztian",
    "Zorionak urtebetetzeagatik zoriontsu izan zaitez",

    # Learning / understanding
    "Ez dut asko ulertu berriro azaldu diezaiokezu",
    "Euskara ikasten ari naiz baina oraindik zaila dut",
    "Ez dakit noiz bukatuko dut lan hau asko geratzen da",

    # Status / how are you
    "Zer moduz zaude dena ondo doa espero dut",
    "Nola dago zure ama espero dut ondo egotea",

    # Location / directions
    "Non dago geltoki gertuena ez dut hemen ondo ezagutzen",

    # Uncertainty / tentative
    "Ez nago ziur agian etorriko naiz gutxienez saiatuko naiz",

    # Weekend / activities
    "Asteburua mendian eman nuen eta oso ederra zegoen",
    "Zein filma ikusi dugu atzo oso luzea zen",

    # Affection
    "Maite zaitut ez daukat hitzik esateko",

    # Home / hosting
    "Ongi etorri etxera afaria prest duzu",

    # Unexpected events
    "Telebista ikusten nengoen eta bat-batean eten zen",
]


class KeystrokeModel:
    def __init__(self, gguf_path, sp_path, name, add_bos=False, bos_id=1):
        self.name = name
        self.add_bos = add_bos
        self.bos_id = bos_id
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_path)
        self.llm = llama_cpp.Llama(
            model_path=gguf_path, n_ctx=512, n_threads=4,
            verbose=False, logits_all=True,
        )
        self.vocab_size = self.sp.GetPieceSize()

    def _encode(self, text):
        ids = self.sp.Encode(text)
        if self.add_bos:
            ids = [self.bos_id] + ids
        return ids

    def predict_next(self, prefix):
        """Given typed text so far (ending with space), return top-1 word + top-K tokens."""
        ids = self._encode(prefix)
        r = self.llm.create_completion(
            ids, max_tokens=12, temperature=0, top_p=1.0, logprobs=5,
        )
        choice = r["choices"][0]
        text = choice["text"]
        top1_word = self._extract_word(text)

        # Top-K first-token candidates
        lp = choice.get("logprobs", {})
        top_lp = lp.get("top_logprobs", [None])
        topk_tokens = list(top_lp[0].keys()) if top_lp and top_lp[0] else []

        return top1_word, topk_tokens, text

    @staticmethod
    def _extract_word(text):
        """Extract the first word from generated text."""
        text = text.strip()
        if not text:
            return ""
        # Take everything up to the first space or punctuation
        match = re.match(r'^[^\s.,!?;:"\'()«»\[\]{}]+', text)
        return match.group(0) if match else ""

    @staticmethod
    def _normalize(word):
        """Normalize a word for comparison: strip SP prefix, lowercase."""
        w = word.replace("▁", "").strip().lower()
        w = w.strip(".,!?;:\"'()«»[]{}")
        return w

    def check_top1(self, predicted, target):
        p = self._normalize(predicted)
        t = self._normalize(target)
        if not p or not t:
            return False
        return p == t

    def check_topk(self, topk_tokens, target):
        """Check if target word appears in top-K first tokens (or is a prefix match)."""
        t = self._normalize(target)
        if not t or len(t) < 2:
            return False
        for tok in topk_tokens:
            w = self._normalize(tok)
            if not w or w.startswith("<") or len(w) < 1:
                continue
            # Exact match on first token
            if w == t:
                return True
            # Prefix match: first token is start of target word
            if len(w) >= 2 and t.startswith(w):
                return True
        return False

    def close(self):
        del self.llm


def eval_message(model, message):
    """Simulate typing a message, return per-word results + keystroke stats."""
    words = message.split()
    if len(words) < 2:
        return None

    # Total characters the user would type (excluding spaces)
    total_chars = sum(len(w) for w in words)
    # Characters of words that CAN be predicted (all except the first word)
    predictable_chars = sum(len(w) for w in words[1:])

    results = []
    top1_saved = 0
    topk_saved = 0

    for i in range(len(words) - 1):
        # What's been typed so far (including trailing space)
        prefix = " ".join(words[:i+1]) + " "
        target = words[i+1]

        try:
            top1_word, topk_tokens, raw = model.predict_next(prefix)
        except Exception as e:
            top1_word, topk_tokens, raw = "", [], ""

        hit1 = model.check_top1(top1_word, target)
        hitk = model.check_topk(topk_tokens, target)

        if hit1:
            top1_saved += len(target)
        if hitk:
            topk_saved += len(target)

        results.append({
            "position": i+1,
            "prefix": prefix.strip()[-30:] + " ▎",
            "target": target,
            "predicted": top1_word,
            "top1_hit": hit1,
            "topk_hit": hitk,
            "chars_saved_top1": len(target) if hit1 else 0,
            "raw": raw[:40],
        })

    return {
        "message": message,
        "n_words": len(words),
        "total_chars": total_chars,
        "predictable_chars": predictable_chars,
        "top1_saved": top1_saved,
        "topk_saved": topk_saved,
        "top1_pct": top1_saved / predictable_chars * 100 if predictable_chars else 0,
        "topk_pct": topk_saved / predictable_chars * 100 if predictable_chars else 0,
        "words": results,
    }


def main():
    ap = argparse.ArgumentParser(description="Keystrokes-saved eval")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--add-bos", action="store_true")
    ap.add_argument("--bos-id", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"{'='*70}")
    print(f"KEYSTROKES-SAVED EVAL: {args.name}")
    print(f"  Model: {args.gguf}")
    print(f"  Messages: {len(MESSAGES)}")
    print(f"  BOS: {'yes (id=%d)' % args.bos_id if args.add_bos else 'no'}")
    print(f"{'='*70}")

    model = KeystrokeModel(args.gguf, args.tokenizer, args.name,
                           add_bos=args.add_bos, bos_id=args.bos_id)

    all_results = []
    total_top1_saved = 0
    total_topk_saved = 0
    total_predictable = 0
    total_chars = 0

    for msg in MESSAGES:
        r = eval_message(model, msg)
        if r is None:
            continue
        all_results.append(r)
        total_top1_saved += r["top1_saved"]
        total_topk_saved += r["topk_saved"]
        total_predictable += r["predictable_chars"]
        total_chars += r["total_chars"]

        # Per-message summary
        print(f"\n┌─ {msg}")
        for w in r["words"]:
            t1 = "✓" if w["top1_hit"] else "✗"
            tk = "✓" if w["topk_hit"] else " "
            saved = f"+{w['chars_saved_top1']}" if w["top1_hit"] else "  "
            print(f"│ {t1}{tk} {w['prefix']:34s} → {w['target']:14s} "
                  f"(got: {w['predicted']:14s}) {saved}")
        print(f"└─ saved: top1={r['top1_saved']}/{r['predictable_chars']} "
              f"({r['top1_pct']:.0f}%)  top5={r['topk_saved']}/{r['predictable_chars']} "
              f"({r['topk_pct']:.0f}%)")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATE: {args.name}")
    print(f"{'='*70}")
    print(f"  Messages:               {len(all_results)}")
    print(f"  Total characters:       {total_chars}")
    print(f"  Predictable characters: {total_predictable}")
    print(f"  Top-1 keystrokes saved: {total_top1_saved} / {total_predictable} "
          f"= {total_top1_saved/total_predictable*100:.1f}%")
    print(f"  Top-5 keystrokes saved: {total_topk_saved} / {total_predictable} "
          f"= {total_topk_saved/total_predictable*100:.1f}%")
    print(f"  Top-1 words correct:    "
          f"{sum(1 for r in all_results for w in r['words'] if w['top1_hit'])}/"
          f"{sum(1 for r in all_results for w in r['words'])}")
    print(f"  Top-5 words correct:    "
          f"{sum(1 for r in all_results for w in r['words'] if w['topk_hit'])}/"
          f"{sum(1 for r in all_results for w in r['words'])}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "model": args.name,
                "aggregate": {
                    "messages": len(all_results),
                    "total_chars": total_chars,
                    "predictable_chars": total_predictable,
                    "top1_saved": total_top1_saved,
                    "topk_saved": total_topk_saved,
                    "top1_pct": total_top1_saved / total_predictable * 100,
                    "topk_pct": total_topk_saved / total_predictable * 100,
                },
                "messages": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to {args.out}")

    model.close()


if __name__ == "__main__":
    main()
