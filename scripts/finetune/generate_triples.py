"""
Phase 4a data prep: generate typo→correct JSON files for isolated.py.

Produces two JSON files (arrays of {"typed": ..., "committed": ...} objects):
  • synth.json — synthetic typos generated from the word-frequency list via
    typo_synthesis.synth_typo(). Each frequent word gets multiple typo variants,
    weighted by log(freq+1) so common words appear more often.
  • real.json  — "real" user corrections from config.eu.EU_SHORTCUTS (known
    shortcut→full-form pairs). These are genuine typing patterns, not synthetic.

isolated.py mixes these at real_mix_ratio (default 0.25 = 25% real, 75% synth).

Usage:
  uv run python -m scripts.finetune.generate_triples \\
      --wordfreq notes/wordfreq.json \\
      --out-synth notes/synth.json \\
      --out-real notes/real.json \\
      --n-synth 200000
"""
from __future__ import annotations
import argparse
import bisect
import json
import math
import random
from itertools import accumulate
from pathlib import Path

from config.eu import EU_SHORTCUTS
from scripts.lib.typo_synthesis import synth_typo
from scripts.lib.runconfig import load_config, pick


def generate_synth(wordfreq: dict, n: int, rng: random.Random) -> list[dict]:
    """Generate n synthetic typo→correct pairs, frequency-weighted.

    Pre-computes cumulative weights ONCE so each word pick is O(log n) via
    bisect, not O(n) per call (the random.choices() default recomputes the
    full cumulative distribution every call — 200K elements × 2.5M attempts
    = hours). This makes 500K pairs take seconds instead of hours.
    """
    words = list(wordfreq.keys())
    freqs = [int(wordfreq[w]) for w in words]
    # log(freq+1) dampener per FUTO wiki — common words appear more but not
    # proportionally (avoids flooding with "eta", "da", etc.)
    weights = [math.log(f + 1) for f in freqs]
    # Pre-compute cumulative weights once — bisect is O(log n) per pick
    cum_weights = list(accumulate(weights))
    total = cum_weights[-1]

    pairs: list[dict] = []
    attempts = 0
    max_attempts = n * 5  # some words yield no typo (too short, non-alpha)
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        # O(log n) weighted pick via bisect on pre-computed cum_weights
        idx = bisect.bisect_right(cum_weights, rng.random() * total, 0, len(words))
        word = words[min(idx, len(words) - 1)]
        typo = synth_typo(word, rng)
        if typo is None or typo == word:
            continue
        pairs.append({"typed": typo, "committed": word})
    return pairs


def generate_real() -> list[dict]:
    """Generate real typo→correct pairs from EU_SHORTCUTS."""
    pairs: list[dict] = []
    for correct, typos in EU_SHORTCUTS.items():
        for typo in typos:
            if typo and correct and typo != correct:
                pairs.append({"typed": typo, "committed": correct})
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase4a_dataprep.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--wordfreq", default=None, help="wordfreq.json from build_wordfreq.py")
    ap.add_argument("--out-synth", default=None, help="Output synth.json path")
    ap.add_argument("--out-real", default=None, help="Output real.json path")
    ap.add_argument("--n-synth", type=int, default=None, help="Number of synth pairs to generate")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)
    wordfreq_path = pick(args.wordfreq, cfg, "wordfreq", None)
    out_synth = pick(args.out_synth, cfg, "out_synth", None)
    out_real = pick(args.out_real, cfg, "out_real", None)
    n_synth = pick(args.n_synth, cfg, "n_synth", 200_000)

    if not wordfreq_path or not out_synth or not out_real:
        ap.error("--wordfreq, --out-synth, and --out-real are required "
                 "(or provide --config configs/phase4a_dataprep.yaml)")

    rng = random.Random(args.seed)

    # Load word frequencies
    wordfreq = json.loads(Path(wordfreq_path).read_text())
    print(f"Loaded {len(wordfreq)} words from {wordfreq_path}")

    # Generate synth pairs
    print(f"Generating {n_synth} synthetic typo→correct pairs...")
    synth = generate_synth(wordfreq, n_synth, rng)
    Path(out_synth).write_text(json.dumps(synth, ensure_ascii=False))
    print(f"  ✓ wrote {len(synth)} synth pairs → {out_synth}")

    # Generate real pairs
    real = generate_real()
    Path(out_real).write_text(json.dumps(real, ensure_ascii=False))
    print(f"  ✓ wrote {len(real)} real pairs → {out_real}")

    # Spot-check
    print("\n=== synth spot-check ===")
    for p in rng.sample(synth, min(8, len(synth))):
        print(f"  {p['typed']!r:20s} → {p['committed']!r}")
    print("\n=== real spot-check ===")
    for p in real[:8]:
        print(f"  {p['typed']!r:20s} → {p['committed']!r}")


if __name__ == "__main__":
    main()
