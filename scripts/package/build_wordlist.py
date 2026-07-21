"""
Build the Basque (eu) wordlist dictionary for FUTO Keyboard's AOSP dictionary engine.

FUTO runs two prediction engines in parallel: the transformer LM (this project's
GGUF) and a classical AOSP dictionary + bigram engine. The dictionary engine
ships wordlists under ``dictionaries/<lang>_wordlist.combined.gz`` (AOSP
"combined" text format). This script builds the Basque entry.

Pipeline
--------
  1. **Stream** the Latxa v2 corpus from HuggingFace — the same curated Basque
     corpus the model was pretrained on (config/eu.py) — into in-memory frequency
     maps. Inflected forms are captured directly from text (affix expansion is
     infeasible: the eu hunspell .aff has 121 620 SFX rules).
  2. **Two tracks**:
       - *Common words* (lowercased): filtered by length/min-count, validated by
         hunspell (eu_ES). This is the high-confidence core.
       - *Proper nouns* (capitalized/acronyms): hunspell rejects names, so a
         separate track captures them via a capitalization-ratio heuristic — a
         token is a proper noun if it is *usually* capitalized (ratio ≥ 0.5),
         which excludes sentence-initial common words ("Eta" at a sentence start
         is dwarfed by lowercase "eta"). This captures the place names, person
         names and acronyms (Bilbo, Euskal, AEB, Nafarroa …) that make up the
         valuable long tail — the field where a wordlist would otherwise lose to
         a proper-noun-heavy reference like Helium314's main_eu.
  3. **Bigrams**: top-N adjacent word pairs are emitted as AOSP combined bigrams
     (``  bigram=<w>,f=<f>`` under the unigram line). The dictionary engine uses
     these for contextual next-word ranking; many community dicts ship none, so
     even a capped set is a strict improvement.
  4. **Guarantee coverage**: must-include high-frequency words from
     ``config/eu.py`` (tokenizer content slots + autocorrect test targets) are
     injected with a frequency floor if the corpus missed them.
  5. **Map** counts → AOSP log-scale ``f ∈ [1,255]`` (255 = probability 1,
     ÷1.15 per level; 0 reserved for profanity → clamp to ≥1).
  6. **Emit** the ``.combined`` text format (header + unigrams + bigrams, sorted
     by f desc) and gzip → ``eu_wordlist.combined.gz``.

Deployment
----------
  * **Build-time (canonical)**: contribute ``eu_wordlist.combined.gz`` to FUTO's
    ``dictionaries/`` dir; the AOSP ``dicttool`` compiles it to a binary ``.dict``.
  * **Side-load today**: compile to a binary ``eu.dict`` (``compile_dict.sh``)
    and import via FUTO → Settings → Languages & Models → Import. The app
    detects the ``0x9bc13afe`` magic + ``locale`` header and registers it as the
    custom main dictionary for ``eu`` (``tryOpeningCustomMainDictionaryForLocale``).

Usage
-----
  uv run python -m scripts.package.build_wordlist
  uv run python -m scripts.package.build_wordlist --lines-per-source 0 --max-words 0   # full corpus, no cap
  uv run python -m scripts.package.build_wordlist --load-freq notes/wordfreq_latxa.json # re-emit without re-streaming
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Basque alphabet: Latin-26 + ñ + (rare) ü ç, plus internal apostrophe/hyphen.
# Excludes acute accents (á é í ó ú) which in standard batua appear only in
# names/dialects — keeping them out yields a cleaner wordlist; the keyboard's
# NFD layer maps them away at typing time anyway.
# TOKEN_RE is case-sensitive (one pass yields lower counts + proper-noun counts).
TOKEN_RE = re.compile(r"[A-ZÑÜÇa-zñüç][A-ZÑÜÇa-zñüç'\-]*")

# Corpus tokenization artifacts: a single letter glued to a hyphen/apostrophe
# ("A-Andoain", "A'B'", "A-Hunting", "a-marra") — these are list/initial
# markers or cross-lingual noise, never real Basque words. Filtered in both
# tracks. Multi-letter hyphenated forms ("Jean-Pierre", "autoeskola-ikasle")
# are kept.
ARTIFACT_RE = re.compile(r"^[A-ZÑÜÇa-zñüç]['\-]")

# AOSP "combined" frequency scale: 255 = probability 1, ÷1.15 per decrement.
# 0 is reserved for profanity, so the usable range is [1, 255].
F_MAX = 255
F_MIN = 1
_LOG_1_15 = math.log(1.15)

# wooorm/dictionaries Basque hunspell files (FLAG NUM format). Used only as a
# build-time validation cache; downloaded on first run.
HUNSPELL_DIC_URL = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/eu/index.dic"
HUNSPELL_AFF_URL = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/eu/index.aff"

# Defaults (overridable via CLI).
DEFAULT_SOURCES = ["wikipedia", "euscrawl-v2", "zelaihandi"]
DEFAULT_LINES_PER_SOURCE = 40_000      # 0 = stream the entire split
DEFAULT_MIN_COUNT = 3                   # common-word minimum corpus count
DEFAULT_MAX_WORDS = 60_000              # 0 = no cap (keep everything)
DEFAULT_MIN_LEN = 2
DEFAULT_MAX_LEN = 30
DEFAULT_PROPER_MIN_COUNT = 3            # proper-noun minimum corpus count
DEFAULT_PROPER_MIN_RATIO = 0.5          # capitalized / total occurrences
DEFAULT_MAX_BIGRAMS = 0                 # 0 = no bigrams (set >0 to enable)
FREQ_FLOOR = 120                        # f assigned to must-include words absent from corpus


# --------------------------------------------------------------------------- #
# Frequency → AOSP log-scale f
# --------------------------------------------------------------------------- #
def count_to_f(count: int, max_count: int) -> int:
    """Map a corpus count to AOSP combined-format frequency f ∈ [1, 255].

    255 = most frequent word (prob 1); each level down divides probability by
    1.15. 0 is reserved for profanity, so clamp lower bound to 1.
    """
    if count <= 0 or max_count <= 0:
        return F_MIN
    levels = math.log(max_count / count) / _LOG_1_15
    return max(F_MIN, min(F_MAX, round(F_MAX - levels)))


# --------------------------------------------------------------------------- #
# Corpus streaming → frequency maps (unigrams, proper nouns, bigrams)
# --------------------------------------------------------------------------- #
def stream_corpus(
    sources: list[str],
    lines_per_source: int,
    text_key: str,
    count_bigrams: bool,
) -> tuple[Counter, Counter, Counter]:
    """Stream Latxa v2 configs from HF.

    Returns (lower_counter, cap_counter, bigram_counter):
      - lower_counter: lowercased word → count (drives the common-word track)
      - cap_counter:   original-case capitalized token → count (proper-noun track)
      - bigram_counter: (w1, w2) lowercased adjacent pair → count (if enabled)
    """
    from datasets import load_dataset  # heavy import; keep local

    lower_counter: Counter[str] = Counter()
    cap_counter: Counter[str] = Counter()
    bigram_counter: Counter[tuple[str, str]] = Counter()

    for src in sources:
        n_lines = 0
        n_chars = 0
        t0 = time.time()
        print(f"  streaming HiTZ/latxa-corpus-v2 [{src}] …", flush=True)
        try:
            ds = load_dataset("HiTZ/latxa-corpus-v2", src, split="train", streaming=True)
        except Exception as e:  # config unavailable / network
            print(f"    ! skipping {src}: {e}", file=sys.stderr)
            continue
        for ex in ds:
            t = ex.get(text_key, "") or ""
            if not t:
                continue
            n_lines += 1
            n_chars += len(t)
            # Single pass: lowercase counts + proper-noun counts (+ bigrams).
            tokens: list[str] = []
            for m in TOKEN_RE.finditer(t):
                tok = m.group(0)
                low = tok.lower()
                lower_counter[low] += 1
                tokens.append(low)
                if tok[0].isupper():
                    cap_counter[tok] += 1
            if count_bigrams:
                for i in range(len(tokens) - 1):
                    bigram_counter[(tokens[i], tokens[i + 1])] += 1
            if lines_per_source and n_lines >= lines_per_source:
                break
        print(
            f"    {src}: {n_lines} lines, {n_chars/1e6:.1f} MB, "
            f"{len(lower_counter):,} unique words cumul, {time.time()-t0:.0f}s",
            flush=True,
        )
    return lower_counter, cap_counter, bigram_counter


# --------------------------------------------------------------------------- #
# Proper-noun selection
# --------------------------------------------------------------------------- #
def select_proper_nouns(
    cap_counter: Counter,
    lower_counter: Counter,
    min_count: int,
    min_ratio: float,
    min_len: int,
    max_len: int,
) -> tuple[dict[str, int], set[str]]:
    """Pick proper nouns from capitalized token counts.

    A lowercased form L is a proper noun if its total capitalized occurrences
    are ≥ ``min_count`` AND make up ≥ ``min_ratio`` of all occurrences of L
    (capitalized + lowercased). This excludes sentence-initial common words
    (e.g. "Eta" → ratio ≈ 0 vs lowercase "eta") while keeping names that are
    conventionally capitalized (Bilbo, Euskal, AEB, Nafarroa).

    Returns (proper_display, proper_lower):
      - proper_display: display form (dominant capitalization) → total cap count
      - proper_lower:   set of lowercased forms promoted to proper-noun track
                        (so they are excluded from the common-word track, to
                        avoid duplicating "euskal" + "Euskal")
    """
    variants: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for cap, c in cap_counter.items():
        if not (min_len <= len(cap) <= max_len):
            continue
        if ARTIFACT_RE.match(cap):
            continue
        variants[cap.lower()].append((cap, c))

    proper_display: dict[str, int] = {}
    proper_lower: set[str] = set()
    for low, varlist in variants.items():
        total_cap = sum(c for _, c in varlist)
        low_total = lower_counter.get(low, 0)  # includes capitalized occurrences
        ratio = total_cap / max(1, low_total)
        if total_cap >= min_count and ratio >= min_ratio:
            varlist.sort(key=lambda x: -x[1])
            display = varlist[0][0]
            proper_display[display] = total_cap
            proper_lower.add(low)
    return proper_display, proper_lower


# --------------------------------------------------------------------------- #
# Hunspell validation
# --------------------------------------------------------------------------- #
def ensure_hunspell(cache_dir: Path) -> Path:
    """Download the eu hunspell .dic/.aff into cache_dir if absent; return base path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dic = cache_dir / "eu.dic"
    aff = cache_dir / "eu.aff"
    if not dic.exists():
        print(f"  downloading hunspell eu.dic …", flush=True)
        urllib.request.urlretrieve(HUNSPELL_DIC_URL, dic)
    if not aff.exists():
        print(f"  downloading hunspell eu.aff …", flush=True)
        urllib.request.urlretrieve(HUNSPELL_AFF_URL, aff)
    return cache_dir / "eu"  # base path (hunspell -d <base>)


def hunspell_valid(words: list[str], base: Path) -> set[str]:
    """Return the subset of `words` that hunspell accepts as valid Basque.

    Uses `hunspell -l` (lists MISSPELLED words) in a single process: valid =
    all − misspelled. Faster than per-word pipe mode.
    """
    if not words:
        return set()
    proc = subprocess.run(
        ["hunspell", "-d", str(base), "-l"],
        input="\n".join(words),
        capture_output=True,
        text=True,
        timeout=600,
    )
    misspelled = {w.strip().lower() for w in proc.stdout.splitlines() if w.strip()}
    return {w for w in words if w not in misspelled}


# --------------------------------------------------------------------------- #
# Must-include safety net (from config/eu.py)
# --------------------------------------------------------------------------- #
def must_include_words() -> set[str]:
    """High-frequency Basque words that MUST appear in the dictionary.

    Drawn from config/eu.py: the tokenizer content-slot words (lemmas the model
    was trained to treat as units) + the autocorrect test targets. These are
    known-good by construction; added with a frequency floor if the corpus
    subset missed them.
    """
    from config.eu import SLOT_28_173, EU_ADJECTIVES, AUTOCORRECT_TESTS

    words = set(SLOT_28_173) | set(EU_ADJECTIVES)
    for _typo, correct in AUTOCORRECT_TESTS:
        words.add(correct.lower())
    return words


# --------------------------------------------------------------------------- #
# .combined emission
# --------------------------------------------------------------------------- #
def emit_combined(
    word_freqs: list[tuple[str, int]],
    bigrams_by_first: dict[str, list[tuple[str, int]]],
    out_path: Path,
    description: str,
    locale: str = "eu",
) -> None:
    """Write the AOSP combined-format wordlist, gzipped.

    Format (see FUTO dictionaries/sample.combined):
      - line 1: header CSV, first attr `dictionary=main:<locale>`
      - unigrams: ` word=<w>,f=<f>` (single leading space = indent level 1)
      - bigrams:  `  bigram=<w2>,f=<f>` under the w1 unigram (two leading spaces)
    Unigrams are sorted by f desc then word asc; bigrams under each by f desc.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    date = int(time.time())
    header = (
        f"dictionary=main:{locale},locale={locale},"
        f"description={description},date={date},version=1\n"
    )
    lines = [header]
    n_bigrams = 0
    for word, f in word_freqs:
        if not word or "," in word or f <= 0:
            continue
        lines.append(f" word={word},f={f}\n")
        for bw, bf in bigrams_by_first.get(word, []):
            if bw and "," not in bw and bf > 0:
                lines.append(f"  bigram={bw},f={bf}\n")
                n_bigrams += 1
    data = "".join(lines).encode("utf-8")
    with gzip.open(out_path, "wb") as gz:
        gz.write(data)
    print(
        f"  wrote {out_path} ({len(lines)-1-n_bigrams} unigrams, {n_bigrams} bigrams, "
        f"{len(data)/1024:.0f} KB raw, {os.path.getsize(out_path)/1024:.0f} KB gzipped)",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help=f"Comma-separated Latxa v2 configs (default: {','.join(DEFAULT_SOURCES)})")
    ap.add_argument("--lines-per-source", type=int, default=DEFAULT_LINES_PER_SOURCE,
                    help="Lines per source (0 = stream the entire split)")
    ap.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS,
                    help="Cap on unigrams (0 = no cap)")
    ap.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--proper-min-count", type=int, default=DEFAULT_PROPER_MIN_COUNT)
    ap.add_argument("--proper-min-ratio", type=float, default=DEFAULT_PROPER_MIN_RATIO)
    ap.add_argument("--max-bigrams", type=int, default=DEFAULT_MAX_BIGRAMS,
                    help="Top-N bigrams to emit (0 = disable bigrams)")
    ap.add_argument("--text-key", default="text", help="HF dataset text field")
    ap.add_argument("--hunspell-cache", default="dictionaries/.hunspell_eu",
                    help="Dir to cache the eu hunspell .dic/.aff (build-time only)")
    ap.add_argument("--no-hunspell", action="store_true",
                    help="Skip hunspell validation (keep all corpus words passing the other filters)")
    ap.add_argument("--save-freq", default=None,
                    help="Save the raw frequency maps (JSON) to this path for reuse")
    ap.add_argument("--load-freq", default=None,
                    help="Load saved frequency maps (JSON) instead of streaming HF")
    ap.add_argument("--out", default="dictionaries/eu_wordlist.combined.gz")
    ap.add_argument("--description", default="Basque wordlist (futo-basque)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    count_bigrams = args.max_bigrams > 0

    # 1. Frequency maps (stream or load)
    lower_counter: Counter[str] = Counter()
    cap_counter: Counter[str] = Counter()
    bigram_counter: Counter[tuple[str, str]] = Counter()

    if args.load_freq:
        print(f"Loading frequency maps from {args.load_freq}")
        blob = json.loads(Path(args.load_freq).read_text())
        if isinstance(blob, dict) and {"unigrams", "proper", "bigrams"} <= set(blob):
            lower_counter = Counter(blob["unigrams"])
            cap_counter = Counter(blob["proper"])
            bigram_counter = Counter(
                {tuple(k.split("\t")): v for k, v in blob["bigrams"].items()
                 if "\t" in k}
            )
        else:  # legacy flat format (unigrams only)
            lower_counter = Counter({k: v for k, v in blob.items()})
        print(f"  loaded {len(lower_counter):,} unigrams, {len(cap_counter):,} proper, "
              f"{len(bigram_counter):,} bigrams")
    else:
        print(f"Streaming Latxa v2 from HF ({args.lines_per_source or 'ALL'} lines/source)")
        lower_counter, cap_counter, bigram_counter = stream_corpus(
            sources, args.lines_per_source, args.text_key, count_bigrams
        )
    print(f"Raw unique lowercased words: {len(lower_counter):,}")
    if cap_counter:
        print(f"Raw capitalized tokens: {len(cap_counter):,}")
    if count_bigrams:
        print(f"Raw bigrams: {len(bigram_counter):,}")

    if args.save_freq:
        p = Path(args.save_freq)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "unigrams": dict(lower_counter),
            "proper": dict(cap_counter),
            "bigrams": {"\t".join(k): v for k, v in bigram_counter.items()},
        }, ensure_ascii=False))
        print(f"  saved frequency maps → {p}")

    # 2. Proper-noun track (decides which lowercased forms are promoted to
    #    capitalized proper nouns and thus excluded from the common-word track)
    proper_display, proper_lower = select_proper_nouns(
        cap_counter, lower_counter,
        args.proper_min_count, args.proper_min_ratio, args.min_len, args.max_len,
    )
    print(f"Proper nouns: {len(proper_display):,} "
          f"(excluded {len(proper_lower):,} lowercased forms from common track)")

    # 3. Common-word track: static filters + hunspell, minus proper-noun forms
    def passes(w: str) -> bool:
        return (
            args.min_len <= len(w) <= args.max_len
            and lower_counter[w] >= args.min_count
            and w not in proper_lower
            and not ARTIFACT_RE.match(w)
        )

    candidates = [w for w in lower_counter if passes(w)]
    print(f"After length+min_count+proper filters: {len(candidates):,} common-word candidates")

    if args.no_hunspell:
        print("hunspell validation SKIPPED (--no-hunspell)")
        valid = set(candidates)
    else:
        base = ensure_hunspell(Path(args.hunspell_cache))
        print(f"Validating {len(candidates):,} words with hunspell eu …")
        valid = hunspell_valid(candidates, base)
        print(f"  hunspell accepted: {len(valid):,} "
              f"({100*len(valid)/max(1,len(candidates)):.1f}% of candidates)")

    # 4. Must-include safety net
    must = must_include_words()
    missing_must = {
        w for w in must if w not in valid and w not in proper_display
        and args.min_len <= len(w) <= args.max_len
    }
    if missing_must:
        print(f"Injecting {len(missing_must)} must-include words missing from corpus "
              f"(f={FREQ_FLOOR} floor)")
        for w in missing_must:
            lower_counter[w] = max(lower_counter.get(w, 0), 1)

    # 5. Build (word, count) source: common (lower) + proper (cap) + must-include
    freq_source: dict[str, int] = {}
    for w in valid:
        freq_source[w] = lower_counter[w]
    for display, c in proper_display.items():
        # don't let a proper noun shadow a common word of the same display form
        if display not in freq_source:
            freq_source[display] = c
        else:
            freq_source[display] = max(freq_source[display], c)
    for w in missing_must:
        freq_source[w] = max(freq_source.get(w, 0), lower_counter.get(w, 1))

    max_count = max(freq_source.values()) if freq_source else 1
    word_freqs = sorted(
        ((w, count_to_f(c, max_count)) for w, c in freq_source.items()),
        key=lambda wf: (-wf[1], wf[0]),
    )
    if args.max_words and len(word_freqs) > args.max_words:
        word_freqs = word_freqs[: args.max_words]
    print(f"Final wordlist: {len(word_freqs):,} unigrams "
          f"(f range {word_freqs[-1][1]}–{word_freqs[0][1]}, max_count={max_count:,})")
    print("Top-10:", [(w, f) for w, f in word_freqs[:10]])

    # 5b. Bigrams: keep top-N where both words are in the final unigram set
    bigrams_by_first: dict[str, list[tuple[str, int]]] = defaultdict(list)
    if count_bigrams and bigram_counter:
        final_words = {w for w, _ in word_freqs}
        kept = [
            (w1, w2, c) for (w1, w2), c in bigram_counter.items()
            if w1 in final_words and w2 in final_words
        ]
        kept.sort(key=lambda x: -x[2])
        kept = kept[: args.max_bigrams]
        max_big = kept[0][2] if kept else 1
        for w1, w2, c in kept:
            bigrams_by_first[w1].append((w2, count_to_f(c, max_big)))
        print(f"Bigrams: {len(kept):,} emitted (of {len(bigram_counter):,} raw, "
              f"cap {args.max_bigrams})")
        if kept:
            print("  top-10 bigrams:", [(w1, w2, c) for w1, w2, c in kept[:10]])

    # 6. Emit .combined.gz
    emit_combined(word_freqs, bigrams_by_first, Path(args.out), args.description)
    print("\nDone. Next: compile to a binary .dict with the AOSP dicttool for "
          "side-loading (scripts/package/compile_dict.sh), or contribute the "
          ".combined.gz to FUTO's dictionaries/.")


if __name__ == "__main__":
    main()
