"""
Phase 1b: clean + stage the BERnaT BSM conversational tier.

BERnaT BSMauthor (Basque Social Media) is re-included despite morpheus's
a-priori exclusion, because the BERnaT paper (Azurmendi et al. 2025,
arXiv:2512.03903) shows diverse data helps without hurting standard-form
accuracy, and FUTO is a phone keyboard (deployment = chat, not Wikipedia).

But social-media text is noisy. This script applies aggressive cleaning:

  STRIP (structural noise, in-place):
    - emoji & pictographs (Unicode ranges + the `emoji` library if installed)
    - URLs (http/https/www)
    - @mentions
    - hashtag symbol (keep the word, drop '#')
    - RT / via @user attribution prefixes

  FILTER (line-level — drop the whole line):
    - code-switched lines (Spanish function-word ratio > BERNAT_MAX_SPANISH_RATIO)
    - too-short lines (< BERNAT_MIN_WORDS after stripping)
    - pure-punctuation / pure-emoji lines after stripping
    - exact duplicate lines (per-source hash set)

  KEEP (this is the value — do NOT "clean" these away):
    - dialectal Basque spelling (real usage; BERnaT says it helps)
    - informal grammar / word order
    - slang / colloquial vocabulary

Output goes to corpora/conversational/shard_*.txt — a SEPARATE directory from
the clean tier so the tokenizer (which globs corpora/clean/) never sees it.
The pretrain phase reads both dirs (pass --corpus corpora/clean corpora/conversational).

Two modes:
  1. Local (default): read morpheus's lightly-cleaned BSM file.
       uv run python -m scripts.corpus.clean_bernat \
           --morpheus-dir ../morpheus-mamba --out corpora/conversational
  2. HF fallback (--from-hf): stream HiTZ/BERnaT-Diverse BSMauthor from HF.
       uv run python -m scripts.corpus.clean_bernat --from-hf --out corpora/conversational
"""
from __future__ import annotations
import argparse
import hashlib
import re
import sys
from pathlib import Path

from config.eu import (
    BERNAT_CONFIG,
    BERNAT_LOCAL_FILE,
    BERNAT_MAX_SPANISH_RATIO,
    BERNAT_MIN_WORDS,
    BERNAT_REPO,
    BERNAT_TARGET_TOKENS,
    BERNAT_TEXT_KEY,
    CHARS_PER_TOKEN,
    is_strictly_eu,
)
from scripts.lib.runconfig import load_config, pick

# --------------------------------------------------------------------------- #
# Noise-stripping regexes
# --------------------------------------------------------------------------- #
# Emoji & pictograph Unicode ranges (covers most emoji without needing the
# `emoji` package). Ranges from the Unicode standard's emoji data.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes ext
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs ext-A
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # miscellaneous symbols (☀ ☂ ☎ etc.)
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F1E6-\U0001F1FF"  # regional indicator pairs (flags)
    "\U00002B00-\U00002BFF"  # misc symbols & arrows
    "]+",
    flags=re.UNICODE,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# RT/via prefix: consumes the whole "RT @user: " / "via @user" unit at once.
_RT_RE = re.compile(r"(?:\bRT\b|\bvia\b)\s*(?:@\w+\s*)?:?\s*", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")          # keep the word, drop the #
# HTML entities commonly found in scraped social media (&amp; &lt; &gt; &quot; &#39; &nbsp;)
_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|#39|#x27|nbsp);")
_WS_RE = re.compile(r"\s+")


def strip_noise(text: str) -> str:
    """Remove emoji, URLs, mentions, hashtag symbols, RT prefixes, HTML entities.
    Returns cleaned text with collapsed whitespace (may be empty if nothing
    survived)."""
    text = _URL_RE.sub(" ", text)
    text = _RT_RE.sub(" ", text)            # strips "RT @user: " as a unit
    text = _MENTION_RE.sub(" ", text)       # any remaining standalone @mentions
    text = _HASHTAG_RE.sub(r"\1", text)  # #kaixo -> kaixo
    text = _EMOJI_RE.sub(" ", text)
    text = _ENTITY_RE.sub(" ", text)        # &amp; -> space (avoid false words)
    text = _WS_RE.sub(" ", text).strip()
    return text


# Lines that are just punctuation / symbols / whitespace after stripping.
_JUNK_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)


def is_keepable(text: str) -> bool:
    """Line-level gate after stripping: enough words + actually Basque."""
    if not text or _JUNK_RE.match(text):
        return False
    words = text.split()
    if len(words) < BERNAT_MIN_WORDS:
        return False
    if not is_strictly_eu(text, max_es_ratio=BERNAT_MAX_SPANISH_RATIO):
        return False
    return True


# --------------------------------------------------------------------------- #
# Shard writer (same convention as build_corpus.py)
# --------------------------------------------------------------------------- #

def shard_writer(out_dir: Path, shard_target_bytes: int = 256 * 1024 * 1024):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    f = None
    bytes_written = 0

    def open_shard():
        nonlocal f, bytes_written, shard_idx
        path = out_dir / f"shard_{shard_idx:05d}.txt"
        f = open(path, "w", encoding="utf-8")
        bytes_written = 0
        return path

    def write(text: str):
        nonlocal f, bytes_written, shard_idx
        if f is None or bytes_written >= shard_target_bytes:
            if f is not None:
                f.close()
                shard_idx += 1
            open_shard()
        line = text + "\n"
        f.write(line)
        bytes_written += len(line.encode("utf-8"))

    def close():
        if f is not None:
            f.close()

    return write, close


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _process_line(raw: str) -> str | None:
    """Clean one raw line. Returns the cleaned text, or None if dropped."""
    text = strip_noise(raw.strip())
    if not is_keepable(text):
        return None
    return text


# --------------------------------------------------------------------------- #
# Mode 1: local morpheus file
# --------------------------------------------------------------------------- #

def clean_local(morpheus_dir: Path, out_dir: Path, target_tokens: int,
                shard_bytes: int) -> int:
    bsm_path = morpheus_dir / BERNAT_LOCAL_FILE
    if not bsm_path.is_file():
        raise SystemExit(
            f"BERnaT BSM file not found: {bsm_path}\n"
            f"Pass --morpheus-dir <path> pointing at a morpheus-mamba checkout,\n"
            f"or use --from-hf to stream from HuggingFace instead."
        )
    print(f"[local] cleaning {bsm_path} ({bsm_path.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)

    write, close = shard_writer(out_dir, shard_target_bytes=shard_bytes)
    total_tokens = 0
    seen_hashes: set[str] = set()
    kept = dropped = 0
    try:
        with open(bsm_path, "r", encoding="utf-8") as src:
            for raw in src:
                text = _process_line(raw)
                if text is None:
                    dropped += 1
                    continue
                h = hashlib.blake2b(text[:200].encode("utf-8"), digest_size=8).hexdigest()
                if h in seen_hashes:
                    dropped += 1
                    continue
                seen_hashes.add(h)
                write(text)
                total_tokens += estimate_tokens(text)
                kept += 1
                if kept % 50_000 == 0:
                    print(f"  kept={kept:,} dropped={dropped:,} tokens≈{total_tokens:,}",
                          file=sys.stderr)
                if total_tokens >= target_tokens:
                    print(f"[done] reached token budget {total_tokens:,}", file=sys.stderr)
                    break
    finally:
        close()
    print(f"[local] kept={kept:,} dropped={dropped:,} tokens≈{total_tokens:,}", file=sys.stderr)
    return total_tokens


# --------------------------------------------------------------------------- #
# Mode 2: HuggingFace streaming
# --------------------------------------------------------------------------- #

def clean_from_hf(out_dir: Path, target_tokens: int, shard_bytes: int,
                  hf_token: str | None) -> int:
    from datasets import load_dataset
    print(f"[hf] streaming {BERNAT_REPO} config={BERNAT_CONFIG!r}", file=sys.stderr)
    ds = load_dataset(BERNAT_REPO, BERNAT_CONFIG, split="train",
                      streaming=True, token=hf_token)

    write, close = shard_writer(out_dir, shard_target_bytes=shard_bytes)
    total_tokens = 0
    seen_hashes: set[str] = set()
    kept = dropped = 0
    try:
        for ex in ds:
            raw = ex.get(BERNAT_TEXT_KEY) or ""
            if not isinstance(raw, str):
                continue
            text = _process_line(raw)
            if text is None:
                dropped += 1
                continue
            h = hashlib.blake2b(text[:200].encode("utf-8"), digest_size=8).hexdigest()
            if h in seen_hashes:
                dropped += 1
                continue
            seen_hashes.add(h)
            write(text)
            total_tokens += estimate_tokens(text)
            kept += 1
            if kept % 50_000 == 0:
                print(f"  kept={kept:,} dropped={dropped:,} tokens≈{total_tokens:,}",
                      file=sys.stderr)
            if total_tokens >= target_tokens:
                print(f"[done] reached token budget {total_tokens:,}", file=sys.stderr)
                break
    finally:
        close()
    print(f"[hf] kept={kept:,} dropped={dropped:,} tokens≈{total_tokens:,}", file=sys.stderr)
    return total_tokens


# --------------------------------------------------------------------------- #
# Self-test: verify the noise strippers behave as documented
# --------------------------------------------------------------------------- #

def selftest() -> int:
    cases = [
        # (raw, must_contain, must_not_contain)
        ("Kaixo @erabiltzailea 😃 https://t.co/abc #eguna", "Kaixo", "@", "https", "#"),
        ("RT @user: horrela da", "horrela da", "RT", "@"),
        ("Egunon mundua 🌍🎉", "Egunon mundua", "🌍", "🎉"),
        ("que tal estas bien", None, None, None),  # Spanish → dropped
    ]
    failures = 0
    for i, (raw, must, *must_not) in enumerate(cases):
        cleaned = strip_noise(raw)
        keep = is_keepable(cleaned) if cleaned else False
        print(f"  [{i}] raw={raw!r}")
        print(f"      cleaned={cleaned!r} keep={keep}")
        if must is None:
            # expect dropped
            if keep:
                print("      ✗ should have been dropped")
                failures += 1
            else:
                print("      ✓ dropped as expected")
            continue
        if must not in cleaned:
            print(f"      ✗ expected {must!r} in output")
            failures += 1
        for bad in must_not:
            if bad and bad in cleaned:
                print(f"      ✗ {bad!r} should have been stripped")
                failures += 1
        if not failures:
            print("      ✓ ok")
    return failures


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase1b_bernat.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--out", default="corpora/conversational",
                    help="Output directory for shard_*.txt (default: corpora/conversational)")
    ap.add_argument("--morpheus-dir", default=None,
                    help="Path to a morpheus-mamba checkout. "
                         "Used to find the BSM file in local mode.")
    ap.add_argument("--from-hf", action="store_true",
                    help="Stream from HuggingFace (HiTZ/BERnaT-Diverse) instead of "
                         "reading the local morpheus BSM file.")
    ap.add_argument("--target-tokens", type=int, default=None,
                    help=f"Approximate token budget (default: {BERNAT_TARGET_TOKENS:,})")
    ap.add_argument("--shard-bytes", type=int, default=None,
                    help="Target bytes per shard file.")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token (only used with --from-hf)")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the built-in cleaning self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        n = selftest()
        sys.exit(1 if n else 0)

    cfg = load_config(args.config, args.mode)
    target_tokens = pick(args.target_tokens, cfg, "target_tokens", BERNAT_TARGET_TOKENS)
    shard_bytes = pick(args.shard_bytes, cfg, "shard_bytes", 256 * 1024 * 1024)
    # --from-hf can also be set in config
    from_hf = args.from_hf or cfg.get("from_hf", False)

    out_dir = Path(args.out)
    if from_hf:
        total = clean_from_hf(out_dir, target_tokens, shard_bytes, args.hf_token)
    else:
        morpheus_dir = pick(args.morpheus_dir, cfg, "morpheus_dir", "../morpheus-mamba")
        total = clean_local(Path(morpheus_dir), out_dir, target_tokens, shard_bytes)

    print(f"[final] total_tokens≈{total:,} out={out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
