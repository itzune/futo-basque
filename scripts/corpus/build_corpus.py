"""
Phase 1: assemble the PRIMARY Basque corpus (the clean tier).

The corpus is the morpheus-mamba project's cleaned Latxa corpus v2
(<morpheus_repo>/data/clean-v3/), which HiTZ already curated, deduplicated,
and quality-audited. We stage those per-source .txt files into our uniform
shard_*.txt format (~256 MB each) for the tokenizer + pretrain phases.

This script writes ONLY the clean tier (corpora/clean/). The conversational
tier (BERnaT BSM) is handled separately by clean_bernat.py and goes to
corpora/conversational/ — it is excluded from tokenizer training.

Two modes:
  1. Local (default): stage from a local morpheus repo checkout.
       uv run python -m scripts.corpus.build_corpus \
           --morpheus-dir ../morpheus-mamba --out corpora/clean
  2. HF fallback (--from-hf): stream from HiTZ/latxa-corpus-v2 per-source
     configs on HuggingFace (applies lighter cleaning). Use this only when the
     morpheus clean-v3 dir is unavailable.
       uv run python -m scripts.corpus.build_corpus --from-hf --out corpora/clean

Optional:
  --sources euscrawl-v2 wikipedia   # subset to specific sources (smoke test)
  --target-tokens 3_000_000_000     # stop after reaching a token budget
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from config.eu import (
    CHARS_PER_TOKEN,
    LATXA_HF_CONFIGS,
    LATXA_HF_EXCLUDED_CONFIGS,
    LATXA_HF_REPO,
    LATXA_HF_TEXT_KEY,
    LATXA_SOURCES,
    is_likely_eu,
)
from scripts.lib.runconfig import load_config, pick

MIN_DOC_CHARS = 100


# --------------------------------------------------------------------------- #
# Shard writer (shared by both modes)
# --------------------------------------------------------------------------- #

def shard_writer(out_dir: Path, shard_target_bytes: int = 256 * 1024 * 1024):
    """Return (write_fn, close_fn). Rotates shard files at ~shard_target_bytes."""
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
    # Basque is agglutinative so words run a little longer than Romance.
    return max(1, len(text) // CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- #
# Mode 1: stage from local morpheus clean-v3
# --------------------------------------------------------------------------- #

def stage_local(morpheus_dir: Path, out_dir: Path, sources: list[str] | None,
                target_tokens: int, shard_bytes: int) -> int:
    """Stage morpheus clean-v3 per-source .txt files into shard_*.txt.

    These files are already cleaned + deduplicated, so we pass them through
    with only a length guard (no re-cleaning — that would undo morpheus's work).
    """
    clean_dir = morpheus_dir / "data" / "clean-v3"
    if not clean_dir.is_dir():
        raise SystemExit(
            f"clean-v3 dir not found: {clean_dir}\n"
            f"Pass --morpheus-dir <path> pointing at a morpheus-mamba checkout,\n"
            f"or use --from-hf to stream from HuggingFace instead."
        )

    # Glob all .txt in clean-v3 (post-exclusion = exactly the 11 approved sources).
    all_files = sorted(clean_dir.glob("*.txt"))
    if not all_files:
        raise SystemExit(f"No .txt files in {clean_dir}")

    # Optionally subset by source name (substring match on filename).
    if sources:
        wanted = set(sources)
        files = [p for p in all_files if any(s in p.name for s in wanted)]
        missing = wanted - {s for s in wanted for p in all_files if s in p.name}
        if missing:
            print(f"[warn] sources not found in clean-v3: {missing}", file=sys.stderr)
            print(f"       available: {[p.name for p in all_files]}", file=sys.stderr)
    else:
        files = all_files

    print(f"[local] staging {len(files)} source(s) from {clean_dir}", file=sys.stderr)
    for p in files:
        print(f"  {p.name}", file=sys.stderr)

    write, close = shard_writer(out_dir, shard_target_bytes=shard_bytes)
    total_tokens = 0
    try:
        for path in files:
            if total_tokens >= target_tokens:
                break
            kept = 0
            with open(path, "r", encoding="utf-8") as src:
                for line in src:
                    text = line.rstrip("\n")
                    if not text:
                        continue  # skip empty lines; data is pre-cleaned by morpheus
                    write(text)
                    total_tokens += estimate_tokens(text)
                    kept += 1
                    if total_tokens >= target_tokens:
                        print(f"[done] reached token budget {total_tokens:,}", file=sys.stderr)
                        break
            print(f"  [{path.name}] staged {kept:,} lines", file=sys.stderr)
    finally:
        close()
    return total_tokens


# --------------------------------------------------------------------------- #
# Mode 2: stream from HuggingFace (fallback)
# --------------------------------------------------------------------------- #

def _normalize_hf(text: str) -> str:
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stage_from_hf(out_dir: Path, sources: list[str] | None, target_tokens: int,
                  shard_bytes: int, hf_token: str | None) -> int:
    """Stream Latxa corpus v2 per-source from HuggingFace with light cleaning."""
    from datasets import load_dataset

    configs = LATXA_HF_CONFIGS
    if sources:
        wanted = set(sources)
        unknown = wanted - set(configs)
        if unknown:
            print(f"[warn] unknown sources (not in LATXA_HF_CONFIGS): {unknown}", file=sys.stderr)
        configs = {k: v for k, v in configs.items() if k in wanted}
    if not configs:
        raise SystemExit("No sources selected. Check --sources against LATXA_SOURCES in config/eu.py.")

    import hashlib
    write, close = shard_writer(out_dir, shard_target_bytes=shard_bytes)
    total_tokens = 0
    try:
        for src_name, hf_config in configs.items():
            if total_tokens >= target_tokens:
                break
            if hf_config in LATXA_HF_EXCLUDED_CONFIGS:
                continue
            print(f"[hf] {src_name}: streaming {LATXA_HF_REPO} config={hf_config!r}", file=sys.stderr)
            try:
                ds = load_dataset(
                    LATXA_HF_REPO, hf_config, split="train",
                    streaming=True, token=hf_token,
                )
            except Exception as e:
                print(f"[hf] {src_name}: FAILED to load: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            seen_hashes: set[str] = set()
            kept = dropped = 0
            for ex in ds:
                text = ex.get(LATXA_HF_TEXT_KEY) or ""
                if not isinstance(text, str) or not text:
                    continue
                text = _normalize_hf(text)
                if len(text) < MIN_DOC_CHARS:
                    dropped += 1
                    continue
                if not is_likely_eu(text):
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
                if kept % 10_000 == 0:
                    print(f"  [{src_name}] kept={kept:,} dropped={dropped:,} tokens≈{total_tokens:,}", file=sys.stderr)
                if total_tokens >= target_tokens:
                    print(f"[done] reached token budget {total_tokens:,}", file=sys.stderr)
                    break
            print(f"  [{src_name}] kept={kept:,} dropped={dropped:,}", file=sys.stderr)
    finally:
        close()
    return total_tokens


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase1_corpus.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--out", default="corpora/clean",
                    help="Output directory for shard_*.txt (default: corpora/clean)")
    ap.add_argument("--morpheus-dir", default=None,
                    help="Path to a morpheus-mamba checkout. Used to find "
                         "data/clean-v3/ in local mode.")
    ap.add_argument("--from-hf", action="store_true",
                    help="Stream from HuggingFace (HiTZ/latxa-corpus-v2) instead of "
                         "staging local morpheus files. Applies lighter cleaning.")
    ap.add_argument("--sources", nargs="*", default=None,
                    help=f"Subset of sources (default: all 11). Choices: {LATXA_SOURCES}")
    ap.add_argument("--target-tokens", type=int, default=None,
                    help="Approximate token budget; stop after reaching it.")
    ap.add_argument("--shard-bytes", type=int, default=None,
                    help="Target bytes per shard file.")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token (only used with --from-hf)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)
    morpheus_dir = pick(args.morpheus_dir, cfg, "morpheus_dir", "../morpheus-mamba")
    target_tokens = pick(args.target_tokens, cfg, "target_tokens", 5_000_000_000)
    shard_bytes = pick(args.shard_bytes, cfg, "shard_bytes", 256 * 1024 * 1024)

    out_dir = Path(args.out)
    if args.from_hf:
        total = stage_from_hf(out_dir, args.sources, target_tokens,
                              shard_bytes, args.hf_token)
    else:
        total = stage_local(Path(morpheus_dir), out_dir, args.sources,
                            target_tokens, shard_bytes)

    print(f"[final] total_tokens≈{total:,} out={out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
