"""
Phase 5: end-to-end GGUF assembly for the FUTO Keyboard Basque (eu) model.

Pipeline:
  1. Stage the HF checkpoint into a temp dir, copying spm_eu.model
     into it as `tokenizer.model` (Llama convention) plus the required
     tokenizer_config.json / special_tokens_map.json so that
     convert_hf_to_gguf.py recognises it.
  2. Run llama.cpp/convert_hf_to_gguf.py on the staged dir → vanilla GGUF (v3).
  3. Run patch_metadata.py on the vanilla GGUF → patched GGUF with
     keyboardlm.* fields (still v3).
  4. Run downgrade_v2.py on the patched GGUF → final v2 GGUF.
     FUTO's vendored llama.cpp requires GGUF v2; v3 crashes the app.
  5. Emit summary diff vs the reference English model's metadata.

Usage:
  uv run python -m scripts.package.to_gguf \\
      --checkpoint finetune/final \\
      --tokenizer tokenizer/spm_eu.model \\
      --llama-cpp /path/to/llama.cpp \\
      --out gguf/eu_futo.gguf

Designed to run on this VM (CPU; no GPU needed). Pulls the checkpoint from the
GPU host with rsync first if --checkpoint points at a remote URL like
gpu-train:/workspace/finetune/final.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.lib.runconfig import load_config, pick


SPECIAL_TOKENS_MAP = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "LlamaTokenizer",
    "model_max_length": 2048,
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "add_bos_token": True,
    "add_eos_token": False,
    "clean_up_tokenization_spaces": False,
    "legacy": False,
}


def stage_checkpoint(checkpoint: Path, sp_model: Path, dest: Path) -> None:
    """Copy the model checkpoint files + the SP tokenizer into a single staging dir."""
    dest.mkdir(parents=True, exist_ok=True)
    # Copy model files
    copied = 0
    for f in checkpoint.iterdir():
        if f.name in {"tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
            continue  # we'll write fresh ones
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            copied += 1
    print(f"  Copied {copied} checkpoint files")
    # SP tokenizer → tokenizer.model (Llama convention)
    shutil.copy2(sp_model, dest / "tokenizer.model")
    print(f"  Copied {sp_model.name} → tokenizer.model")
    # Tokenizer config / special tokens map
    (dest / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG, indent=2))
    (dest / "special_tokens_map.json").write_text(json.dumps(SPECIAL_TOKENS_MAP, indent=2))
    print("  Wrote tokenizer_config.json + special_tokens_map.json")


def run_convert(llama_cpp: Path, staged: Path, out_vanilla: Path) -> None:
    cmd = [
        sys.executable,
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(staged),
        "--outfile", str(out_vanilla),
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_patch(scripts_dir: Path, in_gguf: Path, out_gguf: Path, sp_model: Path,
              languages: str, features: str, history: str | None,
              author: str | None = None, description: str | None = None,
              license: str | None = None, url: str | None = None) -> None:
    cmd = [
        sys.executable,
        str(scripts_dir / "patch_metadata.py"),
        "--in", str(in_gguf),
        "--out", str(out_gguf),
        "--tokenizer", str(sp_model),
        "--languages", languages,
        "--features", features,
    ]
    if history:
        cmd += ["--history", history]
    if author:
        cmd += ["--author", author]
    if description:
        cmd += ["--description", description]
    if license:
        cmd += ["--license", license]
    if url:
        cmd += ["--url", url]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_downgrade(scripts_dir: Path, in_gguf: Path, out_gguf: Path) -> None:
    """Downgrade GGUF v3 → v2 + strip fields the FUTO app can't handle."""
    cmd = [
        sys.executable,
        str(scripts_dir / "downgrade_v2.py"),
        "--in", str(in_gguf),
        "--out", str(out_gguf),
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def diff_metadata(llama_cpp: Path, vanilla: Path, ours: Path, reference: Path, out_diff: Path) -> None:
    dump = llama_cpp / "gguf-py" / "gguf" / "scripts" / "gguf_dump.py"

    def dump_one(path: Path, target: Path) -> None:
        with open(target, "w") as f:
            subprocess.run([sys.executable, str(dump), str(path)], stdout=f, check=True)

    notes = ours.parent.parent / "notes"
    notes.mkdir(exist_ok=True)
    ours_dump = notes / "our_metadata.txt"
    dump_one(ours, ours_dump)
    print(f"  Wrote {ours_dump}")

    if reference.exists():
        result = subprocess.run(
            ["diff", str(reference), str(ours_dump)],
            capture_output=True, text=True
        )
        out_diff.write_text(result.stdout)
        # show only the diff highlights
        added = sum(1 for line in result.stdout.splitlines() if line.startswith("> "))
        removed = sum(1 for line in result.stdout.splitlines() if line.startswith("< "))
        print(f"  Diff vs reference: {removed} lines unique to reference, {added} lines unique to ours")
        print(f"  Full diff: {out_diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config file (configs/phase5_package.yaml)")
    ap.add_argument("--mode", default="full", choices=["mini", "full"],
                    help="Which mode section to load from the config")
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint dir (finetune/final/)")
    ap.add_argument("--tokenizer", required=True, help="spm_eu.model")
    ap.add_argument("--llama-cpp", required=True, help="Path to local llama.cpp clone")
    ap.add_argument("--out", default=None, help="Final GGUF output path (e.g. gguf/eu_futo.gguf)")
    ap.add_argument("--reference", default="reference_model/ml4_1_f16_meta_fixed.gguf",
                    help="Reference English model for diff (optional)")
    ap.add_argument("--languages", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--author", default=None)
    ap.add_argument("--description", default=None)
    ap.add_argument("--license", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--keep-staged", action="store_true",
                    help="Keep the staged checkpoint dir for debugging")
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)
    meta = cfg.get("metadata", {})

    out_str = pick(args.out, cfg, "out", "gguf/eu_futo_v2.gguf")
    languages = pick(args.languages, cfg, "languages", "eu")
    features = pick(args.features, cfg, "features",
                    "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1")
    author = pick(args.author, meta, "author", "Xabier Ezpeleta <xezpeleta@gmail.com>")
    description = pick(args.description, meta, "description",
                       "Basque (euskara) transformer language model for FUTO Keyboard.")
    license = pick(args.license, meta, "license", "MIT")
    url = pick(args.url, meta, "url", "https://github.com/xezpeleta/futo-basque")

    checkpoint = Path(args.checkpoint)
    sp_model = Path(args.tokenizer)
    llama_cpp = Path(args.llama_cpp)
    out = Path(out_str)
    reference = Path(args.reference)
    scripts_dir = Path(__file__).resolve().parent

    if not checkpoint.exists():
        sys.exit(f"Checkpoint dir not found: {checkpoint}")
    if not sp_model.exists():
        sys.exit(f"SentencePiece model not found: {sp_model}")
    if not (llama_cpp / "convert_hf_to_gguf.py").exists():
        sys.exit(f"convert_hf_to_gguf.py not found in {llama_cpp}")

    out.parent.mkdir(parents=True, exist_ok=True)
    notes_dir = out.parent.parent / "notes"
    notes_dir.mkdir(exist_ok=True)

    # 1. Stage
    if args.keep_staged:
        staged = out.parent / "_staged"
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True)
        ctx = None
    else:
        ctx = tempfile.TemporaryDirectory()
        staged = Path(ctx.name)

    print(f"[1/5] Staging checkpoint into {staged}")
    stage_checkpoint(checkpoint, sp_model, staged)

    # 2. Convert HF → GGUF (v3)
    vanilla = out.with_suffix(".vanilla.gguf")
    print(f"[2/5] HF -> vanilla GGUF (v3): {vanilla}")
    run_convert(llama_cpp, staged, vanilla)
    print(f"  vanilla GGUF size: {vanilla.stat().st_size:,} bytes")

    # 3. Patch with FUTO metadata (still v3)
    patched = out.with_suffix(".patched.gguf")
    print(f"[3/5] Patching with keyboardlm.* fields -> {patched}")
    run_patch(scripts_dir, vanilla, patched, sp_model, languages, features, args.history,
              author, description, license, url)
    print(f"  patched GGUF size: {patched.stat().st_size:,} bytes")

    # 4. Downgrade v3 → v2 (FUTO requires v2 — v3 crashes the app)
    print(f"[4/5] Downgrading GGUF v3 -> v2 -> {out}")
    run_downgrade(scripts_dir, patched, out)
    print(f"  final GGUF size: {out.stat().st_size:,} bytes")

    # 5. Diff against reference
    print("[5/5] Diffing against reference English model")
    diff_metadata(llama_cpp, vanilla, out, reference, notes_dir / "metadata_diff.txt")

    if ctx is not None:
        ctx.cleanup()
    else:
        print(f"  Staged dir kept at: {staged}")

    print()
    print(f"DONE: {out}")
    print()
    print("Next steps:")
    print(f"  1. Inspect: python {llama_cpp}/gguf-py/gguf/scripts/gguf_dump.py {out} | head -40")
    print(f"  2. Inference smoke test: {llama_cpp}/build/bin/llama-cli -m {out} -p 'Egun on <XBU>eskerik<XBC>' -n 10")
    print(f"  3. Transfer to phone and side-load via FUTO Keyboard's Languages & Models import.")


if __name__ == "__main__":
    main()
