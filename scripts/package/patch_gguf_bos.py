#!/usr/bin/env python3
"""
Patch eu_futo_v2.gguf: add tokenizer.add_bos_token=false.

The shipped GGUF LACKS this field. llama.cpp defaults to BOS-on for the
Llama architecture, which degrades the model:
  - FUTO-format autocorrect: 82.5% (no BOS) → 60.0% (with BOS)
  - Raw-text re-ranking (surprisal): 9/10 (no BOS) → 7/10 (with BOS)

There is no wllama API override for BOS — it's purely metadata-driven.
So we patch the GGUF itself. Creates a NEW file (original untouched) with
all tensors + metadata preserved, plus the one added field.

Usage:
    uv run python scripts/package/patch_gguf_bos.py
"""
import sys
from pathlib import Path
import gguf


def copy_with_added_field(reader: gguf.GGUFReader, writer: gguf.GGUFWriter,
                          added: dict) -> None:
    """added: {key: (value, GGUFValueType)}"""
    # 1. Copy all existing metadata fields. field.contents() returns the
    #    value in the right Python type for any field (scalar or array).
    for field in reader.fields.values():
        # Suppress virtual fields + fields GGUFWriter manages itself.
        # general.architecture is set by the writer's `arch` constructor arg
        # (we pass "llama"); GGUF.* are virtual. Matches the reference impl.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    # 2. Add the new field(s).
    for key, (value, vtype) in added.items():
        writer.add_key_value(key, value, vtype)
        print(f"  + {key} = {value} ({vtype.name})")

    # 3. Copy tensor info + data.
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()


def main():
    repo = Path(__file__).resolve().parents[2]
    src = repo / "gguf" / "eu_futo_v2.gguf"
    dst = repo / "gguf" / "eu_futo_v2_nobos.gguf"

    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading:  {src}")
    reader = gguf.GGUFReader(str(src))

    existing = reader.get_field("tokenizer.add_bos_token")
    if existing is not None:
        print("NOTE: tokenizer.add_bos_token already present — overwriting to false.")

    print(f"Writing:  {dst}")
    # 2nd arg is `arch` — MUST be "llama" so llama.cpp recognizes the model.
    writer = gguf.GGUFWriter(str(dst), "llama")

    copy_with_added_field(reader, writer, {
        "tokenizer.add_bos_token": (False, gguf.GGUFValueType.BOOL),
    })

    print(f"\nVerifying patched GGUF...")
    r2 = gguf.GGUFReader(str(dst))
    f = r2.get_field("tokenizer.add_bos_token")
    val = bool(f.parts[f.data[0]][0]) if f else None
    n_tensors = len(r2.tensors)
    print(f"  tokenizer.add_bos_token = {val}")
    print(f"  tensors copied: {n_tensors}")
    print(f"  size: {dst.stat().st_size:,} bytes (src: {src.stat().st_size:,})")
    if val is not False:
        print("ERROR: BOS field not set correctly!", file=sys.stderr)
        sys.exit(1)
    print("\n✓ Done. Ship eu_futo_v2_nobos.gguf as the Tier 2 model "
          "(wllama will default to BOS-off).")


if __name__ == "__main__":
    main()
