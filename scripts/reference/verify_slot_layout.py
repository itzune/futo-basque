"""Phase 0 verification: compare our tokenizer layout against the reference model."""
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
import sentencepiece as spm
from scripts.tokenizer.train import build_user_defined_symbols

# --- Load reference proto for trainer spec ---
m = sp_pb2.ModelProto()
m.ParseFromString(open("reference_model/extracted_spm.model", "rb").read())
ts = m.trainer_spec
mt = {1: "UNIGRAM", 2: "BPE", 3: "WORD", 4: "CHAR"}
print("=== Reference SPM trainer_spec ===")
print(f"  model_type: {mt.get(ts.model_type, ts.model_type)}")
print(f"  vocab_size: {ts.vocab_size}")
print(f"  byte_fallback: {ts.byte_fallback}")
print(f"  unk_id: {ts.unk_id}  bos_id: {ts.bos_id}  eos_id: {ts.eos_id}  pad_id: {ts.pad_id}")
print(f"  character_coverage: {ts.character_coverage}")
print(f"  input_sentence_size: {ts.input_sentence_size}")
print(f"  num user_defined_symbols: {len(ts.user_defined_symbols)}")
print(f"  normalizer_rule: {m.normalizer_spec.name}")
print(f"  add_dummy_prefix: {m.normalizer_spec.add_dummy_prefix}")
print(f"  remove_extra_whitespaces: {m.normalizer_spec.remove_extra_whitespaces}")
print(f"  escape_whitespaces: {m.normalizer_spec.escape_whitespaces}")
print()

# --- Load via processor for piece comparison ---
ref = spm.SentencePieceProcessor(model_file="reference_model/extracted_spm.model")
print(f"=== Processor: vocab_size={ref.vocab_size()}, unk_id={ref.unk_id()}, bos_id={ref.bos_id()}, eos_id={ref.eos_id()}, pad_id={ref.pad_id()} ===")
print()

# --- Compare structural slots ---
ours = build_user_defined_symbols()
print(f"=== Our build_user_defined_symbols(): {len(ours)} symbols ===")
print()
print("=== STRUCTURAL SLOT VERIFICATION (IDs 174..207) ===")
mismatches = 0
for i, sym in enumerate(ours):
    ref_id = i + 4
    if 174 <= ref_id <= 207:
        ref_sym = ref.id_to_piece(ref_id)
        if ref_sym != sym:
            mismatches += 1
            print(f"  ID {ref_id}: ours={sym!r:16s} ref={ref_sym!r:16s} <-- MISMATCH")
verdict = "PERFECT MATCH" if mismatches == 0 else "FIX NEEDED"
print(f"  Mismatches in structural range: {mismatches} ({verdict})")
print()
print("=== Boundary checks ===")
for check_id, label in [
    (173, "last content-1"), (174, "<XBU>"), (175, "<XBC>"), (176, "<XEC>"),
    (177, "<XC0>"), (181, "<XC4>"), (182, "<CHAR_A>"), (207, "<CHAR_Z>"),
    (208, "first content-2"), (303, "last emoji"),
]:
    i = check_id - 4
    print(f"  ID {check_id:3d} ({label:16s}): ours={ours[i]!r:16s} ref={ref.id_to_piece(check_id)!r}")
