#!/usr/bin/env python3
"""Diagnose 4m loss=20: compute PLWTrainer loss on a full batch."""
import sys, torch, glob, json
import torch.nn.functional as F
from transformers import LlamaForCausalLM
import sentencepiece as spm
from scripts.lib.datasets import MultiTaskFinetuneDataset

CKPT = "pretrain/base"
SP = "tokenizer/spm_eu.model"
BATCH = 24  # match micro_batch

sp = spm.SentencePieceProcessor()
sp.load(SP)
print(f"vocab={sp.get_piece_size()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()}")

# Load in bf16 like training
model = LlamaForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16)
model.eval().to("cpu")

shards = sorted(glob.glob("corpora/clean/shard_*.txt"))[:4]
ds = MultiTaskFinetuneDataset(
    shard_paths=shards, synth_jsonl="notes/synth.json", real_jsonl="notes/real.json",
    sp_model_path=SP, seq_len=512, plain_ratio=0.60, real_mix_ratio=0.25, seed=1337)

# Collect a full batch
it = iter(ds)
batch = [next(it) for _ in range(BATCH)]

# Collate (simple stack, like default_data_collator)
input_ids = torch.stack([s["input_ids"] for s in batch])  # (B, T)
labels = torch.stack([s["labels"] for s in batch])
loss_weights = torch.stack([s["loss_weights"] for s in batch])
attention_mask = torch.stack([s["attention_mask"] for s in batch])

n_plain = sum(1 for s in batch if s["loss_weights"].min() > 0.5)
n_triple = BATCH - n_plain
print(f"Batch: {n_plain} plain, {n_triple} triple")
print(f"input_ids shape: {input_ids.shape}")
print(f"loss_weights present: {loss_weights is not None}")
print(f"labels has -100: {-100 in labels.unique().tolist()}")

# Replicate PLWTrainer.compute_loss EXACTLY
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100, reduction="none",
    ).view(shift_labels.size())

    valid = (shift_labels != -100).float()

    # Weighted (PLWTrainer path)
    shift_weights = loss_weights[..., 1:].contiguous().to(ce.dtype)
    weighted = ce * shift_weights * valid
    denom_w = (shift_weights * valid).sum().clamp_min(1e-8)
    loss_w = (weighted.sum() / denom_w).item()

    # Unweighted (fallback path)
    loss_u = ((ce * valid).sum() / valid.sum().clamp_min(1e-8)).item()

    # Per-sample breakdown
    print(f"\n=== BATCH LOSS ===")
    print(f"  weighted (PLWTrainer):   {loss_w:.4f}")
    print(f"  unweighted (fallback):   {loss_u:.4f}")
    print(f"  denom_w (weighted tokens): {denom_w.item():.0f}")
    print(f"  denom_u (all valid tokens): {valid.sum().item():.0f}")

    print(f"\n=== PER-SAMPLE ===")
    for i in range(BATCH):
        is_plain = batch[i]["loss_weights"].min() > 0.5
        s_valid = valid[i].sum().item()
        s_weights = shift_weights[i] * valid[i]
        s_denom = s_weights.sum().item()
        s_ce = ce[i] * valid[i]
        s_loss_u = (s_ce.sum() / s_valid).item() if s_valid > 0 else 0
        s_loss_w = ((ce[i] * s_weights).sum() / s_denom).item() if s_denom > 0 else 0
        tag = "PLAIN" if is_plain else "TRIPLE"
        if not is_plain:
            nonpad = input_ids[i][input_ids[i] != sp.pad_id()]
            decoded = sp.decode(nonpad.tolist())[:50]
            print(f"  [{tag}] {i:2d} valid={s_valid:.0f} w_tok={s_denom:.0f} "
                  f"loss_u={s_loss_u:.3f} loss_w={s_loss_w:.3f}  {decoded!r}")
        else:
            print(f"  [{tag}] {i:2d} valid={s_valid:.0f} w_tok={s_denom:.0f} "
                  f"loss_u={s_loss_u:.3f} loss_w={s_loss_w:.3f}")

    # Check: is loss_weights actually being used?
    print(f"\n=== KEY CHECK ===")
    print(f"  loss_weights range: [{loss_weights.min().item():.1f}, {loss_weights.max().item():.1f}]")
    print(f"  If weighted==unweighted, loss_weights has no effect (all same value)")
    print(f"  weighted ({loss_w:.4f}) vs unweighted ({loss_u:.4f})")
