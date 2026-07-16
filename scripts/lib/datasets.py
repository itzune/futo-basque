"""
Streaming datasets for FUTO keyboard multi-task finetuning.

``MultiTaskFinetuneDataset`` interleaves two strictly-segregated data streams:

  1. **Pure plain text** (majority): streamed from corpus shards, packed into
     fixed-length sequences, ALL loss weights = 1.0. Teaches next-word
     prediction on natural Basque. Never contains ``<XBU>`` / ``<CHAR_*>``.

  2. **Isolated autocorrect triples** (minority): ``<XBU><CHAR_*>...<XBC>
     correct<XEC>`` drawn from synth/real typo pairs. Loss weights = 1.0 only
     on the correction span (``<XBC>`` … ``<XEC>``); 0.0 elsewhere. This means
     the model receives **zero gradient for spontaneously generating ``<XBU>``
     from plain text** — the transition ``[plain] → <XBU>`` is never rewarded.

The two streams are kept SEPARATE at the sequence level (never inline-mixed).
A sequence is either 100% plain text or 100% one isolated triple. This is the
critical fix versus the old Phase 4b inline-mixing approach, which trained the
model on a fake dialect where every third word was a structural block and
caused 100% format contamination (model always emits ``<XBU>`` as top-1).

Because the C++ inference layer *injects* ``<XBU>`` when a keypress correction
is needed, the model never needs to generate ``<XBU>`` autonomously — so
masking its generation during training is exactly correct.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
import sentencepiece as spm

from scripts.lib.typo_synthesis import make_xbu_triple
from scripts.lib.plw_trainer import build_loss_weights_for_correction_only


class MultiTaskFinetuneDataset(IterableDataset):
    """Interleave pure plain text with isolated autocorrect triples.

    Parameters
    ----------
    shard_paths : list[str]
        ``shard_*.txt`` files of clean plain-text corpus (Latxa v2 / BSM).
    synth_jsonl, real_jsonl : str
        JSON files of ``{"typed": ..., "committed": ...}`` typo pairs.
    sp_model_path : str
        SentencePiece UNIGRAM model (must contain ``<XBU>``/``<XBC>``/``<XEC>``).
    seq_len : int
        Fixed sequence length (padding applied to short triples).
    plain_ratio : float
        Probability of drawing a plain-text sample (vs. a triple).
    real_mix_ratio : float
        Of the triple samples, fraction drawn from *real* (vs. synth) typos.
    """

    def __init__(
        self,
        shard_paths: list[str],
        synth_jsonl: str,
        real_jsonl: str,
        sp_model_path: str,
        seq_len: int = 512,
        plain_ratio: float = 0.60,
        real_mix_ratio: float = 0.25,
        seed: int = 1337,
        shuffle_buffer: int = 1024,
    ):
        self.shard_paths = sorted(shard_paths)
        self.synth_pairs = json.loads(Path(synth_jsonl).read_text())
        self.real_pairs = json.loads(Path(real_jsonl).read_text())
        self.sp_model_path = sp_model_path
        self.seq_len = seq_len
        self.plain_ratio = plain_ratio
        self.real_mix_ratio = real_mix_ratio
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        print(
            f"[dataset] shards={len(self.shard_paths)} "
            f"synth={len(self.synth_pairs)} real={len(self.real_pairs)} "
            f"plain_ratio={plain_ratio:.2f} real_mix_ratio={real_mix_ratio:.2f}"
        )

    def _iter_plain_shards(self, worker_id: int, num_workers: int):
        """Yield raw text lines from corpus shards (worker-sharded)."""
        for i, path in enumerate(self.shard_paths):
            if i % num_workers != worker_id:
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def _iter_plain_sequences(self, sp, bos, eos, worker_id, num_workers, rng):
        """Pack plain text into seq_len chunks (no <XBU>, no typo injection).

        All loss_weights = 1.0 — learn next-word on every token. Tokenization
        matches pretrain: BOS + sp.encode(line) + EOS, packed to seq_len.
        """
        buffer: list[int] = []
        line_buffer: list[str] = []

        for line in self._iter_plain_shards(worker_id, num_workers):
            line_buffer.append(line)
            if len(line_buffer) >= self.shuffle_buffer:
                rng.shuffle(line_buffer)
                for raw in line_buffer:
                    buffer.append(bos)
                    buffer.extend(sp.encode(raw, out_type=int))
                    buffer.append(eos)
                    while len(buffer) >= self.seq_len:
                        ids = buffer[: self.seq_len]
                        del buffer[: self.seq_len]
                        yield {
                            "input_ids": torch.tensor(ids, dtype=torch.long),
                            "labels": torch.tensor(ids, dtype=torch.long),
                            "loss_weights": torch.ones(
                                self.seq_len, dtype=torch.float32
                            ),
                            "attention_mask": torch.ones(
                                self.seq_len, dtype=torch.long
                            ),
                        }
                line_buffer.clear()

    def _iter_triples(self, sp, bos, eos, pad, xbc_id, xec_id, rng):
        """Yield isolated <XBU>typo<XBC>correct<XEC> triples, padded to seq_len.

        loss_weights: 1.0 for correction span (<XBC>..<XEC>), 0.0 elsewhere.
        The model never gets loss for generating <XBU> (it's masked).
        """
        while True:
            if rng.random() < self.real_mix_ratio and self.real_pairs:
                pair = rng.choice(self.real_pairs)
            else:
                pair = rng.choice(self.synth_pairs)
            typo, correct = pair["typed"], pair["committed"]
            if not typo or not correct or typo == correct:
                continue
            triple = make_xbu_triple(typo, correct)
            ids = [bos] + sp.encode(triple, out_type=int) + [eos]
            if len(ids) > self.seq_len:
                continue
            input_ids = ids + [pad] * (self.seq_len - len(ids))
            labels = [t if t != pad else -100 for t in input_ids]
            loss_weights = build_loss_weights_for_correction_only(
                input_ids,
                xbc_id=xbc_id,
                xec_id=xec_id,
                plw_clean=0.0,
                in_span_weight=1.0,
            )
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
                "attention_mask": torch.tensor(
                    [1 if t != pad else 0 for t in input_ids], dtype=torch.long
                ),
            }

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id * 9973)

        sp = spm.SentencePieceProcessor()
        sp.load(self.sp_model_path)
        bos = sp.bos_id()
        eos = sp.eos_id()
        pad = sp.pad_id()
        xbc_id = sp.piece_to_id("<XBC>")
        xec_id = sp.piece_to_id("<XEC>")

        plain_iter = self._iter_plain_sequences(
            sp, bos, eos, worker_id, num_workers, rng
        )
        triple_iter = self._iter_triples(sp, bos, eos, pad, xbc_id, xec_id, rng)

        while True:
            if rng.random() < self.plain_ratio:
                try:
                    yield next(plain_iter)
                except StopIteration:
                    # Plain corpus exhausted — fall through to triples only.
                    yield next(triple_iter)
            else:
                yield next(triple_iter)
