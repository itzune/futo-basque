# futo-basque

Train a Transformer language model for **Basque (euskara, `eu`)** that is
compatible with [FUTO Keyboard](https://gitlab.futo.org/keyboard/latinime)'s
"Transformer LM" system. The pipeline mirrors the community
[`danmaxis/futo-portuguese`](https://github.com/danmaxis/futo-portuguese) project,
adapted for Basque.

The model is a ~36M-parameter Llama (Q6_K-quantised GGUF) that plugs into the
keyboard for next-word prediction **and** autocorrect via the
`<XBU>…<XBC>…<XEC>` keypress format.

> See [`RESEARCH.md`](./RESEARCH.md) for the full reverse-engineering notes on
> FUTO's model architecture, tokenizer layout, GGUF metadata, and prompt format.

---

## Download the pre-trained model

If you just want to use Basque autocorrect + next-word prediction in FUTO
Keyboard (no training required):

1. Download [`eu_futo_v2.gguf`](https://github.com/itzune/futo-basque/releases/latest/download/eu_futo_v2.gguf) (~49 MB)
2. In FUTO Keyboard: **Settings → Languages & Models → Import model**
3. Select the downloaded `.gguf` file

**Results** (full model, 25M params, 3B pretrain tokens):

| Metric | Score |
|--------|-------|
| Autocorrect top-1 | 82.5% |
| Autocorrect top-5 | 95.0% |
| Next-word top-1 | 8.3% |

> To train from scratch or reproduce, follow the [Quick start](#quick-start) below.

---

## Why Basque works without app patches

Basque uses the Latin-26 alphabet + **ñ**. Standard batua barely uses acute
accents (á é í ó ú) or ü, so the only routine diacritic typo is **ñ→n**. NFD
decomposition handles ñ→n exactly as it handles Portuguese ã→a, so **no new
tokens, no app patches, no keyboard-layout changes** are needed. FUTO already
ships a Basque keyboard layout (`locales/eu.json`); this project supplies the
matching language model.

---

## Pipeline

All scripts run as modules from the repo root: `uv run python -m scripts.<phase>.<script>`.

| Phase | Script | What it does | Config | Needs |
|-------|--------|--------------|--------|-------|
| **0** | `scripts.reference.inspect_model` | Dump the reference English model's metadata + extract its SentencePiece tokenizer | — | `reference_model/*.gguf` |
| **0** | `scripts.reference.dump_slot_map` | Annotated dump of the 300 user-defined-symbol slots (IDs 4–303) | — | `reference_model/*.gguf` |
| **1** | `scripts.corpus.build_corpus` | Stage the 11 Morpheus-cleaned Latxa v2 sources → `corpora/clean/` shards | `phase1_corpus.yaml` | morpheus repo or network |
| **1b** | `scripts.corpus.clean_bernat` | Clean + stage BERnaT BSM social-media posts → `corpora/conversational/` shards | `phase1b_bernat.yaml` | morpheus repo or network |
| **2** | `scripts.tokenizer.train` | Train the SentencePiece UNIGRAM tokenizer (vocab=4096, 300 fixed structural symbols) | `phase2_tokenizer.yaml` | corpus shards |
| **3** | `scripts.pretrain.train` | Pretrain the 25M Llama base model on the **clean tier only** | `phase3_pretrain.yaml` | **GPU** |
| **4a** | `scripts.finetune.build_wordfreq` | Build a word-frequency map from the corpus (for typo sampling) | — | corpus shards |
| **4a** | `scripts.finetune.generate_triples` | Generate synth + real typo→correct JSON pairs | `phase4a_dataprep.yaml` | wordfreq.json |
| **4a** | `scripts.finetune.isolated` | Fine-tune on `<XBU>typo<XBC>correct<XEC>` triples (isolated autocorrect) | `phase4a_isolated.yaml` | **GPU** + base ckpt |
| **4b** | `scripts.finetune.fulltext` | Fine-tune on in-context corrupted sentences (~33% of words typo'd) | `phase4b_fulltext.yaml` | **GPU** + 4a ckpt |
| **4c** | `scripts.finetune.conversational` | **Conversational adaptation** on BERnaT BSM (register shift → chat) | `phase4c_conversational.yaml` | **GPU** + 4b ckpt + BSM |
| **5** | `scripts.package.to_gguf` | Convert HF checkpoint → GGUF + patch FUTO `keyboardlm.*` metadata | `phase5_package.yaml` | finetune ckpt |
| **5** | `scripts.package.patch_metadata` | (called by 5) Write `keyboardlm.*` fields into the GGUF | — | — |
| **5** | `scripts.package.downgrade_v2` | Downgrade GGUF v3→v2 + strip fields the app's llama.cpp doesn't understand | — | GGUF |
| **eval** | `scripts.eval.keyboard` | Autocorrect + next-word accuracy on a hand-curated Basque test set | — | **GPU** + ckpt |

### Library helpers (`scripts/lib/`)

| File | Purpose |
|------|---------|
| `runconfig.py` | YAML run-config loader with mini/full mode support (CLI > config > default) |
| `typo_synthesis.py` | Generate plausible typos (QWERTY adjacency, ñ-loss, transposition, doubling, shortcuts) → `<XBU>…<XBC>…<XEC>` format |
| `progress.py` | Compact training-progress logging callback |
| `plw_trainer.py` | HF Trainer subclass with Prompt-Loss-Weighting (PLW) for correction-only loss |
| `real_eval_callback.py` | Periodic real-typo eval during fine-tuning → CSV |

### Config

**Language-specific data** (corpus sources, word lists, eval tests) lives in
[`config/eu.py`](./config/eu.py). **Run-strategy decisions** (token budgets,
steps, hyperparams, which corpus each phase uses) live in declarative YAML
files under [`configs/`](./configs/). See [`configs/README.md`](./configs/README.md)
for the full guide.

Every script accepts `--config <path> --mode mini|full`. CLI args always
override config values (for quick experiments). The runbook passes the right
config to each phase automatically.

---

## Quick start

```bash
# 1. Install deps (CPU-only torch is fine for phases 0–2, 5, eval-reference)
uv sync

# 2. Fetch the reference English model (the format spec we must match)
uv run python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('breadlicker45/futo-keyboard-lm', 'ml4_1_f16_meta_fixed.gguf', local_dir='reference_model')
print('saved to', p)
"

# 3. Inspect it (writes notes/reference_*.txt + reference_model/extracted_spm.model)
uv run python -m scripts.reference.inspect_model
uv run python -m scripts.reference.dump_slot_map

# 4. Build the corpus (two tiers; both write shard_*.txt)
#    Clean tier (tokenizer + pretrain base) — reuse morpheus's cleaned Latxa v2:
uv run python -m scripts.corpus.build_corpus --config configs/phase1_corpus.yaml --mode full \
    --morpheus-dir ../morpheus-mamba --out corpora/clean
#    Conversational tier (Phase 4c only) — cleaned BERnaT BSM from HuggingFace:
uv run python -m scripts.corpus.clean_bernat --config configs/phase1b_bernat.yaml --mode full \
    --from-hf --out corpora/conversational

# 5. Train the tokenizer (clean tier ONLY — BSM excluded to keep the vocab morpheme-focused)
uv run python -m scripts.tokenizer.train --config configs/phase2_tokenizer.yaml --mode full \
    --corpus corpora/clean --out tokenizer/spm_eu

# 6–9. Pretrain + fine-tune ON A GPU HOST (all params from configs/):
#   uv run python -m scripts.pretrain.train \
#       --config configs/phase3_pretrain.yaml --mode full \
#       --tokenizer tokenizer/spm_eu.model --corpus corpora/clean --out pretrain
#   uv run python -m scripts.finetune.build_wordfreq --corpus corpora/clean --out notes/wordfreq.json
#   uv run python -m scripts.finetune.generate_triples --config configs/phase4a_dataprep.yaml --mode full
#   uv run python -m scripts.finetune.isolated \
#       --config configs/phase4a_isolated.yaml --mode full \
#       --base pretrain/base --tokenizer tokenizer/spm_eu.model
#   uv run python -m scripts.finetune.fulltext \
#       --config configs/phase4b_fulltext.yaml --mode full \
#       --base finetune/stage_a/final --tokenizer tokenizer/spm_eu.model --corpus corpora/clean
#   uv run python -m scripts.finetune.conversational \
#       --config configs/phase4c_conversational.yaml --mode full \
#       --base finetune/stage_b/final --tokenizer tokenizer/spm_eu.model --corpus corpora/conversational

# 10. Package → GGUF (converts, patches metadata, downgrades v3→v2 automatically)
uv run python -m scripts.package.to_gguf \
    --config configs/phase5_package.yaml --mode full \
    --checkpoint finetune/stage_c/final --tokenizer tokenizer/spm_eu.model \
    --llama-cpp /path/to/llama.cpp
```

Transfer the final `.gguf` to your phone and side-load it via **FUTO Keyboard →
Settings → Languages & Models → Import model**.

---

## Critical constraints (don't break these)

These are reverse-engineered hard requirements — violating any one crashes the
app or silently breaks autocorrect (full details in `RESEARCH.md`):

1. **`<CHAR_X>` is a keypress token, not literal text.** The typed part of an
   autocorrect triple is encoded as one `<CHAR_A>`…`<CHAR_Z>` per keystroke
   (accent-stripped, uppercased via NFD), *not* the raw word: `<XBU><CHAR_T><CHAR_E><CHAR_H><XBC>The<XEC>`.
2. **`char_embed_mixing_v1` is required** whenever `xbu_char_autocorrect_v1` is
   enabled — without it the app **SIGSEGVs** at inference. The features string
   must be: `base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1`.
3. **GGUF must be v2**, not v3. The app's vendored llama.cpp only parses v2.
   Run `downgrade_v2` if `convert_hf_to_gguf` emitted v3.
4. **`keyboardlm.ext_tokenizer_data` must be `[UINT8]`**, not `[INT32]`. The
   patch script forces this via `add_key_value(..., sub_type=UINT8)`.
5. **`<CHAR_A>`…`<CHAR_Z>` must be 26 contiguous sequential IDs** (182–207).
   The tokenizer asserts this; don't reorder the structural symbols.

### Tokenizer layout (IDs)

```
0–3      pad / bos / eos / unk
4–303    300 user-defined symbols (declaration order = ID order)
           4–27    <FUTO0>..<FUTO23>          (filler)
           28–173  146 Basque common words    (content — replaceable)
           174–176 <XBU> <XBC> <XEC>          (STRUCTURAL — keep)
           177–181 <XC0>..<XC4>               (STRUCTURAL — keep)
           182–207 <CHAR_A>..<CHAR_Z>         (STRUCTURAL — keep, sequential)
           208–263 56 Basque adjectives       (content — replaceable)
           264–303 40 emoji                   (content — replaceable)
304–559  byte fallback <0x00>..<0xFF>
560–4095 learned UNIGRAM pieces (3536 pieces)
```

**Trainer parameters** (verified against the reference model in Phase 0):
`model_type=unigram`, `byte_fallback=True`, `character_coverage=0.9995`,
`treat_whitespace_as_suffix=True` (inverted_space), `add_dummy_prefix=False`,
`remove_extra_whitespaces=False`, `pad/bos/eos/unk=0/1/2/3`.

### Vocabulary size: 4096 (not the reference's 15008)

The reference English model uses vocab=15008, but English is isolating — surface-form
memorization is fine. **Basque is agglutinative**: a large vocab memorizes common
inflected forms (`etxea`, `etxera`, `etxetik`) as single tokens, so the model never
learns that `-a`/`-ra`/`-tik` are reusable suffixes.

A controlled ablation in our sibling project [`morpheus-mamba`](https://github.com/itzune/morpheus)
(21 Basque test words, 4 vocab sizes) found:

| Vocab (learned) | MorphAcc | Behavior |
|-----------------|----------|----------|
| ~4K | **66.7%** | `etxetik → ▁etxe + tik` ✅ morpheme-aligned |
| ~8K | 61.9% | Inconsistent |
| ~16K | 52.4% | Surface-form memorization begins |
| ~32K | 28.6% | `etxetik → ▁etxetik` ❌ whole-word |

FUTO has 560 reserved slots, so vocab=4096 gives 3536 learned pieces — matching the
4K MorphAcc regime. The 36M model is smaller than morpheus's 91M, so morpheme splitting
helps even more (less capacity to waste on surface forms). The 2048 context window
easily absorbs the 39% fertility increase (2.58 vs 1.85 tokens/word).

The FUTO app reads vocab size dynamically (`llama_n_vocab()`), so this is fully
compatible — no hardcoding. The tokenizer training script includes a MorphAcc
spot-check that warns if splitting degrades.

---

## Basque-specific adaptation notes

- **Corpus** (two tiers — see RESEARCH.md §11.4, §11.6 for the full rationale):
  - *Clean* (`corpora/clean/`): Morpheus's cleaned [Latxa corpus v2](https://huggingface.co/datasets/HiTZ/latxa-corpus-v2) — 11 HiTZ-curated, deduplicated sources (~4.77 B tokens, LLM-audited avg quality 4.6/5). Used for **tokenizer training + pretrain base + Phase 4b**. Target: **3B tokens staged** (not 5B — our 25M model's sweet spot is 80-120:1 ratio, §11.6).
  - *Conversational* (`corpora/conversational/`): [BERnaT BSMtime](https://huggingface.co/datasets/HiTZ/BERnaT-Diverse) social-media posts (~250 M tokens), aggressively cleaned (emoji/URL/mention/code-switch stripping). **Phase 4c only** — excluded from tokenizer and pretrain per the revised strategy (§11.6). FUTO's own English pipeline uses SlimPajama (web) for pretrain + a small conversational finetune at the end; we follow the same pattern. The BERnaT paper (Azurmendi et al. 2025) shows diverse data helps without hurting standard-form accuracy.
- **Phase 4c (conversational adaptation)**: NEW stage matching FUTO's own pipeline. The model is finetuned on BSM with the same 1/3 typo augmentation but at lower LR (2e-5). FUTO's wiki calls this "an important step" — without it, the model suggests Wikipedia-register continuations instead of chat.
- **Typo synthesis**: QWERTY (not ABNT2) adjacency; the Portuguese accent-swap
  rule (`é→ê`) is removed (Basque has no such accents); ñ→n handled by NFD.
  Adjacency + transposition/doubling are the dominant Basque typo classes.
- **Tokenizer slots**: 146 + 56 high-frequency Basque words (function words,
  pronouns, auxiliaries, common verbs/nouns/adjectives). Padded with `<FUTO>`
  filler to the exact slot count. The structural slots are untouched.

---

## Status

- [x] Project scaffold + config + YAML run-configs (`configs/*.yaml`)
- [x] All scripts ported from `futo-portuguese`, adapted for Basque
- [x] Phase 4c (conversational adaptation) added — matching FUTO's own pipeline
- [x] Phase 0: download reference model, dump metadata, verify slot layout
- [x] Corpus strategy: Latxa v2 (clean) + BERnaT BSM (conversational, Phase 4c)
- [x] Mini validation: full pipeline (1→5) runs end-to-end, model works in FUTO app
- [x] Full run: 3B pretrain tokens, all finetune phases (4a/4b/4c), model tested in FUTO app
