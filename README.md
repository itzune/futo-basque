# futo-basque

Train a Transformer language model for **Basque (euskara, `eu`)** that is
compatible with [FUTO Keyboard](https://gitlab.futo.org/keyboard/latinime)'s
"Transformer LM" system. The pipeline mirrors the community
[`danmaxis/futo-portuguese`](https://github.com/danmaxis/futo-portuguese) project,
adapted for Basque.

The model is a **25M-parameter Llama** (f16 GGUF) that plugs into the
keyboard for next-word prediction **and** autocorrect via the
`<XBU>…<XBC>…<XEC>` keypress format.

> See [`RESEARCH.md`](./RESEARCH.md) for the full reverse-engineering notes on
> FUTO's model architecture, tokenizer layout, GGUF metadata, and prompt format.
> See [`TRAINING_PROCESS.md`](./TRAINING_PROCESS.md) for the detailed training
> log, including the pretrain bug postmortem.

---

## Download the pre-trained model

If you just want to use Basque autocorrect + next-word prediction in FUTO
Keyboard (no training required):

1. Download [`eu_futo_v2.gguf`](https://github.com/itzune/futo-basque/releases/latest/download/eu_futo_v2.gguf) (~49 MB)
2. In FUTO Keyboard: **Settings → Languages & Models → Import model**
3. Select the downloaded `.gguf` file

**Results** (25M params, 3B pretrain tokens):

We measure real keyboard utility — **keystrokes saved** while typing realistic
messaging messages (WhatsApp/Telegram-style). After each word, the model
suggests the next word; if correct, those characters are saved (user taps the
suggestion instead of typing).

| Metric | Score |
|--------|:---:|
| Keystrokes saved (top-1 suggestion) | **8.4%** |
| Keystrokes saved (top-5 suggestions bar) | **28.9%** |
| Next-word top-1 (prompt eval) | 50.0% |
| Next-word top-5 (prompt eval) | 41.7% |

On a typical 40-character message, the suggestions bar saves ~12 keystrokes —
about a third of the typing. The model generates clean Basque (`ikasten`,
`zait`, `izango`, `da`) with no control-token contamination.

#### Examples

`Eskerrik asko denagatik oso ondo pasa nuen` ("Thanks for everything, I had a great time")

```
  ✓ Eskerrik ▎                → asko      ✓  saved 4 chars
    Eskerrik asko ▎           → denagatik    (model suggested: zure)
    Eskerrik asko denagatik ▎ → oso          (model suggested: eta)
    …asko denagatik oso ▎     → ondo         (model suggested: pozik)
  ✓ …asko denagatik oso ondo ▎ → pasa      ✓  saved 4 chars
    …denagatik oso ondo pasa ▎ → nuen        (model suggested: duzuen)
  → 8/28 predictable characters saved (29%)
```

`Ongi etorri etxera afaria prest duzu` ("Welcome home, dinner is ready for you")

```
  ✓ Ongi ▎                    → etorri    ✓  saved 6 chars
    Ongi etorri ▎             → etxera      (model suggested: gure)
    …
  → 6/27 predictable characters saved (22%)
```

The model nails formulaic openings and strong collocations (`Eskerrik asko`,
`Ongi etorri`, `Non dago`, `Zein filma ikusi`) where Basque has predictable
patterns, and misses genuinely open-ended content words.

> **Note on autocorrect:** standalone GGUF eval in FUTO control-token format
> (`keyboard.py`) scores 0% on autocorrect — the model learned plain-text
> next-word well but the 40% triple ratio in finetuning wasn't sufficient to
> master the `<XBU><CHAR_*><XBC>…<XEC>` format. In the real FUTO app, the
> [hybrid dictionary engine](./RESEARCH.md) compensates by proposing
> real-word candidates for the transformer to re-rank. The Basque dictionary
> wordlist that powers that engine is now built — see
> [Basque dictionary](#basque-dictionary-autocorrect-candidate-engine) below.

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
| **4a** | `scripts.finetune.generate_triples` | Generate synth + real typo→correct JSON pairs → `notes/synth.json` + `notes/real.json` | `phase4a_dataprep.yaml` | wordfreq.json |
| **4m** | `scripts.finetune.multitask` | **Unified multi-task finetune** — 60% plain text (PLW=1.0) + 40% isolated triples, from pretrain base | `phase4_multitask.yaml` | **GPU** + base ckpt + triples |
| **4mr** | `scripts.finetune.multitask --mode recover` | Restart 4m from a checkpoint with lower LR + grad clipping (after a loss spike) | `phase4_multitask.yaml` | **GPU** + 4m checkpoint |
| *4a* | *`scripts.finetune.isolated`* | *Legacy: fine-tune on `<XBU>typo<XBC>correct<XEC>` triples (isolated)* | *`phase4a_isolated.yaml`* | *GPU + base ckpt* |
| *4b* | *`scripts.finetune.fulltext`* | *Legacy: fine-tune on in-context corrupted sentences* | *`phase4b_fulltext.yaml`* | *GPU + 4a ckpt* |
| *4c* | *`scripts.finetune.conversational`* | *Legacy: conversational adaptation on BERnaT BSM* | *`phase4c_conversational.yaml`* | *GPU + 4b ckpt + BSM* |
| **5** | `scripts.package.to_gguf` | Convert HF checkpoint → GGUF + patch FUTO `keyboardlm.*` metadata | `phase5_package.yaml` | finetune ckpt |
| **5** | `scripts.package.patch_metadata` | (called by 5) Write `keyboardlm.*` fields into the GGUF | — | — |
| **5** | `scripts.package.downgrade_v2` | Downgrade GGUF v3→v2 + strip fields the app's llama.cpp doesn't understand | — | GGUF |
| **5d** | `scripts.package.build_wordlist` | Build `eu_wordlist.combined.gz` — stream Latxa v2 → count → hunspell-validate → AOSP combined format | — | network |
| **5d** | `scripts.package.compile_dict.sh` | Compile `eu_wordlist.combined.gz` → binary `eu.dict` (AOSP dicttool, v2/202) for side-loading | — | java |
| **eval** | `scripts.eval.keyboard` | Autocorrect + next-word accuracy on a hand-curated Basque test set | — | **GPU** + ckpt |
| **eval** | `scripts.eval.keystrokes` | **Keystrokes-saved** on realistic messaging messages (measures real keyboard utility) | — | GGUF + tokenizer |

### Library helpers (`scripts/lib/`)

| File | Purpose |
|------|---------|
| `runconfig.py` | YAML run-config loader with mini/full mode support (CLI > config > default) |
| `typo_synthesis.py` | Generate plausible typos (QWERTY adjacency, ñ-loss, transposition, doubling, shortcuts) → `<XBU>…<XBC>…<XEC>` format |
| `progress.py` | Compact training-progress logging callback |
| `datasets.py` | `MultiTaskFinetuneDataset` — interleaves plain text + isolated triples with loss weighting |
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

# 6–9. Pretrain + finetune ON A GPU HOST (all params from configs/):
#   uv run python -m scripts.pretrain.train \
#       --config configs/phase3_pretrain.yaml --mode full \
#       --tokenizer tokenizer/spm_eu.model --corpus corpora/clean --out pretrain
#   uv run python -m scripts.finetune.build_wordfreq --corpus corpora/clean --out notes/wordfreq.json
#   uv run python -m scripts.finetune.generate_triples --config configs/phase4a_dataprep.yaml --mode full
#   uv run python -m scripts.finetune.multitask \
#       --config configs/phase4_multitask.yaml --mode full \
#       --base pretrain/base --tokenizer tokenizer/spm_eu.model \
#       --corpus corpora/clean --synth-jsonl notes/synth.json --real-jsonl notes/real.json

# 10. Package → GGUF (converts, patches metadata, downgrades v3→v2 automatically)
uv run python -m scripts.package.to_gguf \
    --config configs/phase5_package.yaml --mode full \
    --checkpoint finetune/stage_m/final --tokenizer tokenizer/spm_eu.model \
    --llama-cpp /path/to/llama.cpp
```

Transfer the final `.gguf` to your phone and side-load it via **FUTO Keyboard →
Settings → Languages & Models → Import model**.

> **Server runbook:** for GPU-host execution, use `./run_server.sh full 1 1b 2 3 4a 4m 5`
which orchestrates all phases with the correct dependency groups. See the header
of [`run_server.sh`](./run_server.sh) for all options (mini/full modes, individual
phases, `4mr` recover mode).

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
4K MorphAcc regime. The 2048 context window easily absorbs the 39% fertility
increase (2.58 vs 1.85 tokens/word).

The FUTO app reads vocab size dynamically (`llama_n_vocab()`), so this is fully
compatible — no hardcoding. The tokenizer training script includes a MorphAcc
spot-check that warns if splitting degrades.

---

## Basque-specific adaptation notes

- **Corpus** (two tiers — see RESEARCH.md §11.4, §11.6 for the full rationale):
  - *Clean* (`corpora/clean/`): Morpheus's cleaned [Latxa corpus v2](https://huggingface.co/datasets/HiTZ/latxa-corpus-v2) — 11 HiTZ-curated, deduplicated sources (~4.77 B tokens, LLM-audited avg quality 4.6/5). Used for **tokenizer training + pretrain base + Phase 4m plain-text stream**. Target: **3B tokens staged** (not 5B — our 25M model's sweet spot is 80-120:1 ratio, §11.6).
  - *Conversational* (`corpora/conversational/`): [BERnaT BSMtime](https://huggingface.co/datasets/HiTZ/BERnaT-Diverse) social-media posts (~250 M tokens), aggressively cleaned (emoji/URL/mention/code-switch stripping). **Legacy Phase 4c only** — excluded from tokenizer and pretrain per the revised strategy (§11.6). FUTO's own English pipeline uses SlimPajama (web) for pretrain + a small conversational finetune at the end; the legacy 4c stage follows the same pattern. The BERnaT paper (Azurmendi et al. 2025) shows diverse data helps without hurting standard-form accuracy.
- **Pretrain objective fix (critical)**: the original pretrain script had a
  **double causal-shift bug** — `input_ids=ids[:-1]`, `labels=ids[1:]` caused
  HF Trainer to shift again internally, so the model learned skip-1 prediction
  `P(token[i+2] | token[i])` instead of next-token. Fixed in `a377081` to
  `input_ids=ids`, `labels=ids`. See `scripts/pretrain/diag_objective.py` for
  the diagnostic that catches this regression.
- **Phase 4m (unified multi-task finetune)**: replaces the old sequential 4a→4b→4c
  pipeline, which caused 100% format contamination (the model learned control
  tokens as literal text). The unified approach interleaves 60% plain text
  (PLW=1.0, next-word prediction) with 40% isolated `<XBU>…<XBC>…<XEC>` triples
  (correction-only loss), strictly segregated at the sequence level. This keeps
  the model's plain-text fluency while teaching the autocorrect format.
- **Typo synthesis**: QWERTY (not ABNT2) adjacency; the Portuguese accent-swap
  rule (`é→ê`) is removed (Basque has no such accents); ñ→n handled by NFD.
  Adjacency + transposition/doubling are the dominant Basque typo classes.
- **Tokenizer slots**: 146 + 56 high-frequency Basque words (function words,
  pronouns, auxiliaries, common verbs/nouns/adjectives). Padded with `<FUTO>`
  filler to the exact slot count. The structural slots are untouched.

---

## Basque dictionary (autocorrect candidate engine)

FUTO's hybrid autocorrect works in two halves: a **classical dictionary engine**
proposes real-word candidates (is *kaixo* a word? *kaixp* is not), and the
**transformer** re-ranks them by contextual probability. The model ships as the
transformer half; the dictionary half was missing for Basque.

Two deliverables now close that gap:

| File | Format | Role |
|------|--------|------|
| `dictionaries/eu_wordlist.combined.gz` | AOSP *combined* text (source) | What FUTO compiles into the app at build time; human-readable; contributes upstream |
| `dictionaries/eu.dict` | AOSP v2 binary (magic `0x9bc13afe`, ver 202) | What the app's import UI accepts today for side-loading |

**Build pipeline** (`scripts/package/build_wordlist.py`):
1. Stream Latxa v2 from HF (wikipedia + euscrawl-v2 + zelaihandi, 600k lines)
2. **Two tracks**: common words (lowercased, min-count ≥3, **hunspell eu_ES**
   validated — rejects gibberish, accepts correctly inflected forms like
   `etxea`/`etxera`/`etxetik` that affix expansion would miss) + a **proper-noun
   track** (capitalized tokens / acronyms selected by a capitalization-ratio
   heuristic: a token is a name if it's *usually* capitalized, which excludes
   sentence-initial common words). This captures the place names, person names
   and acronyms — `Bilbo`, `Euskal`, `Gipuzkoako`, `AEB` — that hunspell rejects.
3. **Bigrams**: top-80k adjacent word pairs emitted as AOSP bigrams
   (`  bigram=<w>,f=<f>` under the unigram) for contextual next-word ranking.
4. Inject must-include words from `config/eu.py` (autocorrect test targets, etc.)
5. Map corpus counts → AOSP log-scale frequency `f` ∈ [1,255] (255 = prob 1);
   `compile_dict.sh` compiles the `.combined.gz` → binary `.dict`

**Result**: 791,021 unigrams + 80,000 bigrams, 4.0 MB gzipped / 5.6 MB binary.
f range 147–255 (graduated — only 1 word clamped at f=255). Top words are clean
Basque function words (`eta, da, ez, ere, bat, du, izan, dira, egin, behar`);
top bigrams are real collocations (`ez da` "is not", `izango da` "will be",
`eskerrik asko` "thank you", `egin behar` "must do"). All 40 autocorrect test
targets from `config/eu.py` are present. Rebuild:

```bash
uv run python -m scripts.package.build_wordlist \
  --lines-per-source 200000 --max-words 0 --max-bigrams 80000 \
  --save-freq notes/wordfreq_latxa.json
./scripts/package/compile_dict.sh
```

**Side-load**: copy `dictionaries/eu.dict` to your phone and open it (or import
via Settings → Languages & Models). The app detects the `0x9bc13afe` magic +
`locale=eu` header and registers it as the Basque main dictionary
(`DictionaryFactory.tryOpeningCustomMainDictionaryForLocale`). See RESEARCH.md
§11.2 for the import path.

### Comparison with FUTO's referenced Basque dictionary

FUTO's dictionaries page (`keyboard.futo.tech/dictionaries?locale=eu-ES`) does
**not** ship its own Basque dictionary — it links to Helium314's community AOSP
dict (`main_eu.dict` on Codeberg). Ours beats it on every field:

| field | FUTO/H314 `main_eu` | Ours |
|-------|--------------------|------|
| unigrams | 106,786 | **791,021** (7.4×) |
| bigrams | 0 | **80,000** |
| autocorrect targets | 39/40 | **40/40** |
| proper nouns | 7,986 | **338,308** |
| `da` rank (#1 Basque word) | f=109, 5,312 above | **f=248, 1 above** |
| words clamped at f=255 | 31 | **1** |

The frequency-quality gap is the most consequential: H314's frequencies are
saturated (top-15 all clamped at f=255, and `da` — the #1 Basque word — ranks
behind 5,312 others), so the keyboard can't order candidates. Ours graduates
(`eta`=255, `da`=248, `ez`=245, …) from real Latxa corpus counts. Reproduce the
comparison with `scripts/package/compare_full.py`.

---

## Status

- [x] Project scaffold + config + YAML run-configs (`configs/*.yaml`)
- [x] All scripts ported from `futo-portuguese`, adapted for Basque
- [x] Phase 0: download reference model, dump metadata, verify slot layout
- [x] Corpus strategy: Latxa v2 (clean) + BERnaT BSM (conversational, legacy 4c)
- [x] Mini validation: full pipeline (1→5) runs end-to-end, model works in FUTO app
- [x] Full pretrain: 24,000 steps (10h) on 3B tokens from Latxa v2 clean tier
- [x] **Pretrain bug fix** (`a377081`): resolved double causal-shift — root cause of
      broken next-word prediction. Diagnostic confirmed: skip-1 loss 3.8 < next-token 7.6.
- [x] **Unified multi-task finetune** (4m): 18k steps, 60% plain + 40% triples
- [x] **Diagnostic tooling**: objective diagnostic (`diag_objective.py`),
      next-word eval (`nextword_pretrain.py`), loss diagnostic (`diag_4m_loss.py`)
- [x] **v2.0.0 released**: 50% next-word top-1, 0% contamination
- [x] **Basque dictionary** (`dictionaries/eu_wordlist.combined.gz` + `eu.dict`) for
      FUTO's hybrid dictionary engine — 791k unigrams + 80k bigrams from 600k
      Latxa v2 lines, hunspell-validated + proper-noun track, beats FUTO's
      referenced (Helium314) dict on every field (7.4× words, graduated
      frequencies, bigrams, 40/40 autocorrect targets)
- [ ] Increase triple ratio in 4m to improve FUTO-format autocorrect (currently 0% standalone)
