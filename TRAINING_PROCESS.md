# Training a Transformer Language Model for Basque Keyboard Prediction

> **Purpose of this document.** This is a self-contained technical brief describing the
> full training pipeline for a Basque (euskara) text-prediction model targeting the
> **FUTO Keyboard** Android app. It is written for review by a senior ML engineer. It
> explains the objective, the target system's constraints, every major design decision
> and its rationale, the data strategy, and the training recipe. No internal codebase
> knowledge is required to read it.

---

## 1. Objective

### 1.1 What we are building

A small (~25M parameter) **Llama-architecture transformer language model** that runs
**on-device** on Android phones inside the FUTO Keyboard app, providing two capabilities
for Basque:

1. **Next-word prediction** — suggest the most likely next word given the text typed so far.
2. **Autocorrect** — correct a partially- or incorrectly-typed word in real time, given
   surrounding context and the sequence of physical key presses.

The model is distributed as a **GGUF** file (llama.cpp format) with FUTO-specific
metadata, and is side-loaded into the app via its "Import from file" mechanism.

### 1.2 Why this matters

FUTO Keyboard is a privacy-focused, offline Android keyboard (forked from AOSP's
LatinIME). It ships an English transformer model but does **not** yet ship models for
other languages. Basque has **no dictionary** in FUTO at all — only a keyboard layout.
This model would be the **primary prediction engine** for Basque users.

There is a community precedent: a complete end-to-end pipeline exists for Brazilian
Portuguese, which we use as a structural template (the FUTO format is language-agnostic
for any language using the Latin-26 alphabet).

### 1.3 The target deployment

- **On-device inference** via llama.cpp, hard-capped at **2048-token context**.
- **Latency-critical**: predictions must appear as the user types; the model is tiny
  (~25M params) to keep inference fast on mobile hardware.
- **No server**: everything runs offline. The model weights are the entire system.

---

## 2. The Target System: FUTO's Keyboard LM Format

The model is **not** "any GGUF model." FUTO defines an undocumented but reverse-engineered
format with strict requirements. Understanding these constraints is essential because they
shape every downstream decision.

### 2.1 Model architecture

The reference English model (~36M params) is a vanilla Llama:

```python
LlamaConfig(
    vocab_size               = 15008,   # reference; ours differs (see §4.2)
    hidden_size              = 512,
    intermediate_size        = 1024,
    num_hidden_layers        = 8,
    num_attention_heads      = 8,
    num_key_value_heads      = 8,       # MHA, no GQA
    max_position_embeddings  = 2048,
    rms_norm_eps             = 1e-6,    # NOT the HF default 1e-5
    rope_theta               = 10000.0,
    tie_word_embeddings      = False,
)
```

Our model matches this **exactly**, except for `vocab_size` (we use 4096 — see §4.2). This
gives ~25M parameters. We deliberately keep the architecture identical to the reference so
that the FUTO C++ inference path (which has hard-coded assumptions about Llama internals)
works without modification.

### 2.2 The keypress autocorrect format (the central design constraint)

This is the most novel and least-documented part of the system. Autocorrect is **not** done
by feeding the literal typed text. Instead, each physical key press becomes a **discrete
`<CHAR_X>` token**, and the model is trained to emit the corrected word as plain text
between two structural tokens:

```
[context] <XBU> <CHAR_T><CHAR_E><CHAR_H> <XBC> The <XEC>
                   ^^^^^^^^^^^^^^^^^^^              ^^^^^^
                   typed keypresses                 corrected word
```

The structural tokens are:
- `<XBU>` — "begin user input" (marks the start of the typed keypress sequence)
- `<XBC>` — "begin correction" (the model starts emitting the corrected word here)
- `<XEC>` — "end correction"
- `<CHAR_A>` … `<CHAR_Z>` — one token per physical key (26 tokens, **must** occupy 26
  contiguous, sequential token IDs — the C++ does pointer arithmetic: `CHAR_A + i`)

**Critical implication for diacritics:** The keyboard emits one `<CHAR_X>` token per
physical key press. Diacritics are **not** separate tokens. A long-press `á` emits only
`<CHAR_A>`; the accent is dropped from the prompt. The model must **reconstruct** the
correctly-accented form from context + base letters. The robust conversion normalizes via
NFD and strips combining marks:

```python
import unicodedata

def to_keypress_chars(typed: str) -> list[str]:
    out = []
    for ch in typed:
        decomposed = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        for c in base.upper():
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out
```

This means **only languages using the Latin-26 alphabet are supported without patching
the Android app itself.** Basque (ñ, á, é, í, ó, ú, ü) is fine — all diacritics decompose
to base letters. The model's job is to predict the correct accented surface form.

### 2.3 The embedded tokenizer

The GGUF file carries the SentencePiece tokenizer as a raw byte blob in a custom metadata
field (`keyboardlm.ext_tokenizer_data`, typed as `[UINT8]`). FUTO ignores llama.cpp's
standard `tokenizer.ggml.tokens` array entirely.

The tokenizer is a **SentencePiece UNIGRAM** model with a fixed-layout vocabulary. The
critical structural slots (verified against the reference) are:

| Token IDs | Contents |
|-----------|----------|
| 0–3 | `<pad>`, `<s>`, `</s>`, `<unk>` |
| 4–303 | 300 user-defined symbols (fixed structural layout — see below) |
| 304–559 | 256 byte-fallback tokens (`<0x00>`…`<0xFF>`, auto-generated) |
| 560+ | Learned UNIGRAM pieces from corpus |

Within the 300 user-defined symbols, there is a **structured layout** that must be preserved
exactly:

```
IDs 4–27     (24)  reserved filler slots
IDs 28–173  (146)  content slots → high-frequency language-specific words
ID  174       (1)  <XBU>   (looked up by name)
ID  175       (1)  <XBC>   (looked up by name)
ID  176       (1)  <XEC>   (looked up by name)
IDs 177–181   (5)  <XC0>–<XC4>  (swipe mode; only <XC0> is referenced)
IDs 182–207  (26)  <CHAR_A>–<CHAR_Z>  (MUST be 26 contiguous sequential IDs)
IDs 208–263  (56)  more content slots → high-frequency language-specific words
IDs 264–303  (40)  emoji slots
```

The structural tokens (`<XBU>`, `<XBC>`, `<XEC>`, `<XC0>`, `<CHAR_A>`) are looked up **by
name** at load time. If any resolves to `0` (the `<unk>` fallback), the app crashes. The
`<CHAR_A>`–`<CHAR_Z>` range is accessed by computed index (`CHAR_A + i`), so it must be
contiguous.

> **Note on an app bug we discovered:** The C++ uses `ASSERT(id != 0)` to validate
> structural tokens, but SentencePiece's `PieceToId()` returns the `unk_id` (=3), not 0,
> for missing pieces. So a *missing* structural token silently passes the assert and
> surfaces as a confusing runtime failure rather than a clean load-time error. This makes
> correct tokenizer layout verification essential — we cannot rely on the app to catch it.

### 2.4 Feature flags

The model declares a space-separated feature string in metadata. The app understands:

| Feature | Meaning |
|---------|---------|
| `base_v1` | basic next-word LM |
| `inverted_space` | tokenizer trained with `treat_whitespace_as_suffix=true` (space attaches to preceding token) |
| `xbu_char_autocorrect_v1` | enables the `<XBU><CHAR_*><XBC>…<XEC>` autocorrect path |
| `char_embed_mixing_v1` | **required** if `xbu_char_autocorrect_v1` is set — the C++ mixes `<CHAR_*>` embeddings into the prompt; without it, every keystroke SIGSEGVs |
| `xc0_swipe_typing_v1` | ML swipe decoding (needs extra encoder tensors) |
| `lora_finetunable_v1` | on-device LoRA personalization |

**Our feature set:** `base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1`

We omit `xc0_swipe_typing_v1` (no swipe training data; the `<XC0>`–`<XC4>` tokens remain in
the vocab but are unused; the app falls back to dictionary-based swipe).

### 2.5 GGUF version constraint

The app requires **GGUF version 2**, not the current v3. The packaging step downgrades the
GGUF header after conversion.

---

## 3. Why Basque Is Challenging

### 3.1 Agglutinative morphology

Basque is a language isolate with **agglutinative morphology** — it stacks suffixes onto
roots to express case, number, and grammatical function. A single root like *etxe* (house)
generates a large paradigm:

```
etxe     (root: house)
etxea    (the house, absolutive)
etxera   (to the house, allative)
etxetik  (from the house, ablative)
etxekoak (the ones from the house, genitive plural)
etxeetara (to the houses, plural allative)
```

This has two consequences for a tokenizer:

1. **Vocabulary explosion.** A large vocab will memorize common inflected forms as single
   tokens, hiding morpheme boundaries from the model. The model then cannot generalize
   across the paradigm (it never sees `-ra` or `-tik` as independent units).
2. **Fertility.** A small vocab forces morpheme-level splitting, producing more tokens per
   word — but the model *sees* the morphemes and can generalize.

### 3.2 Low-resource

Basque has ~750K native speakers. Available corpora are small compared to English. The
largest cleaned corpus (Latxa v2, from HiTZ) is ~4.77B tokens — a fraction of what English
models train on.

### 3.3 Dialectal variation

Basque has significant dialectal diversity, plus a standardized register (*batua*) used in
education and media. A phone keyboard must handle both — users type in their dialect but
expect autocorrect toward standard forms.

---

## 4. Tokenizer Design

### 4.1 UNIGRAM, not BPE

We use **SentencePiece UNIGRAM** (matching the reference English model). Rationale:

- UNIGRAM's probabilistic segmentation handles ambiguous morpheme boundaries better than
  BPE's deterministic greedy longest-match — important for agglutinative Basque where
  suffix boundaries can be ambiguous.
- Matching the reference model's `model_type` is the safe choice for format compatibility.

**SentencePiece parameters (all verified against the reference):**

| Parameter | Value | Why |
|-----------|-------|-----|
| `model_type` | `unigram` | Reference uses UNIGRAM; better for agglutinative morphology |
| `character_coverage` | `0.9995` | Matches reference; covers 99.95% of characters |
| `byte_fallback` | `true` | 256 byte-fallback tokens guarantee no OOV |
| `treat_whitespace_as_suffix` | `true` | Required by the `inverted_space` feature flag |
| `add_dummy_prefix` | `false` | Matches reference; a dummy `▁` before `<XBC>`word would corrupt the autocorrect format |
| `remove_extra_whitespaces` | `false` | Matches reference; preserve original whitespace |
| `pad/bos/eos/unk` | `0/1/2/3` | Matches reference |
| `input_sentence_size` | `10,000,000` | Sample 10M sentences from the 3B-token corpus for training |

### 4.2 Vocabulary size: 4096 (not the reference's 15008)

**This is our most significant deviation from the reference.** The reference English model
uses vocab=15008. We use **4096**. The decision is based on a controlled ablation:

#### The fertility paradox

We ran a tokenizer ablation (21 Basque test words across 5 roots, identical SentencePiece
params except vocab size, on a 336MB corpus sample), measuring **MorphAcc** — does the
tokenizer split at the root-suffix boundary?

| Vocab | MorphAcc | Fertility (tokens/word) | Example: `etxetik` |
|-------|----------|-------------------------|---------------------|
| 4K | **66.7%** (14/21) | 2.58 | `▁etxe` + `tik` ✅ |
| 8K | 61.9% (13/21) | 2.28 | `▁etxe` + `tik` ✅ (inconsistent across paradigm) |
| 16K | 52.4% (11/21) | 2.06 | `▁etxetik` ❌ whole word |
| 32K | 28.6% (6/21) | 1.85 | `▁etxetik` ❌ whole word |

At 32K, common inflected forms are single tokens — the model never sees the morphemes, so
it can't generalize across the paradigm. At 4K, the tokenizer is *forced* to split,
consistently exposing morpheme boundaries.

#### Why this transfers to FUTO

1. **Smaller model benefits more.** Our 25M model has less capacity to waste on surface-form
   memorization than the reference or larger models — morpheme splitting helps more.
2. **Context absorbs fertility.** FUTO's 2048-token context easily absorbs the 39% fertility
   increase (2.58 vs 1.85 tokens/word). ~794 words fit in context.
3. **Dynamic vocab.** The FUTO app reads vocab size dynamically (`llama_n_vocab()` /
   `spm.GetPieceSize()`) — verified in the C++ source. There is no hardcoded 15008 anywhere.
4. **No warm-start from English.** Warm-starting from the English reference is a dead
   argument regardless of vocab size: the learned pieces differ between English and Basque
   corpora (different SentencePiece training). Only structural + byte-fallback tokens are
   shared, and those carry no semantics.

**Reservation we hold:** The MorphAcc finding is tokenizer-level (architecture-independent).
The downstream-quality evidence for "smaller vocab is better" comes from a sibling project
that trained a full 91M Mamba2 model with the 4K tokenizer (converged, PPL 7.13) and from
the QuechuaTok finding (different language family). We did **not** train full models at
8K/16K/32K for Basque to get end-to-end quality comparisons. The parameter-efficiency and
fertility arguments are strong, but this is the decision most worth scrutinizing.

### 4.3 Structural slot layout

We preserve the exact reference layout (verified token-by-token against the reference
model's extracted SentencePiece). The content slots (146 + 56 = 202 slots) are filled with
high-frequency Basque words (pronouns, common verbs, function words, adjectives) and 40
emoji matching the reference set. The structural tokens (`<XBU>`, `<XBC>`, `<XEC>`,
`<XC0-4>`, `<CHAR_A-Z>`) occupy identical IDs to the reference.

With vocab=4096 and 560 reserved slots (4 control + 300 structural + 256 byte-fallback),
we have **3536 learned UNIGRAM pieces** — within 5% of the 4K regime from the ablation.

---

## 5. Corpus Strategy

### 5.1 Two-tier approach

We use a **two-tier corpus strategy**, with a strict scoping rule about which tier touches
which training phase:

| Tier | Source | Tokens | Used in |
|------|--------|--------|---------|
| **Clean** | Latxa corpus v2 (11 HiTZ-curated sources) | 3.0B staged (of 4.77B available) | Tokenizer, pretrain, autocorrect finetune |
| **Conversational** | BERnaT BSM (Basque Social Media) | 250M | Conversational adaptation finetune **only** |

**The critical scoping rule: BSM touches Phase 4c (conversational finetune) ONLY. It never
enters the tokenizer training or the pretrain base.**

### 5.2 Clean tier: Latxa corpus v2

The clean tier reuses the **Latxa corpus v2** from HiTZ (the same group that built the
BERnaT corpus). We reuse a sibling project's already-cleaned version (`clean-v3`) — no
re-cleaning, since undoing rigorous deduplication and quality filtering would degrade
quality.

The 11 sources, with quality assessment:

| Source | Quality | Description |
|--------|---------|-------------|
| euscrawl-v2 | 5.0 | News/media crawl, best source (56% of Latxa v1) |
| parleus | 4.9 | Parliament transcriptions |
| zelaihandi | 4.9 | Curated diverse corpus |
| bopv | 4.8 | Basque Government official gazette |
| botha | 4.8 | Álava provincial gazette |
| colossal-oscar | 4.7 | Cleaned Common Crawl |
| wikipedia | 4.6 | Basque Wikipedia dump |
| cultura-x | 4.6 | Cleaned web (CulturaX) |
| hplt-v2 | 4.4 | HPLT v2 crawl |
| fineweb2 | 4.3 | FineWeb2 |
| finepdfs | 3.7 | FinePDFs (digit-filtered) |

Three sources were **excluded**: hplt-v1 (superseded by v2), bog (quality concerns),
aldizkariak (magazine crawl with OCR noise). This mirrors the sibling project's audit.

### 5.3 Conversational tier: BERnaT BSM

**BERnaT-Diverse** (Azurmendi et al. 2025, arXiv:2512.03903) is a published corpus from
HiTZ containing ~11M Basque social-media posts by ~13K users (~188M words). We use the
**BSMtime** configuration (individual posts in chronological order, 34–280 chars each).

#### Why we re-include BSM (contradicting the sibling project's assumption)

The sibling project excluded BSM based on an untested assumption: that social-media text
would "bias a morphology-targeted tokenizer toward informal patterns at the expense of the
formal Basque needed for good morphological segmentation."

But the BERnaT paper ran the controlled experiment and found the **opposite**:

> "Models trained on both standard and diverse data consistently outperform those trained
> on standard corpora, improving performance across all task types **without compromising
> standard benchmark accuracy**."

**Why this matters for FUTO specifically:**

1. **Distribution match.** FUTO is a phone keyboard. Users type chat messages, not
   Wikipedia articles. BSM is the closest deployment-distribution data that exists for Basque.
2. **The two FUTO tasks split cleanly.** Autocorrect wants standard *batua* as the
   correction target (handled by finetune with curated targets — BSM never touches this).
   Next-word prediction wants conversational fluency — BSM is exactly this.
3. **Scale is safe.** BSM is ~250M tokens. Against a 3B Latxa base, that's ~8% — enough to
   shift the model toward casual phrasing, not enough to corrupt it.

#### Why BSM is in the finetune only, NOT the pretrain

We apply the sibling project's *valid concern* (dialect/argot polluting the vocabulary)
to the **right phase**:

| Phase | Sees BSM? | Why |
|-------|-----------|-----|
| Tokenizer training | ❌ No | Keep UNIGRAM vocabulary morpheme-focused on standard batua. BSM slang could pollute the piece inventory. |
| Pretraining | ❌ No | Pretrain on diverse, clean, standard Basque (Latxa). BSM noise/dialect would pollute the base LM. This matches FUTO's own pipeline (SlimPajama web pretrain → conversational finetune at the end). |
| Autocorrect finetune | ❌ No | Correction targets must be standard batua forms. |
| Conversational finetune (4c) | ✅ Yes | The model learns conversational register using the *clean* vocabulary. This is exactly what BERnaT did, and what FUTO's own wiki describes. |

This gives us the conversational fluency (BERnaT's finding) without risking the
morpheme-aligned vocabulary (the sibling project's valid concern, correctly scoped).

#### BSM cleaning pipeline

BSM is aggressively cleaned before use:

- **Strip:** emoji (Unicode ranges), URLs, @mentions, hashtag symbols (keep the word),
  RT/via attribution prefixes, HTML entities.
- **Filter:** code-switched lines (Spanish function-word ratio > 0.15), too-short lines
  (< 3 words after stripping), pure-punctuation/emoji lines, exact duplicates.
- **Keep:** dialectal Basque spelling, informal grammar/word order, slang — this is the
  value, not noise.

Code-switching detection uses a Spanish function-word ratio (Basque and Spanish share
heavy code-switching due to geography):

```python
SPANISH_FUNCTION_WORDS = {"el", "la", "los", "las", "de", "que", "en", "y", "a",
                          "un", "una", "es", "por", "con", "no", "se", "del", ...}

def spanish_ratio(text: str) -> float:
    words = [w.lower() for w in text.split()]
    if not words:
        return 0.0
    es_count = sum(1 for w in words if w in SPANISH_FUNCTION_WORDS)
    return es_count / len(words)

def is_strictly_eu(text: str, max_es_ratio: float = 0.15) -> bool:
    # Reject all-Spanish AND code-switched lines, keep dialectal Basque
    if not is_likely_eu(text):  # quick all-Spanish check
        return False
    return spanish_ratio(text) <= max_es_ratio
```

**Two BSM gotchas handled:**
1. **BSMauthor vs BSMtime are the same text, reordered** (by-author vs chronological).
   Using both = 2× duplication for an autoregressive LM. We use BSMtime only (individual
   posts, not 91KB timeline blobs).
2. **EKC (historical texts) excluded.** BERnaT-Diverse also includes EKC: 338 classical
   Basque texts from the 16th–19th century, pre-standardization. For a modern phone
   keyboard, 400-year-old Basque is wrong-distribution noise.

Verified cleaning result: 17% drop rate (e.g., 900K posts kept, 185K dropped), avg 112
chars/line, 0 emoji leaking, 0.005% URL residue.

### 5.4 Data scale: 3B tokens for pretrain (not 5B)

We revised the pretrain target from 5B → **3B tokens** based on scaling-law analysis:

| Reference point | Model size | Tokens | Ratio (tokens:param) |
|-----------------|-----------|--------|----------------------|
| Chinchilla optimal | 25M | 500M | 20:1 |
| Sibling project (converged) | 91M | 8.8B | 97:1 |
| MiniCPM / Mosaic sweet spot | — | — | 80–200:1 |
| **Our target** | **25M** | **~6B seen** (3B staged × ~2 epochs) | **~250:1** |

The 3B staged × ~2 epochs of pretraining = ~6B tokens-seen ≈ 250:1 ratio. This is on the
higher end, but: (a) the sibling project's 91M model was still improving at 97:1, and
smaller models can benefit from more passes; (b) we keep 5 checkpoints and will pick the
best by loss curve if it plateaus early.

We explicitly chose **not** to go to 5B because: (a) diminishing returns past
~100–200:1 for a model this small; (b) the mini validation (525M tokens, 21:1) already
produced plausible predictions, suggesting the data quality matters more than raw volume;
(c) faster iteration.

---

## 6. Training Pipeline

The pipeline has 6 stages. Each builds on the previous checkpoint. All training uses mixed
precision (bf16), AdamW with cosine schedule, and logs to Weights & Biases.

### Phase 1: Stage clean corpus → `corpora/clean/`

Stage 3B tokens from the Latxa clean-v3 sources into 256MB plain-text shards. No
re-cleaning (data is already deduplicated and quality-filtered). Token budget is hit
mid-way through the 11th source alphabetically — all sources contribute.

### Phase 1b: Clean + stage BSM → `corpora/conversational/`

Stream BSMtime from HuggingFace, apply the cleaning pipeline (§5.3), stage 250M tokens.
Output is used by Phase 4c only.

### Phase 2: Train tokenizer (clean tier only)

Train the SentencePiece UNIGRAM tokenizer on the **clean corpus only** (never BSM).
Validate:
- All structural tokens (`<XBU>`, `<XBC>`, `<XEC>`, `<XC0-4>`, `<CHAR_A-Z>`) present at
  correct IDs (verified against reference token-by-token).
- `<CHAR_A>`–`<CHAR_Z>` are 26 contiguous sequential IDs.
- MorphAcc spot-check: `etxetik` → `▁etxe` + `tik`? (warns if splitting degrades below 60%).
- XBU format round-trips: `<XBU><CHAR_H><CHAR_I><XBC>kaixo <XEC>` tokenizes correctly.
- `inverted_space` convention: space attaches to preceding token.

### Phase 3: Pretrain base model (clean tier only)

**This is the dominant phase (~8–10h on an L40 GPU).**

| Parameter | Value |
|-----------|-------|
| Corpus | Clean tier only (3B tokens, ~2 epochs) |
| Total steps | 24,000 |
| Seq length | 1024 |
| Global batch | 256 (micro 16 × grad-accum 16) |
| Tokens/step | ~262K |
| Total tokens seen | ~6.3B |
| Learning rate | 3.0e-4 |
| Weight decay | 0.1 |
| Warmup | 2,000 steps (~8%) |
| Save every | 5,000 steps (keep last 5) |

**Output:** Llama base checkpoint (~25M params, 97MB fp32).

The model architecture is read from config (hidden=512, ffn=1024, 8 layers, 8 heads, MHA,
context=2048, rms_norm_eps=1e-6, rope_theta=10000, untied embeddings). Vocab size is read
dynamically from the tokenizer (4096), not hardcoded.

### Phase 4a: Isolated autocorrect finetune

Teach the model the `<XBU><CHAR_*><XBC>correction<XEC>` format using synthetic typo→correct
pairs. This is **format learning** — the model learns to map keypress sequences to corrected
words in isolation.

**Data:** 500K synthetic typo→correct pairs (full mode; 200K in mini), generated from a
200K-word frequency list built from the clean corpus, log-frequency-weighted. Plus ~54
real shortcut pairs (curated Basque chat abbreviations). Mix: 75% synthetic, 25% real.

**Typo synthesis** generates realistic errors using four operations weighted by what Basque
typists actually do:

1. **Keyboard-adjacency substitution** (dominant class) — uses a QWERTY adjacency map:
   ```python
   ADJ = {
       'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', ...
       'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', ...
   }
   # e.g., "kaixo" → "kaixp" (o→p, adjacent on QWERTY)
   ```
2. **Diacritic loss** — ñ→n, á→a, é→e (NFD decomposition + strip combining marks). This is
   the main Basque diacritic typo class.
3. **Letter transposition** — adjacent letters swapped ("gaur" → "garu").
4. **Doubling / deletion / insertion** — common motor errors ("eskerrik" → "eskkerrik").

**Training:**

| Parameter | Value |
|-----------|-------|
| Total steps | 20,000 |
| Seq length | 64 (triples are short) |
| Global batch | 256 (micro 64 × grad-accum 4) |
| Learning rate | 1.0e-4 |
| PLW (prompt loss weight) | 0.0 (only the correction span contributes to loss) |

**Output:** Stage A checkpoint (finetune/stage_a/final/).

### Phase 4b: In-context autocorrect finetune

Take clean corpus sentences, inject ~33% of words as typo-correction triples inline, and
train full-sequence loss so the model learns autocorrect **in context** (not just isolated
words).

The training data looks like:

```
Egun on, <XBU><CHAR_K><CHAR_A><CHAR_I><CHAR_X><CHAR_P><XBC>kaixo <XEC> zer moduz?
```

The model must correct `kaixp` → `kaixo` while also predicting the surrounding clean text.

| Parameter | Value |
|-----------|-------|
| Corpus | Clean tier (Latxa) |
| Total steps | 15,000 |
| Seq length | 512 |
| Global batch | 192 (micro 24 × grad-accum 8) |
| Learning rate | 5.0e-5 |
| Typo rate | 0.33 (1/3 of words corrupted) |
| PLW | 0.05 (preserve pretrain next-word ability; see below) |

**On PLW:** We use `plw=0.05` for Phase 4b (proactive mitigation, see §9.2). The loss
weights tokens inside `<XBU>…<XEC>` spans at 1.0 (full weight on autocorrect learning) and
clean context at 0.05. Since loss is computed on shifted tokens, this means the model barely
trains on clean→clean transitions (next-word prediction, already learned in pretrain) and
clean→`<XBU>` transitions (the source of format overfit). The mini validation's 0% next-word
accuracy with plw=1.0 showed exactly this failure mode — the model learned to emit structural
tokens too aggressively. Phase 4c keeps plw=1.0 because its goal is register shift, which
requires training on clean BSM context.

**Output:** Stage B checkpoint (finetune/stage_b/final/).

### Phase 4c: Conversational adaptation (BSM)

**This stage is taken directly from FUTO's own English training pipeline.** FUTO's wiki
describes it verbatim:

> "The model is finetuned on a much smaller corpus representing Internet first-person speech
> with the same 1/3 misspelling augmentation. This has appeared to be an important step for
> the model to comprehend that sentences can start with 'I'm', certain slang/lingo, etc."

Without this stage, the model suggests Wikipedia-register continuations ("Furthermore, …")
instead of chat-register ("yeah, …").

| Parameter | Value |
|-----------|-------|
| Corpus | Conversational tier (BERnaT BSM, 250M tokens) |
| Total steps | 5,000 |
| Seq length | 512 |
| Global batch | 192 (micro 24 × grad-accum 8) |
| Learning rate | 2.0e-5 (lower than 4b — adaptation, not retraining) |
| Typo rate | 0.33 (same as 4b, per FUTO wiki: "the same") |
| PLW | 1.0 |

At 192 batch × 512 seq = 98K tokens/step, 5000 steps ≈ 490M tokens-seen ≈ ~2 epochs of
the 250M BSM. This is intentionally short — "a much smaller corpus" per FUTO's description.

**Output:** Stage C checkpoint (finetune/stage_c/final/). This is the final model.

### Phase 5: Package to GGUF

1. Stage the checkpoint into HuggingFace format.
2. Convert to GGUF via llama.cpp's `convert_hf_to_gguf.py`.
3. Patch metadata: add all `keyboardlm.*` fields (languages="eu", features string,
  SentencePiece tokenizer as `[UINT8]` byte blob, general.name/author/description/license/url).
4. Downgrade GGUF v3 → v2 (FUTO requires v2).

**Output:** Final GGUF (~49MB), side-loaded into FUTO via Import from file.

---

## 7. Key Decisions and Rationale (Summary)

| # | Decision | Rationale | Confidence |
|---|----------|-----------|------------|
| 1 | **Llama architecture, 25M params** | Matches reference; FUTO C++ has hard-coded Llama assumptions; small enough for on-device latency | High |
| 2 | **UNIGRAM tokenizer (not BPE)** | Reference uses UNIGRAM; better for agglutinative morphology; probabilistic segmentation handles ambiguous boundaries | High |
| 3 | **Vocab size 4096 (not 15008)** | Ablation shows 4K→66.7% MorphAcc vs 28.6% at 32K; smaller model benefits more from morpheme splitting; context absorbs fertility | Medium-high (tokenizer-level evidence strong; no full-model sweep at other vocabs) |
| 4 | **`add_dummy_prefix=false`, `remove_extra_whitespaces=false`** | Matches reference; dummy prefix would corrupt the `<XBC>`word format | High (verified) |
| 5 | **Two-tier corpus (Latxa + BSM)** | BERnaT paper shows diverse data helps without hurting benchmarks; BSM is deployment-distribution data | High |
| 6 | **BSM in finetune only, NOT pretrain/tokenizer** | Sibling project's concern about dialect polluting vocabulary is valid — applied to the right phase; matches FUTO's own pipeline (web pretrain → conversational finetune) | High |
| 7 | **BSMtime (not BSMauthor)** | BSMauthor concatenates posts into 91KB timeline blobs; BSMtime has individual posts suitable for line-level cleaning | High (verified) |
| 8 | **3B pretrain tokens (not 5B)** | 80–250:1 ratio is the sweet spot for 25M; diminishing returns past ~200:1; faster iteration | Medium-high |
| 9 | **Phase 4c included** | FUTO's own wiki calls it "an important step"; shifts register from Wikipedia to chat | High |
| 10 | **Omit `xc0_swipe_typing_v1`** | No swipe training data; app falls back to dictionary swipe | High |
| 11 | **`plw=0.05` for 4b, `plw=1.0` for 4c** | 4b: preserve pretrain next-word ability, reduce format overfit (arXiv:2401.13586). 4c: register shift needs clean-context training. | Medium-high (proactive fix for mini's 0% next-word) |
| 12 | **GGUF v2, SentencePiece as `[UINT8]`** | Hard requirements of the FUTO format; verified against reference | High |

---

## 8. Evaluation

### 8.1 Test suites

We use two hand-curated Basque test suites (52 tests total):

**Autocorrect (40 tests):** Given a typo as keypresses, does the model predict the correct
word? Covers the dominant Basque typo classes:
- Keyboard-adjacency substitutions (35 tests) — e.g., `kaixp` → `kaixo`, `narkatu` → `barkatu`
- Diacritic loss / ñ→n (1 test) — `inaki` → `iñaki`
- Doubling (2 tests) — `eskkerrik` → `eskerrik`
- Transposition (2 tests) — `garu` → `gaur`

**Next-word (12 tests):** Given a Basque prefix, does the model predict a plausible
continuation? Each test has 3–5 acceptable answers (e.g., `"Egun on, zer"` →
`["moduz", "berri", "da", "nola"]`).

### 8.2 Mini validation baselines (end-to-end pipeline proof)

Before the full run, we completed a **mini validation** — the entire pipeline at 1/12th
scale (500M pretrain tokens, 2000 steps per phase) to prove the pipeline works end-to-end
and establish baselines:

| Metric | Stage B (before 4c) | Stage C (after 4c) |
|--------|---------------------|---------------------|
| Autocorrect top-1 | 40.0% (16/40) | 42.5% (17/40) |
| Autocorrect top-5 | 65.0% (26/40) | 62.5% (25/40) |
| Next-word top-1 | 0% (0/12) | 0% (0/12) |
| Next-word top-8 | 0% (0/12) | 0% (0/12) |

**Diagnostic:** The mini model is **overfit to the XBU autocorrect format** — in next-word
mode, it predicts structural tokens (`<XBU>`, `<XEC>`) and commas instead of real words.
This is expected for the mini scale (524M pretrain tokens, 2000 finetune steps): the
finetune is a large fraction of total training, so the format dominates. The full run
(3B pretrain tokens, 24K steps) should fix this by giving a much stronger base LM and
making the finetune a smaller fraction of total training.

The mini model **does work in the FUTO app** — it predicts valid Basque continuations
(e.g., "amonaren etxera joan" → "zen | da", both valid verb forms) and loads without
crashes. This validates the entire format pipeline (tokenizer layout, GGUF metadata,
feature flags, keypress format).

### 8.3 What success looks like for the full model

- Autocorrect top-1 ≥ 60% (the English reference achieves ~74% top-1)
- Next-word top-1 > 0% (the mini model's 0% indicates format overfit that should resolve)
- No structural-token leakage in next-word mode
- Model feels usable in the FUTO app for real typing

---

## 9. Risks and Open Questions

These are the areas where a senior ML engineer's review would be most valuable:

### 9.1 Vocabulary size (4096 vs larger)

Our strongest evidence (the MorphAcc ablation) is tokenizer-level. We have not trained full
models at 8K/16K/32K for Basque to get end-to-end quality comparisons. The
parameter-efficiency and fertility arguments are strong, and a sibling 91M Mamba2 model
trained successfully with 4K, but the possibility remains that a larger vocab would help
downstream quality despite worse MorphAcc (e.g., by reducing sequence length and improving
attention efficiency). **Mitigation:** If the full model underperforms, retraining the
tokenizer at 8K and repeating the pipeline is a ~2-day experiment.

### 9.2 Next-word prediction quality

The mini model's 0% next-word accuracy (format overfit) is the biggest open question. The
mini used plw=1.0 (full-sequence loss), which trains the clean→`<XBU>` transition at full
weight — the model learned to emit structural tokens too aggressively in next-word mode.

**Proactive mitigation applied:** Phase 4b now uses plw=0.05 (the arXiv:2401.13586
recommendation). This downweights clean-context transitions (preserving the pretrain's
next-word ability) while keeping full weight on span internals (autocorrect learning).
Phase 4c retains plw=1.0 because its goal is register shift (requires training on clean BSM
context), but 4c is short (5000 steps) so format-overfit risk is bounded.

The full run's stronger pretrain (6.3B vs 524M tokens) also helps — a stronger base LM is
more resistant to overwriting. If format overfit persists despite plw=0.05, the fallback is
reducing Phase 4b step count (15000 → 10000) to shrink finetune's share of total training.

### 9.3 Data scale (3B vs more)

We chose 3B based on scaling-law ratios, but Basque is low-resource and the model is small.
There's a risk we're undertraining. The sibling project's 91M model was still improving at
8.8B tokens (97:1). Our ~250:1 ratio is aggressive. **Mitigation:** We save 5 checkpoints
and will pick the best by loss curve; if loss is still descending at 24K steps, we can
continue training.

### 9.4 BSM cleaning quality

The cleaning pipeline achieves a 17% drop rate and verified low noise, but BSM is social
media — there's an inherent noise floor. The code-switching filter (Spanish ratio > 0.15)
is a heuristic and may both miss some code-switching and wrongly drop legitimate Basque
that happens to use Spanish loanwords.

### 9.5 Diacritic reconstruction

The model must reconstruct accents (á, é, í, ó, ú, ü, ñ) from base letters + context, since
the keypress format drops diacritics. Basque accent usage is relatively sparse (mostly ñ
and loanword accents), which helps, but this is a capability the model must learn from the
pretrain data (which has correct accents) and apply during inference (when accents are
stripped from the prompt).

### 9.6 No swipe typing

We omit `xc0_swipe_typing_v1`. Swipe typing falls back to the dictionary engine (which
doesn't exist for Basque), so swipe will be non-functional. This is acceptable for v1 but
is a feature gap vs the English model.

---

## 10. Infrastructure

- **Training hardware:** NVIDIA L40 GPU (46GB VRAM), 30GB system RAM. All phases fit
  comfortably (peak GPU usage in mini validation: 2.7GB).
- **Total estimated time (full run):** ~14h (Phase 3 pretrain dominates at ~10h; finetunes
  ~3h; data prep + packaging ~1h).
- **Experiment tracking:** Weights & Biases (project: `futo-eu`). One run per training phase
  (4 runs total: pretrain, 4a, 4b, 4c).
- **Reproducibility:** All hyperparameters in declarative YAML configs with mini/full modes.
  Mini mode runs the entire pipeline in ~75 min for validation.

---

## Appendix A: The Complete Training Recipe at a Glance

```
Phase 1   Stage 3B clean tokens (Latxa v2, 11 sources)        → corpora/clean/
Phase 1b  Clean + stage 250M BSM (BERnaT BSMtime)             → corpora/conversational/
Phase 2   Train UNIGRAM tokenizer (4096 vocab, clean only)    → tokenizer/
Phase 3   Pretrain Llama-25M (24K steps, 3e-4, clean only)    → pretrain/base/
Phase 4a  Isolated autocorrect (20K steps, 1e-4, synth typos) → finetune/stage_a/
Phase 4b  In-context autocorrect (15K steps, 5e-5, 33% typos) → finetune/stage_b/
Phase 4c  Conversational adaptation (5K steps, 2e-5, BSM)     → finetune/stage_c/
Phase 5   Package → GGUF v2 + keyboardlm.* metadata           → eu_futo_v2.gguf
```

## Appendix B: Final Model Specification

| Property | Value |
|----------|-------|
| Architecture | Llama (MHA, no GQA) |
| Parameters | ~25M |
| Hidden size | 512 |
| FFN size | 1024 |
| Layers | 8 |
| Attention heads | 8 |
| Context length | 2048 |
| Vocab size | 4096 (UNIGRAM) |
| Tokenizer | SentencePiece UNIGRAM, byte-fallback, inverted-space |
| GGUF version | 2 |
| File size | ~49MB |
| Features | `base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1` |
| Language | `eu` (Basque) |
