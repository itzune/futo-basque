# FUTO Keyboard Transformer LM — Research for Basque Text Prediction

> **Goal of this document:** Understand exactly what kind of transformer model FUTO
> Keyboard ships for English text prediction, how it is loaded/imported, and what it
> would take to train an equivalent model for **Basque (euskara, `eu`)**.
>
> Research conducted 2026-07-14 against FUTO Keyboard **v0.1.29.1** (released
> 2026-06-22) and the public source mirror at
> [github.com/futo-org/android-keyboard](https://github.com/futo-org/android-keyboard)
> (internal canonical repo: `gitlab.futo.org/keyboard/latinime`).

---

## 0. TL;DR / Executive Summary

- FUTO Keyboard runs a small **~36M-parameter Llama-architecture transformer** on-device
  via **llama.cpp**, in **GGUF** format, with an embedded **SentencePiece** tokenizer.
- It ships **English only**. The UI says other languages are "coming soon", but the app
  **already supports importing third-party `.gguf` models** (Settings → Languages & Models
  → Transformer Models → **Import from file**), as long as they match FUTO's undocumented
  "KeyboardLM" format.
- The format is **not** "any GGUF model". The model must carry special `keyboardlm.*`
  GGUF metadata, a fixed-layout SentencePiece tokenizer (vocab 15008 with 300 reserved
  user-defined symbols at fixed token IDs), and declare specific **feature flags**.
- The most novel part is the **keypress prompt format**: autocorrect is done by feeding
  each typed character as a discrete `<CHAR_X>` token wrapped in
  `<XBU>…<XBC>correction<XEC>`. The model predicts the corrected word as plain text.
- **Basque is feasible.** Basque uses the Latin-26 alphabet (diacritics ñ á é í ó ú ü are
  long-press variants and decompose to base letters via NFD, exactly like Portuguese).
  This is the same situation as the community **Brazilian Portuguese** effort, for which a
  complete, battle-tested end-to-end pipeline already exists and can serve as a direct
  template.
- There is **no Basque dictionary** shipped with FUTO today (only a keyboard layout,
  `locales/eu.json`), so the transformer model would be the primary prediction engine.

---

## 1. What FUTO Keyboard is

FUTO Keyboard is an offline, privacy-focused Android keyboard, forked from AOSP's
**LatinIME**. Source: [github.com/futo-org/android-keyboard](https://github.com/futo-org/android-keyboard),
licensed under the FUTO Source First License 1.1.

It runs **two prediction engines in parallel and merges them**:

1. **A classical AOSP-style dictionary + bigram engine.** Ships wordlists for ~28
   languages under `dictionaries/<lang>_wordlist.combined.gz`. Gives spellcheck-grade
   completions and simple autocorrect.
2. **A Llama-architecture transformer LM** ("Transformer LM") for context-aware
   autocorrect + next-word prediction. Runs on-device via llama.cpp. **English only today.**

From the official docs ([Text Prediction](https://docs.keyboard.futo.org/settings/textprediction.html)):

> "You can have the keyboard predict the next word you'll type or make more intelligent
> auto-corrections, which uses a pre-trained transformer language model based on
> publicly-available data sets, by toggling on Transformer LM. **Note: This only currently
> works for English, but we are working on making it working for other languages.**"

There is an open, high-priority issue
([#1212 — Transformer Models for More Languages](https://github.com/futo-org/android-keyboard/issues/1212))
where many users volunteer to train models for Norwegian, French, Russian, etc. FUTO's
response so far is "working on it, no ETA". The community has started filling the gap.

---

## 2. The English model: what it actually is

### 2.1 The shipped model

From `java/src/org/futo/inputmethod/latin/xlm/ModelPaths.kt`:

```kotlin
val BASE_MODEL_RESOURCE = R.raw.ml4_q6_k       // bundled as a raw Android resource
val BASE_MODEL_NAME = "ml4_q6_k"
val DEPRECATED_MODEL_NAME = "ml4_1_f16_meta_fixed"
val MODEL_OPTION_KEY = ... setOf("en:$BASE_MODEL_NAME")
```

- The current shipped model is **`ml4_q6_k`** (a Q6_K-quantized build).
- The previous/leaked model is **`ml4_1_f16_meta_fixed`** (F16). A community re-upload of
  this exact file is on HuggingFace: **[`breadlicker45/futo-keyboard-lm`](https://huggingface.co/breadlicker45/futo-keyboard-lm)**
  (`ml4_1_f16_meta_fixed.gguf`). This is the **de-facto reference spec** — "just exported
  from the app and uploaded here."
- On first run the raw resource is copied to `filesDir/transformer-models/ml4_q6_k.gguf`.
  Imported user models also live in that directory.

### 2.2 Architecture (verified against the reference model)

A **vanilla Llama** config, ~36M parameters:

```python
LlamaConfig(
    vocab_size                 = 15008,
    hidden_size                = 512,
    intermediate_size          = 1024,
    num_hidden_layers          = 8,
    num_attention_heads        = 8,
    num_key_value_heads        = 8,        # MHA, no GQA
    max_position_embeddings    = 2048,     # NOT 512 — the old wiki implied 512, reference is 2048
    rms_norm_eps               = 1e-6,     # NOT 1e-5
    rope_theta                 = 10000.0,
    tie_word_embeddings        = False,
)
```

Inference context size is hard-coded to **2048** tokens
(`#define LLAMA_CONTEXT_SIZE 2048` in `native/jni/src/ggml/LanguageModel.h`).

The C++ inference layer (`native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp`)
wraps llama.cpp and exposes two main calls to Kotlin:
- `PredictNextWord(context, banned_words)` — next-word prediction.
- `PredictCorrection(context, mixes, swipe_mode, capitals, banned_words)` — autocorrect
  given the current partial/typed input.

The Kotlin side (`xlm/LanguageModel.kt`) merges the transformer's suggestions with the
AOSP dictionary suggestions, applies a "Transformer LM strength" slider, and surfaces up
to 128 candidate words.

---

## 3. The tokenizer: 300 user-defined symbols at fixed slots

This is the part that is **most format-critical and least documented**. The tokenizer is a
**SentencePiece UNIGRAM** model with **vocab size 15008**, embedded *inside* the GGUF file
(not the standard `tokenizer.ggml.tokens` array — FUTO ignores that and uses its own
`keyboardlm.ext_tokenizer_data` blob).

The 15008 IDs are laid out as:

| ID range | Count | Contents |
|---|---|---|
| 0–3 | 4 | `<pad>`, `<s>`, `</s>`, `<unk>` (control + unknown) |
| 4–303 | 300 | **User-defined symbols** (fixed structural layout — see below) |
| 304–559 | 256 | Byte-fallback `<0x00>`…`<0xFF>` (auto-generated by SentencePiece) |
| 560–15007 | 14448 | UNIGRAM pieces learned from corpus (your language fills this) |

The 300 user-defined symbols (indices 4–303) are a **structured layout, not just any 300
strings**. Verified by reading the reference model + the FUTO C++ source
(`native/jni/src/ggml/LanguageModel.cpp`,
`native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp`):

```
4..27    (24)  <FUTO0>..<FUTO23>           reserved/inert filler slots
28..173 (146)  content slots               → REPLACE with your-language common words/contractions
174       (1)  <XBU>    STRUCTURAL: autocorrect "begin user input"   (name-looked-up)
175       (1)  <XBC>    STRUCTURAL: autocorrect "begin correction"   (name-looked-up)
176       (1)  <XEC>    STRUCTURAL: autocorrect "end correction"     (name-looked-up)
177..181  (5)  <XC0>..<XC4>                 only <XC0> is referenced (swipe mode); rest reserved
182..207 (26)  <CHAR_A>..<CHAR_Z>           STRUCTURAL: per-keypress tokens
                ⚠️ MUST be 26 contiguous, sequential IDs (C++ does pointer arithmetic)
208..263 (56)  more content slots          → REPLACE with your-language common words
264..303 (40)  emoji set
```

### 3.1 Two lookup mechanisms in the C++ (important)

- **By name** (via SentencePiece `PieceToId`): `<XBU>`, `<XBC>`, `<XEC>`, `<XC0>`,
  `<CHAR_A>`, and the SP space marker `▁`. Their IDs can be anywhere as long as the name
  resolves to non-zero.
- **By computed index**: `<CHAR_B>`…`<CHAR_Z>` are read as `LETTERS_TO_IDS[0] + i`. So the
  26 `<CHAR_*>` symbols **must** occupy 26 contiguous IDs. The simplest way to guarantee
  this is to list them sequentially in `user_defined_symbols` (SentencePiece preserves
  declaration order).

If `<XBU>`, `<XBC>`, `<XEC>`, `<XC0>`, or `<CHAR_A>` resolves to `0` (=`<unk>`), the C++
**asserts and crashes** at model load.

---

## 4. The keypress prompt format (the key undocumented detail)

The model is **not** prompted with literal typo text like `<XBU>teh<XBC>`. Each character
the user types becomes a **discrete `<CHAR_X>` token**, and the model is trained to
predict the corrected word as plain text between `<XBC>` and `<XEC>`.

```
context + <XBU><CHAR_T><CHAR_E><CHAR_H><XBC>The <XEC>
                 ^^^^^^^^^^^^^^^^^^^^^^^ typed keys        ^^^^^^ corrected word
```

Verified example against the reference English model:

```
prompt:  <XBU><CHAR_T><CHAR_E><CHAR_H><XBC>
output:  The <XEC>...        → ~74% top-1 accuracy

prompt:  <XBU>teh<XBC>       → 0% accuracy (model produces nonsense)
```

This is generated by FUTO's own `TrainingDataGenerator.kt` (used for on-device
fine-tuning), which defines the special tokens and the per-character mapping:

```kotlin
const val TOKENIZER_BEGIN_USER_INPUT = "<XBU>"
const val TOKENIZER_BEGIN_CORRECTION = "<XBC>"
const val TOKENIZER_END_CORRECTION   = "<XEC>"

private val TOKENIZER_LETTER_MAPPING = hashMapOf(
    'a' to "<CHAR_A>", 'b' to "<CHAR_B>", ..., 'z' to "<CHAR_Z>",
)

// format: <XBU><CHAR_h><CHAR_e><CHAR_l><CHAR_l><CHAR_o><XBC>hello <XEC>
```

### 4.1 Implication for diacritics (critical for Basque)

The Android keyboard emits **one `<CHAR_X>` token per physical key press**. Diacritics are
**not** separate tokens. For a long-press `á`, the keyboard emits only `<CHAR_A>`; the
diacritic is dropped from the prompt. The model must **reconstruct** the correct accented
form from context + base letters.

The robust typo→keypress conversion normalizes via NFD and strips combining marks:

```python
import unicodedata
def to_keypress_chars(typed: str) -> list[str]:
    out = []
    for ch in typed:
        decomposed = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        if base in ("ç", "Ç"):
            base = "C"
        for c in base.upper():
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out
```

**→ This is why the `<CHAR_*>` set is hard-coded to the 26 Latin letters.** Basque
(ñ, á, é, í, ó, ú, ü) and Portuguese (ã, ç, é, …) are all fine because their diacritics
decompose to a–z. Languages outside Latin-26 (Cyrillic, Greek, Arabic, CJK) would need
**patches to the FUTO Android keyboard itself**, not just a different tokenizer — out of
scope.

---

## 5. GGUF metadata (the `keyboardlm.*` namespace)

Defined in `native/jni/src/ggml/ModelMeta.h` / `ModelMeta.cpp`. A valid FUTO model must
carry these KV fields **in addition** to standard llama.cpp GGUF fields:

| Field | GGUF type | Purpose |
|---|---|---|
| `general.name` | STRING | model name |
| `general.author` | STRING | author |
| `general.description` | STRING | description |
| `general.license` | STRING | license |
| `general.url` | STRING | source URL |
| `keyboardlm.languages` | STRING | space-separated BCP-47 tags, e.g. `"en"` or `"eu"` |
| `keyboardlm.finetuning_count` | UINT32 | number of on-device fine-tunes applied (0 for fresh) |
| `keyboardlm.history` | STRING | freeform history log |
| `keyboardlm.features` | STRING | space-separated feature flags (see §6) |
| `keyboardlm.ext_tokenizer_type` | STRING | `"sentencepiece"` |
| `keyboardlm.ext_tokenizer_data` | **[UINT8]** | raw bytes of the `.spm` SentencePiece model |

> ⚠️ `keyboardlm.ext_tokenizer_data` **must be `[UINT8]`**, not `[INT32]`. The naive
> `add_array(name, list(bytes))` produces `[INT32]` (4× size, wrong layout) and FUTO's C++
> rejects it. Use `add_key_value(..., sub_type=GGUFValueType.UINT8)`.

---

## 6. Feature flags

From `ModelPaths.kt`, the set of features the app currently understands:

```kotlin
private val supportedFeatures = setOf(
    "base_v1",
    "inverted_space",
    "xbu_char_autocorrect_v1",
    "lora_finetunable_v1",
    "xc0_swipe_typing_v1",
    "char_embed_mixing_v1",
    "experiment_linear_208_209_210",
)
```

What they mean:

| Feature | Meaning |
|---|---|
| `base_v1` | basic next-word LM |
| `inverted_space` | SentencePiece trained with `treat_whitespace_as_suffix=true` (space attaches to the *preceding* token, `▁`-suffix style) |
| `xbu_char_autocorrect_v1` | enables the `<XBU><CHAR_*><XBC>…<XEC>` autocorrect path |
| `char_embed_mixing_v1` | **required** if `xbu_char_autocorrect_v1` is set — the C++ populates `LlamaAdapter::embeddings` from the `<CHAR_*>` token embeddings and mixes them into the prompt; without it, every keystroke **SIGSEGVs** |
| `lora_finetunable_v1` | model is set up for on-device LoRA personalization (see §7) |
| `xc0_swipe_typing_v1` | ML swipe decoding (needs extra encoder tensors at hard-coded indices 208/209/210) |
| `experiment_linear_208_209_210` | the swipe encoder weights (IDs 208/209/210) |

**Minimum working feature set for autocorrect + next-word (what we need for Basque):**

```
base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1
```

(`xc0_swipe_typing_v1` + `experiment_linear_208_209_210` can be omitted — the keyboard
falls back to dictionary-based swipe.)

---

## 7. On-device LoRA fine-tuning ("Personalized suggestions")

FUTO can fine-tune the model **on the phone** from the user's own typed text — this is the
"Personalized suggestions" toggle. Implemented in `xlm/AdapterTrainer.kt` +
`native/jni/org_futo_inputmethod_latin_xlm_AdapterTrainer.cpp` (which calls
`native/jni/src/ggml/finetune.h` → `finetune_train`, i.e. llama.cpp's LoRA trainer).

Key parameters (hard-coded defaults for on-device personalization):

```cpp
params.lora_r = 16;            params.lora_alpha = 16;
params.common.n_threads = 6;
params.common.n_epochs = 1;
params.common.n_ctx = 64;
params.common.n_batch = 2;
params.common.n_gradient_accumulation = 2;
params.common.adam_alpha = 1e-3;
params.common.adam_n_iter = 128;
params.common.warmup = 10;
```

- Training examples come from the user's typed text (each example trimmed + space-appended).
- The SentencePiece tokenizer is loaded from the model's own `ext_tokenizer_data`.
- After training, the LoRA adapter is applied to the base model with
  `llama_model_apply_lora_from_file(...)` and a **full merged GGUF** is written out
  (with `finetuning_count` incremented and a history entry appended).
- Requires the base model to declare `lora_finetunable_v1` and be prepared with the right
  tensor metadata.

> The community Portuguese guide did **not** enable on-device LoRA (`lora_finetunable_v1`)
> because preparing a model for it requires specific tensor metadata that isn't fully
> documented. We can defer this; personalization is a nice-to-have, not a requirement for
> a working Basque model.

---

## 8. How model import works (the "Import from file" path)

`ModelPaths.importModel()` is the gate. When you pick a `.gguf` file in the UI
(`modelmanager/ModelList.kt` → "Import from file"), it:

1. Checks the extension is `.gguf`.
2. Reads the first 4 bytes and verifies the **`GGUF` magic** (`'G','G','U','F'`).
3. Copies the file to `filesDir/transformer-models/<name>.gguf`.
4. Loads metadata via `ModelInfoLoader` (native) and checks:
   - **features must be non-empty** → otherwise:
     *"Model is a valid GGUF file, but does not support use as a keyboard language model
     (it lacks KeyboardLM metadata). … models must support specific features and prompt
     formats; arbitrary gguf models are unsupported at this time. Refer to the model
     creation documentation for more details."*
   - **all features must be recognized** (or start with `opt_` / `_`) → otherwise rejected
     as "probably need to update FUTO Keyboard".
5. Lists models grouped by `keyboardlm.languages` and lets you assign one per language via
   the `lmModelsByLanguage` setting (e.g. `"eu:my_basque_model"`).

So a correctly-formatted Basque `.gguf` with `keyboardlm.languages = "eu"` will appear
under the Basque language and can be selected — **no app modification needed**.

---

## 9. How to train a model for a new language (the full pipeline)

There is **no base-model training script in the FUTO repo** — only the C++ inference and
on-device LoRA trainer. The base `ml4` English model was trained internally at FUTO and is
not reproducible from the public repo.

However, a complete community pipeline exists: **[`danmaxis/futo-portuguese`](https://github.com/danmaxis/futo-portuguese)**
(GUIDE.md + `scripts/`), which trained a working Brazilian-Portuguese model end-to-end and
side-loaded it. It is the authoritative real-world reference and a near-perfect template
for Basque. The pipeline:

```
[corpus] --Phase 2--> [tokenizer] --Phase 3--> [pretrain base]
                                                 │
                                         Phase 4 (3 stages)
                                                 │
                                          [final HF ckpt]
                                                 │
                                          convert + patch
                                                 │
                                          [.gguf v2 file]
                                                 │
                                              phone
```

### Phase 0 — extract the reference model as the spec
```bash
pip install sentencepiece huggingface_hub gguf protobuf numpy
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
hf download breadlicker45/futo-keyboard-lm --local-dir reference_model/
python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
       reference_model/ml4_1_f16_meta_fixed.gguf > reference_metadata.txt
```
Treat `reference_metadata.txt` as the spec — match every field/tensor name.

### Phase 1 — assemble a target-language corpus
- **Register matters more than register.** A Wikipedia-only model is bad at predicting chat.
- For Basque we use a **two-tier** strategy (see §11.4): a *clean* tier (Morpheus's
  Latxa corpus v2, 11 HiTZ-curated sources, ~4.77 B tokens) for tokenizer + pretrain
  base, and a *conversational* tier (BERnaT BSM, ~250 M tokens, aggressively cleaned)
  as a pretrain supplement only. The tokenizer trains on the clean tier so the UNIGRAM
  vocabulary stays morpheme-focused.
- Both scripts support a local mode (stage from a morpheus-mamba checkout) and an HF
  streaming fallback (`--from-hf`). Stream + dedup + write ~256 MiB shards. Target
  **2–5 B tokens** (Morpheus's quality-over-quantity finding; our 36 M model is smaller
  than their 91 M so this is plenty).

### Phase 2 — train the SentencePiece tokenizer
4096-vocab UNIGRAM (not the reference's 15008 — see §11.3.1 for the fertility/MorphAcc rationale),
300 user-defined symbols pinned at IDs 4–303, byte-fallback, inverted-space.
Validate `<CHAR_A>..<CHAR_Z>` are sequential and `<XBU>/<XBC>/<XEC>/<XC0>` are non-zero.
- ⚠️ **Memory trap:** SP UNIGRAM is RAM-bound (~5–10× input size, more than BPE due to EM). Use a host with **32+ GiB
  RAM**, or pre-sample to ~1–2 GiB.

### Phase 3 — pretrain the base model (HuggingFace Trainer)
```python
TrainingArguments(
    max_steps=100_000,                  # 20k for a mini validation run
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,     # global batch 256
    learning_rate=3e-4, warmup_steps=2000,
    lr_scheduler_type="cosine", bf16=True, weight_decay=0.1, save_steps=5000,
)
# seq_len 1024; ~262k tokens/step; ~26B tokens over 100k steps ≈ 5 epochs of a 5B corpus
```
Expect final loss ~3–4, perplexity ~30–50. Plateau by 60–80% of steps.

### Phase 4 — autocorrect fine-tune (3 stages) — teaches the `<XBU>…<XEC>` format
- **4a — isolated triples:** synthetic `<XBU><CHAR_*>…<XBC>correct<XEC>` from a word
  frequency map. Loss masked to the correction span only. ~5–10k steps, seq 64, lr 1e-4.
  - Typo classes for diacritic languages: **drop accents** (dominant), cedilla loss,
    keyboard-adjacency (QWERTY), transposed/doubled/missing chars, real shortcuts speakers
    type. Weight sampling by `log(freq+1)`.
- **4b — in-context:** inject ~20–30% of corpus words as typo triples in real sentences;
  loss over all tokens. ~20–30k steps, seq 512, lr 5e-5. (Don't exceed ~25% typo rate or
  the model mode-collapses to "always emit a triple".)
- **4c — conversational adaptation:** short fine-tune on casual text (subtitles/chat) at
  low typo rate (~0.10), ~5k steps, lr 2e-5. Shifts toward typing register.

### Phase 5 — package as a FUTO-compatible GGUF (the gotcha-heavy part)
1. Stage the HF checkpoint with the SentencePiece as `tokenizer.model` + minimal
   `tokenizer_config.json` / `special_tokens_map.json`.
2. `python llama.cpp/convert_hf_to_gguf.py staged/ --outfile vanilla.gguf --outtype f16`
3. Requantize `output.weight` to **Q6_K** (matches reference):
   `llama-quantize --allow-requantize --output-tensor-type q6_k vanilla.gguf q6kout.gguf f16`
4. Patch FUTO metadata (`keyboardlm.*`, including the SentencePiece as `[UINT8]`).
5. **Downgrade GGUF v3 → v2** and strip ~9 newer-than-v2 KV fields that FUTO's vendored
   llama.cpp can't parse (`general.size_label`, `general.type`,
   `llama.attention.key_length`, `llama.attention.value_length`, `llama.vocab_size`,
   `tokenizer.ggml.add_bos_token`, `tokenizer.ggml.add_eos_token`,
   `tokenizer.ggml.padding_token_id`, `tokenizer.ggml.pre`); patch version byte at offset
   4–7 from `\x03…` to `\x02…`.
6. Final diff against `reference_metadata.txt` — only `keyboardlm.languages` and
   `keyboardlm.history` should differ.

Result: a ~62 MB `.gguf` that loads and runs on a real phone.

### Hardware / time budget (single RTX 3090)
| Phase | Time |
|---|---|
| Corpus assembly (3–5B tokens) | 1–3 h (network-bound) |
| Tokenizer training | 15–60 min (RAM-bound) |
| Pretrain (~100k steps) | **30–50 h** |
| Fine-tune (4a+4b+4c) | 3–8 h |
| GGUF assembly + side-load | <30 min |

Minimum to ship: one NVIDIA GPU with **16+ GiB VRAM** (24 GiB comfortable). CPU training
impractical. Mini validation run (single source, 20k pretrain steps) ≈ 14 h total.

---

## 10. The five gotchas (in priority order)

1. **Keypress format is `<CHAR_X>` tokens, not literal text.** Easy to misread; verify
   with a test prompt against the reference English model before training for days.
2. **`char_embed_mixing_v1` is required if you declare `xbu_char_autocorrect_v1`.** The
   wiki frames them as independent; they are not. Without it, every keystroke SIGSEGVs in
   `DecodePromptAndMixes`.
3. **GGUF must be version 2.** Recent `convert_hf_to_gguf.py` emits v3 with extra KV
   fields FUTO's vendored llama.cpp can't parse → needs a downgrade pass.
4. **`keyboardlm.ext_tokenizer_data` must be `[UINT8]`**, not `[INT32]`.
5. **`<CHAR_A>..<CHAR_Z>` must be 26 contiguous, sequential token IDs** (C++ does pointer
   arithmetic on `<CHAR_A>`'s ID). List them in order in `user_defined_symbols`.

Debugging a side-loaded model that crashes: enable wireless ADB, then
`adb logcat | grep -E "Fatal signal|F libc|F DEBUG"`. A small fault address (under
`0x100000`) = `nullptr + offset` = missing feature/uninitialized field. `ASSERT failed`
at app start = a missing structural special token in the tokenizer.

---

## 11. Basque-specific analysis & feasibility

### 11.1 Is Basque compatible with the format? — **Yes**

- Basque uses the **Latin-26 alphabet**. The only non-ASCII letters are **ñ** and the
  accented vowels **á é í ó ú ü**, all of which are **long-press variants** of base keys
  in FUTO's own Basque layout (`tools/make-keyboard-text-py/locales/eu.json`):
  ```json
  "a": ["á","à","ä",...], "e": ["é",...], "i": ["í",...],
  "o": ["ó",...], "u": ["ú","ü",...], "n": ["ñ","ń"], "c": ["ç",...]
  ```
- All these diacritics **decompose to base Latin letters via NFD** (ñ→n, á→a, ü→u, ç→c).
  This is **exactly the Portuguese situation** (ã, ç, é) and is fully handled by the
  NFD keypress conversion. **No patches to the FUTO keyboard are needed.**

### 11.2 Current Basque support in FUTO
- ✅ Keyboard **layout** exists (`locales/eu.json`, `TestsBasqueES.java`).
- ❌ **No Basque wordlist dictionary** in `dictionaries/` (only cs, da, de, el, en, es, fi,
  fr, hr, it, iw, lt, lv, nb, nl, pl, pt_BR, pt_PT, ro, ru, sl, sr, sv, tr). So today a
  Basque user gets only the (Latin-derived) base AOSP behavior — no real Basque
  predictions.
- ❌ No Basque transformer model.

→ A Basque transformer model would be the **primary** prediction engine for Basque users,
  which makes this project high-value. (Optionally, a Basque `eu_wordlist.combined.gz` could
  be contributed separately for the dictionary engine, but it's not required to use the
  transformer.)

### 11.3 Basque language considerations for training
- **Agglutinative morphology.** Basque builds words by stacking suffixes
  (e.g. *etxea* "the house", *etxera* "to the house", *etxekoak* "those of the house").
  Words can be long. Implications:
  - Subword tokenization handles this well; UNIGRAM is especially suited (probabilistic
    segmentation handles morphological ambiguity better than BPE's greedy approach).
  - The autocorrect `<CHAR_*>` path still works (it's character-level on the input side);
    the model just has to learn to output longer, suffixed words.
  - **Vocabulary size is critical** — see §11.3.1 below.
- **Diacritic typo class is dominant.** Missing ñ (ñ→n) and missing accents (á→a, ü→u) will
  be the most common corrections the model must learn — this is exactly what the
  `<XBU>…<XBC>` format is designed for.
- **Typing register.** Basque has diglossia with Spanish; many Basque speakers type a mix.
  A Basque-only model is the cleanest first target; multilingual eu+es can come later
  (FUTO supports multilingual typing, but each model is single-language).

### 11.4 Basque corpus strategy (two-tier: Latxa v2 + BERnaT BSM)

The initial plan (generic HF web crawls: Wikipedia-eu, OSCAR-eu, mC4-eu, CC-100-eu +
OpenSubtitles/CommonVoice) was **replaced** after examining the morpheus-mamba
project's corpus quality research. Morpheus's key finding: **data quality is the
binding constraint, not quantity** — their 91 M model converged at ~8.8 B tokens,
and a 2–5 B token high-quality subset would likely match 10 B mixed-quality.

#### 11.4.1 Clean tier — Morpheus's Latxa corpus v2 (primary)

We reuse morpheus-mamba's `data/clean-v3/` — the cleaned, deduplicated,
per-source-audited [Latxa corpus v2](https://huggingface.co/datasets/HiTZ/latxa-corpus-v2)
(HiTZ, the Basque NLP center). This skips reimplementing morpheus's entire cleaning
pipeline (3 phases: dedup → structural clean → digit/fragment filtering).

**11 sources included** (LLM-audited quality 1–5, from morpheus's 40-line-per-source audit):

| Source | Quality | Type | Notes |
|---|---|---|---|
| euscrawl-v2 | 5.0 | news/media | best source — 56 % of EusCrawl v1 |
| parleus | 4.9 | parliament | transcriptions of Basque Parliament sessions |
| zelaihandi | 4.9 | diverse | curated multi-genre corpus |
| bopv | 4.8 | gazette | Basque Government official gazette |
| botha | 4.8 | gazette | Álava provincial gazette |
| colossal-oscar | 4.7 | web crawl | cleaned Common Crawl |
| wikipedia | 4.6 | encyclopedic | Basque Wikipedia dump (Sep 2025) |
| cultura-x | 4.6 | web crawl | cleaned web (CulturaX) |
| hplt-v2 | 4.4 | web crawl | HPLT v2 |
| fineweb2 | 4.3 | web crawl | FineWeb2 |
| finepdfs | 3.7 | PDFs | FinePDFs (digit-filtered in Phase 3) |

Final: **11 sources, ~140 M lines, ~15 GB, ~4.77 B pre-tokenized tokens** (uint16 .npy).
Used for **tokenizer training + pretrain base**.

**4 sources morpheus excluded** (and why):

| Source | Why excluded |
|---|---|
| hplt-v1 | 83.8 % duplicates, only 4.9 % Basque signal — net negative |
| bog | Phase-2 sentence splitting fragmented legal text (36/40 lines incomplete) |
| aldizkariak | 35 % boilerplate (author lists, English titles, citation numbers) |
| **BERnaT BSM** | morpheus excluded a priori (dialectal/code-switching fear) — **we re-include** (see §11.4.2) |

#### 11.4.2 Conversational tier — BERnaT BSM (re-included)

Morpheus excluded BERnaT BSM (Basque Social Media) based on an **untested assumption**:

> "Twitter/conversational text introduces dialectal Basque, heavy code-switching, emoji,
> and non-standard orthography that would bias a morphology-targeted tokenizer toward
> informal patterns at the expense of the formal Basque needed for good morphological
> segmentation." — morpheus corpus-quality-fast-audit.md

But `HiTZ/BERnaT-Diverse` is a **published paper** (Azurmendi et al. 2025,
arXiv:2512.03903, submitted to LREC 2026) from **the same HiTZ group** that built the
Latxa corpus. They ran the controlled experiment morpheus guessed at, and found the
**opposite**:

> "Models trained on both standard and diverse data consistently outperform those
> trained on standard corpora, improving performance across all task types **without
> compromising standard benchmark accuracy**."

**Why this matters for FUTO specifically:**

1. **Distribution match.** FUTO is a phone keyboard. Users type chat messages, not
   Wikipedia articles. BSM (11 M real Basque posts by 13 K users, ~188 M words) is the
   closest deployment-distribution data that exists for Basque.
2. **The two FUTO tasks split cleanly.** Autocorrect (`typo → correct word`) wants
   standard batua as the *correction target* — handled by fine-tune phase 4a with curated
   batua targets (BSM never touches this). Next-word prediction wants conversational
   fluency — BSM is exactly this.
3. **Scale is safe.** BSM is ~250 M tokens. Against a 2–5 B Latxa base, that's 5–10 % —
   enough to shift the model toward casual phrasing, not enough to corrupt it.

**The critical scoping rule: BSM is in pretraining, NOT in tokenizer training.**

| Phase | Sees BSM? | Why |
|---|---|---|
| Tokenizer training (Phase 2) | ❌ No | Keep UNIGRAM vocabulary morpheme-focused on standard batua. BSM slang could pollute the piece inventory. |
| Pretraining (Phase 3) | ✅ Yes (~5–10 %) | Model learns conversational patterns in its *weights* using the clean vocabulary. This is exactly what BERnaT did. |
| Autocorrect fine-tune (4a) | ❌ No | Targets must be standard batua forms. |
| Fulltext fine-tune (4b) | ✅ Maybe | Real conversational sentences with injected typos — good distribution match. |

This gives us the conversational fluency (BERnaT's finding) without risking the
morpheme-aligned vocabulary (morpheus's valid concern, applied to the right phase).

**BSM cleaning pipeline** (`scripts/corpus/clean_bernat.py`):

- *Strip*: emoji (Unicode ranges), URLs, @mentions, hashtag symbols (keep the word),
  RT/via attribution prefixes.
- *Filter*: code-switched lines (Spanish function-word ratio > 0.15), too-short lines
  (< 3 words after stripping), pure-punctuation/emoji lines, exact duplicates.
- *Keep*: dialectal Basque spelling, informal grammar/word order, slang — this is the value.

**Two BSM gotchas:**
1. **BSMauthor vs BSMtime are the SAME text**, reordered (by-author vs chronological).
   Using both = 2× duplication for an autoregressive LM. We use **BSMauthor only**
   (grouped by author = coherent voice).
2. **EKC (historical texts) — excluded.** BERnaT-Diverse also includes EKC: 338 classical
   Basque texts from the 16th–19th century, pre-standardization. For a modern phone
   keyboard, 400-year-old Basque is wrong-distribution noise.

### 11.5 Basque tokenizer slot fillers (sketch)
Following the Portuguese template, the 300 user-defined symbols would be filled with
high-frequency Basque items, e.g. for the content slots (28–173, 208–263):

- **Common words/shortcuts:** `eta` (and), `ez` (no), `bai` (yes), `da` (is),
  `dira` (are), `zezen`, `ni` (I), `zu` (you), `gu` (we), `hau` (this), `horixe` (that),
  `bat` (one), `bi` (two), `asko` (a lot), `gutxi` (few), `orain` (now), `gero` (then),
  `hemen` (here), `han` (there), `ni`/`zu`/`bera` pronouns, verb forms of *izan* (to be)
  and *ukan/edun* (to have)…
- **Common suffixes** as subword pieces will be learned automatically (don't waste
  user-defined slots on them).
- **Chat shortcuts** Basque speakers actually type (if any standard ones exist).
- Keep the 40 emoji slots as parity with the reference.

#### 11.3.1 Vocabulary size: 4096, not 15008 (the fertility paradox)

The reference English model uses vocab=15008. English is isolating, so large-vocab
surface-form memorization is harmless. **Basque is agglutinative** — a large vocab
memorizes common inflected forms as single tokens, hiding morpheme boundaries from
the model.

Our sibling project [`morpheus-mamba`](https://github.com/itzune/morpheus) ran a
controlled tokenizer ablation (21 Basque test words across 5 roots, 336 MB corpus
sample, identical SentencePiece params except vocab size). Results:

| Vocab | MorphAcc consistency | Fertility | Example: `etxetik` |
|-------|---------------------|-----------|--------------------|
| 4K | **66.7%** (14/21) | 2.58 | `▁etxe` + `tik` ✅ |
| 8K | 61.9% (13/21) | 2.28 | `▁etxe` + `tik` ✅ (inconsistent across paradigm) |
| 16K | 52.4% (11/21) | 2.06 | `▁etxetik` ❌ whole word |
| 32K | 28.6% (6/21) | 1.85 | `▁etxetik` ❌ whole word |

**MorphAcc** = does the tokenizer split at the root-suffix boundary? At 32K, common
forms (`etxea`, `etxera`, `etxetik`) are single tokens — the model never sees `-ra` or
`-tik` as independent units, so it can't generalize across the paradigm. At 4K, the
tokenizer is *forced* to split, giving the model consistent morpheme tokens.

This replicates the QuechuaTok finding (Contreras 2026) for Basque. The morpheus-mamba
team trained a full 91M Mamba2 model with the 4K tokenizer (converged, PPL 7.13). They
did **not** train full models at 8K/16K/32K for Basque — the downstream-quality
evidence for "smaller is better" comes from QuechuaTok (different language family).
However, the MorphAcc finding is tokenizer-level (architecture-independent) and the
parameter-efficiency argument is strong.

**FUTO-specific adaptation:** FUTO has 560 reserved slots (4 control + 300 structural +
256 byte-fallback), vs morpheus's ~260. So to match morpheus's ~3740 *learned* pieces
(their 4K), FUTO needs vocab ≈ 3740 + 560 = 4300. We use **4096** (3536 learned pieces —
within 5% of the 4K regime, given the smooth MorphAcc curve).

**Why this is even more justified for FUTO than morpheus:**
1. The 36M model is smaller than morpheus's 91M → less capacity to waste on surface
   forms → morpheme splitting helps more.
2. FUTO's 2048 context window (vs morpheus's 1024) easily absorbs the 39% fertility
   increase (2.58 vs 1.85 tokens/word). ~794 words fit in context vs morpheus's ~397.
3. The FUTO app reads vocab size dynamically (`llama_n_vocab()` / `spm.GetPieceSize()`) —
   verified in `JNI_LanguageModel.cpp`. No hardcoding of 15008 anywhere.
4. Warm-starting from the English reference is a dead argument regardless of vocab size:
   learned pieces differ between English and Basque corpora (different SP training).
   Only structural + byte-fallback tokens are shared, and those carry no semantics.

**Verification:** the tokenizer training script includes a MorphAcc spot-check
(`etxetik` → `▁etxe` + `tik`?) that warns if splitting degrades below 60%.

The structural slots (`<FUTO0-23>`, `<XBU>`, `<XBC>`, `<XEC>`, `<XC0-4>`,
`<CHAR_A-Z>`) **must** stay at the exact same positions as the reference.

---

### 11.6 How much data, and what kind? (data strategy analysis)

This section was prompted by a key observation: our **mini validation model** (only
2000 pretrain steps, ~525M tokens, 25M params) already produces plausible Basque
predictions (e.g. "amonaren etxera joan" → "zen | da" — both valid verb forms). This
raised three questions:

1. Do we really need 5B tokens, or is less enough?
2. A mobile predictive keyboard is a *conversational* use case — shouldn't we
   prioritize chat/SMS/social data over web crawl?
3. Is there research to guide this?

**Sources consulted:** FUTO's own official wiki (GitLab, not the Portuguese community
guide), the Gboard federated learning paper (Hard et al. 2018, arXiv:1811.03604), the
SlimPajama-DC data combinations study (Shen et al. 2023, arXiv:2309.10818), the morpheus
data-scaling analysis, and the Vertanen & Kristensson mobile text corpus research.

#### 11.6.1 How FUTO trained their English model (from FUTO's own wiki)

FUTO's [official wiki](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
describes their training pipeline verbatim:

1. **Pretrain on SlimPajama** — "The model was pretrained on a few billion tokens of
   SlimPajama." SlimPajama is a 627B-token English dataset combining CommonCrawl (52%),
   C4 (27%), GitHub (5%), Books (4%), ArXiv (5%), Wikipedia (4%), StackExchange (3%).
   This is **general-purpose web text**, not chat/SMS.
2. **Phase 4a (isolated autocorrect triples)** — synthetic typo→correct pairs, weighted
   by `log(N)` frequency.
3. **Phase 4b (in-context autocorrect)** — SlimPajama sentences with ~1/3 of words
   replaced by typo-correction triples.
4. **Phase 4c (conversational adaptation)** — "the model is finetuned on a much smaller
   corpus representing **Internet first-person speech** with the same 1/3 misspelling
   augmentation. This has appeared to be an important step for the model to comprehend
   that sentences can start with 'I'm', certain slang/lingo, etc."

**Key insight:** FUTO's English model is ~95% web text + a small conversational
domain-adaptation finetune at the end. They explicitly identify this as important but
keep it as a *small final-stage finetune*, not the pretraining base.

FUTO also says: "The model should be pretrained on text from your language. A few
billion tokens or so may suffice."

#### 11.6.2 The Gboard finding: training data composition

Google's Gboard team published the definitive study on keyboard LM training data
(Hard et al. 2018, arXiv:1811.03604). Their key data finding:

> **Gboard's training data was 60% chat apps, 35% web input, 5% long-form text.**

| App type | Share of training data |
|----------|----------------------|
| Chat (messaging) | **60%** |
| Web input | 35% |
| Long-form text | 5% |

They trained a 1.4M-parameter CIFG RNN on 7.5 billion sentences (avg 4.1 words/sentence
= ~30B tokens). They explicitly noted: "the client caches are believed to more
accurately represent the true typing data distribution" and that federated training
on this distribution beat server training on logged data.

**However**, their model was an n-gram-replacement RNN (10K vocab, 96-dim, 1.4MB) —
*not* a transformer LM doing next-word + autocorrect. Gboard's later transformer work
(Xu et al. 2023, arXiv:2305.18465) used federated learning with differential privacy but
did not publish data composition details.

#### 11.6.3 The SlimPajama-DC finding: diversity matters

The SlimPajama-DC study (Shen et al. 2023) systematically tested removing/combining data
domains for LLM pretraining. Their key finding:

> "More domain combinations with diverse training data bring better overall accuracies."

Removing any single domain hurt performance. The best configurations were *diverse*
combinations, not single-domain. **However**, this study was for general-purpose LLMs
(GPT-Neo, Cerebras-GPT, 1.3B params) evaluated on MMLU/HellaSwag/ARC — not keyboard
prediction. The diversity finding transfers; the specific domain weights don't.

#### 11.6.4 Synthesis: what this means for our Basque model

| Question | Answer | Evidence |
|----------|--------|----------|
| Do we need 5B tokens? | **No — 2-3B is likely sufficient** for a 25M param model | Morpheus 91M converged at 8.8B (97:1 ratio). Our 25M model at 2-3B = 80-120:1, in the MiniCPM/Mosaic sweet spot. Chinchilla optimal for 25M = 500M tokens (20:1). |
| Should we prioritize chat data? | **Yes, but as a finetune stage, not the pretrain base** | FUTO's own pipeline: SlimPajama (web) for pretrain + small conversational finetune at the end. Gboard's data was 60% chat, but their model was a different architecture and use case. |
| Is BERnaT BSM enough conversational data? | **Yes, for the finetune stage** | BSM is ~250M tokens of social media — plenty for a Phase 4c-style adaptation finetune. opus-100 en-eu adds ~6.6M tokens of subtitle dialogue if more is needed. |
| Should we add OpenSubtitles? | **Optional** — ~6.6M tokens from opus-100 en-eu, more from OpenSubtitles2024 | Subtitles are translated dialogue, reasonably conversational. But the volume is small vs BSM. |

#### 11.6.5 Recommended data strategy (revised)

Based on this analysis, our current approach is mostly right but should be adjusted:

**Pretrain (Phase 3): 2-3B tokens, not 5B.**
- Use the clean Latxa tier (~2-3B tokens sampled from the 4.77B available).
- This is a 80-120:1 token:param ratio for our 25M model — well within the
  MiniCPM/Mosaic optimal range, and avoids wasting compute on diminishing returns.
- Morpheus's 91M model converged at 8.8B tokens; our 25M model will converge earlier.

**Phase 4a/4b (autocorrect finetune): as planned.**
- Synthetic typo triples + in-context corrupted sentences.
- This is where the FUTO-specific format is learned.

**Phase 4c (conversational adaptation): ADD THIS.**
- We were missing this stage! FUTO's own pipeline includes it and calls it
  "an important step."
- Use BERnaT BSM (~250M tokens) with low typo rate (~10%) for a short finetune.
- This shifts the model toward conversational register.
- This is exactly what the Portuguese guide's Phase 4c does, and what FUTO's wiki describes.

**Do NOT make BSM the pretrain base.** The pretrain should be on diverse, clean,
standard Basque (Latxa). BSM is for the final adaptation stage. Rationale:
1. FUTO's English model pretrains on SlimPajama (web), not chat.
2. BSM has noise/dialect/argot that would pollute the tokenizer and base LM.
3. The tokenizer trains on clean text only (already our approach).
4. A small finetune at the end is more effective than mixing throughout.

#### 11.6.6 Why our mini model already works

The mini model (525M tokens, 2000 steps) produces reasonable predictions because:
1. **Basque is morphologically regular** — the 4K tokenizer captures morphemes, and the
   agglutinative structure means patterns generalize well.
2. **525M tokens = 21:1 ratio** for 25M params — already at Chinchilla optimal!
3. **Next-word prediction is easier than general LM** — the model only needs to predict
   the most likely continuation, not understand deep semantics.
4. **The Latxa corpus is high quality** — HiTZ-curated, already cleaned.

This doesn't mean 525M tokens is enough for production quality. The mini model predicts
common continuations but will fail on rare words, complex syntax, and autocorrect edge
cases. More data + the conversational finetune will improve all of these.

---

## 12. Recommended action plan for this project

1. **Reproduce the reference.** Run Phase 0: download `breadlicker45/futo-keyboard-lm`,
   dump metadata, extract the embedded SentencePiece, print pieces 4–303. This is the spec.
2. **Clone the Portuguese repo as a template** (`danmaxis/futo-portuguese`) and adapt its
   `scripts/0N_*.py` for Basque. Its tokenizer script (`02_train_tokenizer.py`) already
   implements the exact slot layout — swap the pt-BR word lists for Basque ones.
3. **Assemble a Basque corpus.** Two tiers (see §11.4): the *clean* tier reuses
   morpheus-mamba's `data/clean-v3/` (Latxa corpus v2, 11 HiTZ-curated sources, ~4.77 B
   tokens) staged into `corpora/clean/`; the *conversational* tier is BERnaT BSMtime
   (~250 M tokens), aggressively cleaned into `corpora/conversational/`. Train the
   tokenizer on the clean tier only; pretrain on clean tier only; use BSM only for
   Phase 4c conversational adaptation (see §11.6). Set up with `uv` per project
   conventions.
4. **Train the tokenizer** (32+ GiB RAM host), validate the `<CHAR_*>` sequentiality and
   structural-token presence.
5. **Pretrain** the Llama-25M base on a GPU (16–24 GiB). Target **2-3B tokens** (~80-120:1
   ratio, see §11.6), ~20-30k steps. Start with a 2k-step mini run on 500M tokens to
   validate the whole pipeline before committing 8-12h.
6. **Fine-tune (4a/4b/4c)** with Basque typo synthesis (accent/ñ loss is the dominant
   class — reuse the Portuguese `lib_typo_synthesis.py`). **Phase 4c (conversational
   adaptation)** uses BERnaT BSM at low typo rate (~10%) — this is the stage FUTO's own
   wiki calls "an important step" (see §11.6.1).
7. **Package** as GGUF v2 with `keyboardlm.*` metadata, `languages="eu"`, features
   `base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1`, Q6_K output,
   SentencePiece as `[UINT8]`. Downgrade to v2.
8. **Side-load** via FUTO → Languages & Models → Import from file → assign to Basque →
   enable Transformer LM. Debug crashes via `adb logcat` (see §10).
9. **Evaluate** with a ~30-item Basque autocorrect suite (accent-loss, ñ-loss,
   transpositions, common shortcuts) — aim toward the English reference's ~74% top-1.
10. *(Optional, later)* contribute a `eu_wordlist.combined.gz` dictionary, and/or enable
    `lora_finetunable_v1` for on-device personalization.

---

## 13. Key source files in `futo-org/android-keyboard`

| File | What it does |
|---|---|
| `java/src/org/futo/inputmethod/latin/xlm/ModelPaths.kt` | model storage, **import validation**, default model `ml4_q6_k`, supported feature set |
| `java/src/org/futo/inputmethod/latin/xlm/LanguageModel.kt` | Kotlin inference API (`getSuggestions`/`rescoreSuggestions`), merges transformer + dictionary |
| `java/src/org/futo/inputmethod/latin/xlm/AdapterTrainer.kt` | on-device LoRA trainer (Kotlin) |
| `java/src/org/futo/inputmethod/latin/xlm/TrainingDataGenerator.kt` | special-token + `<CHAR_*>` prompt format, synthetic misspelling generation |
| `java/src/org/futo/inputmethod/latin/uix/settings/pages/modelmanager/ModelList.kt` | the "Transformer Models" / "Import from file" UI |
| `native/jni/src/ggml/LanguageModel.h/.cpp` | `LlamaAdapter` (llama.cpp wrapper), `LLAMA_CONTEXT_SIZE=2048`, feature constants |
| `native/jni/src/ggml/ModelMeta.h/.cpp` | `keyboardlm.*` metadata load/write |
| `native/jni/src/ggml/train.h` / `train.cpp` | llama.cpp training helpers (Adam, cosine schedule, checkpointing) |
| `native/jni/src/ggml/finetune.h` | LoRA `train_params` + `finetune_train` |
| `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp` | JNI inference: prompt construction, `PredictNextWord`/`PredictCorrection`, sampling, logit transforms |
| `native/jni/org_futo_inputmethod_latin_xlm_AdapterTrainer.cpp` | JNI LoRA training + apply + export merged GGUF |
| `native/jni/src/sentencepiece/*` | vendored SentencePiece library (BPE/char/unigram) |
| `dictionaries/*` | AOSP wordlists (no `eu` exists) |
| `tools/make-keyboard-text-py/locales/eu.json` | Basque keyboard layout (more-keys for á é í ó ú ü ñ ç) |

---

## 14. References

- **FUTO Keyboard site:** https://keyboard.futo.org/
- **Docs (Text Prediction):** https://docs.keyboard.futo.org/settings/textprediction.html
- **Docs (Languages & Models):** https://docs.keyboard.futo.org/settings/languagesmodels.html
- **Source (GitHub mirror):** https://github.com/futo-org/android-keyboard
- **Issue #1212 — Transformer Models for More Languages:** https://github.com/futo-org/android-keyboard/issues/1212
- **Reference English model (HF, community re-upload):** https://huggingface.co/breadlicker45/futo-keyboard-lm
- **Official Keyboard-LM wiki (was at `gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs`):** now returns "This page doesn't exist" — content preserved/verified via the guide below.
- **★ Community end-to-end guide + scripts (Portuguese) — primary template:** https://github.com/danmaxis/futo-portuguese/blob/main/GUIDE.md
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **Current app version at research time:** v0.1.29.1 (2026-06-22)

---

## Appendix: Phase 0 — Reference model verification (2026-07-14)

The reference English model (`breadlicker45/futo-keyboard-lm`, `ml4_1_f16_meta_fixed.gguf`,
62 MB Q6_K) was downloaded and inspected. The embedded SentencePiece tokenizer was
extracted and its proto parsed. **All structural assumptions above were confirmed.**
Three bugs in our ported tokenizer were caught and fixed:

| Parameter | Reference (verified) | Our port (before) | Status |
|---|---|---|---|
| `model_type` | **UNIGRAM** | BPE | ✅ fixed |
| `add_dummy_prefix` | **False** | True (default) | ✅ fixed |
| `remove_extra_whitespaces` | **False** | True (default) | ✅ fixed |
| `byte_fallback` | True | True | ✅ matched |
| `vocab_size` | 15008 | 15008 | ✅ matched |
| `character_coverage` | 0.9995 | 0.9995 | ✅ matched |
| `pad/bos/eos/unk` | 0/1/2/3 | 0/1/2/3 | ✅ matched |
| user_defined_symbols | 300 (IDs 4–303) | 300 (IDs 4–303) | ✅ matched |
| structural slots 174–207 | XBU/XBC/XEC/XC0-4/CHAR_A-Z | identical | ✅ **perfect match** |

**Why UNIGRAM over BPE for Basque:** Basque is strongly agglutinative (stacks suffixes:
*etxea* → *etxera* → *etxekoak*). UNIGRAM's probabilistic segmentation handles ambiguous
morpheme boundaries better than BPE's deterministic greedy longest-match. The reference
English model also uses UNIGRAM, so matching it is the safe choice.

**Why `add_dummy_prefix=False`:** SentencePiece by default prepends a dummy `▁` to input
text. The reference disables this. For the autocorrect format `<XBU>…<XBC>word<XEC>`, a
dummy prefix would inject an unwanted `▁` token before the corrected word, corrupting the
format the model must learn.

**Features string (reference):**
```
base_v1 inverted_space xbu_char_autocorrect_v1 xc0_swipe_typing_v1 char_embed_mixing_v1
```
Our v1 omits `xc0_swipe_typing_v1` (no swipe-typing training data; the `<XC0>`–`<XC4>`
tokens remain in the vocab but are unused). Can be added in a future version if we train
with swipe data.

**Notes artifacts generated:**
- `notes/reference_first_64_tokens.txt` — first 64 token IDs
- `notes/reference_special_tokens.txt` — all 260 non-NORMAL tokens
- `notes/reference_full_features.txt` — the features string
- `notes/reference_slot_map.md` — annotated 300-slot user-defined-symbol map
- `reference_model/extracted_spm.model` — the reference SentencePiece model (474 KB)

---

docs, the reference GGUF on HuggingFace, and the community Portuguese training guide;
cross-checking the guide's claims against the actual C++ source where it cites line
numbers. All architectural and metadata details above are facts about how the app works,
verified against the source.*
