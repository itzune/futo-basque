# FUTO-Basque vs Morpheus: Comparative Analysis & FUTO Engineering Review

> **Purpose:** (1) Compare futo-basque and morpheus for next-word/token prediction.
> (2) Deep-review FUTO Keyboard's transformer engineering strategies and identify
> what could be applied to morpheus.
>
> Date: 2026-07-15. Based on FUTO source (`JNI_LanguageModel.cpp`, `LanguageModel.kt`,
> `ModelMeta.cpp`), the morpheus writeup, and both projects' eval artifacts.

---

## Part 1 — Head-to-Head: Which is Better at Next-Word/Token Prediction?

### 1.1 The two models at a glance

| Dimension | futo-basque | morpheus |
|-----------|-------------|----------|
| **Architecture** | Llama transformer (MHA, no GQA) | Mamba-2 SSM |
| **Parameters** | 25M | 91M (3.6×) |
| **Tokenizer** | SP UNIGRAM, vocab=4096 | SP UNIGRAM, vocab=4000 |
| **Context** | 2048 max (trained @ seq_len 1024) | 1024 (SSM state = fixed ~16KB/layer) |
| **Training data** | 3B tokens (Latxa v2) + finetune | ~10B tokens (~2.16 epochs of 4.62B corpus) |
| **Tokens:params ratio** | ~120:1 | ~110:1 |
| **Deployment format** | GGUF v2, Q6_K, 49MB | GGUF Q4_K_M 55MB / Q5_K_M 66MB |
| **Deployment target** | FUTO Keyboard app (Android) | Custom demo server (Docker/WASM) |
| **Primary capability** | **Autocorrect** (keypress format) + next-word | **Next-word / ghost-text** continuation |
| **Inference layer** | FUTO app's C++ (banned tokens, beam, confidence) | Custom (retokenization fallback, sticky merge) |

### 1.2 Raw token prediction (PPL / loss)

The vocabularies are nearly identical (4096 vs 4000), so per-token loss/PPL is **roughly comparable**:

| Model | Loss | PPL | Notes |
|-------|------|-----|-------|
| **morpheus** | **1.964** (valid) | **7.13** | Held-out validation, converged |
| **futo-basque** | 4.33 (train) | ~76 | Training loss at step 24K; no valid PPL measured |

**Morpheus is dramatically better at raw token prediction** (~10× lower PPL). This is expected from scaling laws: morpheus has 3.6× more parameters and 3.3× more training data. The tokens:params ratios are similar (~110-120:1), so both are "appropriately trained" for their size — morpheus is simply a bigger, better LM.

> **Caveat:** futo-basque's 4.33 is *training* loss (validation PPL wasn't measured), so the real gap may be slightly different, but the order of magnitude is clear.

### 1.3 Next-word prediction

| Metric | futo-basque | morpheus | Winner |
|--------|-------------|----------|--------|
| **Next-word top-1** | **8.3%** (1/12) | **50.3%** (75/149) | morpheus (6×) |
| **Next-word top-3 (acceptance)** | — | 85.2% (127/149) | morpheus |
| **Next-word top-5** | — | 85.9% (128/149) | morpheus |
| **NW-CSR (with inference eng.)** | — | 0.40 | morpheus |
| **Word accuracy (no IE)** | — | 60.4% (90/149) | morpheus |

**Morpheus is dramatically better at next-word prediction.** But the comparison needs important context:

#### Why futo-basque's next-word is broken (8.3%)

Looking at the actual eval failures, futo-basque suffers severe **format overfit**. The model was finetuned so heavily on the `<XBU>…<XEC>` autocorrect format that in plain next-word mode (no `<XBU>` prompt) it still emits structural tokens:

| Prompt | futo-basque top-1 | What it should be |
|--------|-------------------|-------------------|
| `Ni euskara` | `ri<XEC>` | `ikasten` |
| `Zein da zure` | `<XBU><CHAR_E><CHAR_S>` | `izena` |
| `Bai, gustatu` | `!!!!` | `zait` |
| `Eskerrik asko` | `.` | `denagatik` |

The model predicts the *autocorrect format* instead of plain words. This is a known artifact of the 3-phase finetune (Phase 4a/4b teach the `<XBU>` format so aggressively that it bleeds into next-word mode). The PLW=0.05 mitigation in Phase 4b helped (mini was 0%; full is 8.3%) but didn't fully solve it.

#### Why the comparison isn't fully apples-to-apples

1. **Different eval sets:** futo-basque uses 12 hand-written tests with "plausible" gold (not exact-match). Morpheus uses 30 CSR sentences (149 words) with exact-match gold.
2. **Different inference stacks:** futo-basque's eval runs the model *directly* (greedy top-k, no app inference engineering). Morpheus's best numbers include its full inference engineering layer (retokenization fallback + sticky merge), which **quadruples** its raw CSR (0.094 → 0.40).
3. **Different paradigms:** futo-basque is an *autocorrect* model that also does next-word. Morpheus is a *pure LM* that does next-word via inference engineering.

Even accounting for all this, morpheus wins decisively. Its *simplified* next-word (no inference engineering, 60.4% word accuracy) already crushes futo-basque's 8.3%.

### 1.4 Autocorrect (where futo-basque shines)

| Metric | futo-basque | morpheus |
|--------|-------------|----------|
| **Autocorrect top-1** | **82.5%** (33/40) | N/A (doesn't do autocorrect) |
| **Autocorrect top-5** | **95.0%** (38/40) | N/A |

futo-basque's designed purpose is autocorrect — given a typo (`kaixp`), predict the correct word (`kaixo`). It does this well (82.5%). **Morpheus has no autocorrect capability at all.** It does word *completion* (prefix → full word), not word *correction* (typo → correct word). These are fundamentally different tasks.

### 1.5 Verdict

| Task | Winner | Margin |
|------|--------|--------|
| Raw token prediction (PPL) | **morpheus** | ~10× lower PPL |
| Next-word prediction | **morpheus** | 6× higher top-1 (50% vs 8%) |
| Autocorrect (typo → correct) | **futo-basque** | morpheus doesn't do this |
| Inference latency (on-device) | **morpheus** | O(1) SSM, no KV cache; 97ms on 2017 laptop |
| Deployment reach | **futo-basque** | Runs inside FUTO Keyboard (real Android app) |

**For next-word/token prediction, morpheus is unambiguously better** — it's a bigger, better-trained language model with a sophisticated inference engineering layer. futo-basque traded next-word quality for autocorrect capability (the FUTO format requirement). The two models solve *different problems*: morpheus predicts what comes next; futo-basque corrects what you typed.

---

## Part 2 — FUTO Engineering Strategies: Deep Review

FUTO's C++ inference layer (`JNI_LanguageModel.cpp`, ~1300 lines) is a masterclass in
production keyboard-LM engineering. Below, each strategy is analyzed for applicability
to morpheus. Strategies are ordered by impact.

### Strategy 1: Contextual Logit Banning (`transform_logits`) ⭐ HIGH IMPACT

**What FUTO does:** Before sampling, FUTO modifies the raw logits based on context:

- **Word-separator banning:** All tokens containing symbols (`.!@#$%^&*()…`) are banned
  (`-999.0f`), and their probability mass is **redirected to the SPACE token**. This
  forces the model to end words at spaces, not punctuation.
- **First-token-of-word banning:** Tokens starting with `'` or `-` are banned at word
  start (prevents `'-word` predictions).
- **Capitalization-aware banning:** If the user is typing `FirstCapital` or `AllCapitals`,
  lowercase-leading tokens are banned. This prevents suggesting `etxea` when the user
  typed `Etxe`.
- **General banning:** Known-bad tokens (e.g. `"-▁"`) are always banned.

**Applicability to morpheus:** **HIGH.** Morpheus's inference engineering (§5.4) handles
the tokenization trap with retokenization fallback, but does **no logit-level
constraining**. Adding capitalization-aware banning and symbol→space redirection would
directly improve candidate quality — morpheus currently can suggest lowercase
continuations of capitalized words, and punctuation-terminated tokens pollute the
candidate pool. This is a low-effort, high-yield addition to `src/eval_utils.py` and
the demo server.

### Strategy 2: Confidence-Based Prediction Modes ⭐ HIGH IMPACT

**What FUTO does:** FUTO classifies every prediction into one of three modes based on
the **ratio** between top-1 and top-2 probability:

```
if top1 > threshold × top2:        → "autocorrect" (auto-replace)
else if top1 > (threshold×0.1)×top2: → "uncertain" (show as suggestion)
else:                               → "clueless" (don't suggest)
```

This is a **calibration-free** heuristic — it uses relative probability ratios, not
absolute thresholds, so it doesn't need temperature tuning per model. "Clueless" mode
gets a massive score penalty (`probMult: 500000→10, probOffset: 100000→-100000`),
effectively hiding bad predictions.

**Applicability to morpheus:** **HIGH.** Morpheus's demo shows 3 chips but has no
notion of "should I even suggest?" — it always shows candidates. The confidence-ratio
heuristic would let morpheus suppress low-confidence predictions (reducing noise) and
potentially enable an autocorrect mode (auto-accept when top-1 >> top-2). This directly
addresses morpheus's future-work item about repetition loops and noisy completions
(§7.1.2) — "clueless" mode would hide them.

### Strategy 3: Two-Engine Merge (Dictionary + LM Rescoring) ⭐ HIGH IMPACT

**What FUTO does:** FUTO runs **two prediction engines in parallel** and merges:

1. **Classical AOSP dictionary + bigram engine** — high recall, knows the full
   wordlist, handles OOV via edit distance.
2. **Transformer LM** — high precision, context-aware.

The LM can **rescore the dictionary's candidates** (`rescoreSuggestions`): it takes the
dictionary's word list + scores, normalizes them, computes the LM probability of each
candidate's first token (÷ token count for length normalization), and reweights. This
combines dictionary recall with LM precision.

**Applicability to morpheus:** **HIGH — and this is morpheus's biggest gap.** Morpheus
is a pure LM with no dictionary fallback. For rare words, proper nouns, or OOV items the
LM doesn't know, morpheus simply fails. A lightweight Basque wordlist (frequency
dictionary) + LM rescoring would:
- Rescue OOV/rare words the LM misses (high recall)
- Let the LM's context re-rank them (high precision)
- Provide a fallback when the LM is "clueless"

This is arguably more valuable than morpheus's planned distillation (§7.2.3) for
improving real-world autocomplete hit rate. FUTO's rescoring formula
(`transformedScore × logits[first_token] / n_tokens`) is a simple, effective pattern.

### Strategy 4: The Keypress Autocorrect Format (`<XBU>…<XBC>…<XEC>`) ⭐ HIGH IMPACT (new capability)

**What FUTO does:** This is FUTO's signature innovation. Instead of word completion
(prefix → full word), FUTO does word **correction** (typed chars → correct word):

```
<context> <XBU> <CHAR_T><CHAR_E><CHAR_H> <XBC> The <XEC>
                   ↑ typed keystrokes          ↑ model predicts correction
```

Each typed character is encoded as a discrete `<CHAR_X>` token (accent-stripped,
uppercased via NFD). The model sees *exactly what was typed* and predicts what was
*meant*. This elegantly handles typos, transpositions, fat-finger errors.

**Applicability to morpheus:** **HIGH — would add a capability morpheus entirely lacks.**
Morpheus does word completion, not correction. If morpheus adopted this format via a
finetune (synthetic typo→correct pairs, like futo-basque's Phase 4a), it could do
*both* next-word prediction AND autocorrect. The 91M model has enough capacity to learn
both registers. This would make morpheus a complete keyboard LM, not just a predictor.

**Caveat:** This requires the `<CHAR_A>…<CHAR_Z>` structural tokens in the vocabulary
and a finetune dataset of typo→correct pairs. futo-basque's `typo_synthesis.py` and
`generate_triples.py` are directly reusable references.

### Strategy 5: Beam Search with Probability Products ⭐ MEDIUM IMPACT

**What FUTO does:** FUTO's `Sample()` function does proper beam search for multi-token
word candidates:

1. Keep `NUM_RESULTS=3` parallel sequences (each gets its own KV cache slot via
   `llama_kv_cache_seq_cp`).
2. At each step, extend each sequence with top-k tokens, multiply probabilities
   (`P(seq) = P(tok₁) × P(tok₂) × …`), re-sort, keep top-3.
3. Stop when a sequence hits `<XEC>` (end of correction) or a word boundary (space token).
4. Handles the case where multiple children come from the same parent (reassigns seq_id
   + copies KV cache).

**Applicability to morpheus:** **MEDIUM.** Morpheus uses greedy decoding + retokenization
fallback (parallel prefix queries). Beam search would produce better multi-token
completions (proper probability scoring vs. greedy), but:
- Mamba-2 has no KV cache to parallelize across beams (each beam needs its own SSM state)
- The retokenization fallback already handles the main failure mode (tokenization trap)
- Beam search adds latency (multiple forward passes)

For morpheus, a **cheaper variant** — top-k sampling at each step with early stopping at
word boundaries — would capture most of the benefit without full beam search.

### Strategy 6: Char Embedding Mixing (`char_embed_mixing_v1`) ◻ MEDIUM (mobile-only)

**What FUTO does:** FUTO doesn't just feed `<CHAR_T>` for a "T" keystroke — it takes the
**exact (x,y) screen coordinates** of the tap, decomposes them into proximity weights
across the 4 nearest keys (`ProximityInfo::decomposeTapPosition`), and **mixes the
embeddings** of those 4 `<CHAR_X>` tokens weighted by proximity. An `encoder_weight`
matrix can also map raw (x,y) → embedding (for swipe typing). This handles imprecise
touch typing — if you tap between T and R, the model sees a mix of both.

**Applicability to morpheus:** **LOW (current), MEDIUM (if mobile).** Morpheus's demo is
a text editor / browser keyboard, not a touch keyboard with coordinate data. But if
morpheus targets mobile (its §7.2.3 distillation goal), this is a brilliant strategy
for noisy touch input. The `encoder_weight` matrix approach (2D coordinates → embedding)
is more elegant than discrete token mixing.

### Strategy 7: Context Safeguarding (`safeguardContext`) ◻ LOW IMPACT

**What FUTO does:** Aggressively trims context before feeding the model:
- Max 70 chars or 16 words
- Trim to last sentence boundary (`.?!`)
- Then to last comma
- Then to last 5 words
- "Longest match" optimization: reuse previous context via suffix matching

**Applicability to morpheus:** **LOW.** Morpheus's Mamba-2 SSM state is **fixed-size**
(~16KB/layer) regardless of context length — there's no latency penalty for long
context. This is a key Mamba advantage (§5.2: "decode speed is context-length
independent"). FUTO needs context trimming because transformer KV cache grows linearly;
morpheus doesn't. However, the "recent context matters most" intuition is still valid
for quality — very old context can mislead. A soft version (trim to last 1-2 sentences)
could help quality without being a latency necessity.

### Strategy 8: On-Device LoRA Personalization (`AdapterTrainer`) ◻ MEDIUM (already planned)

**What FUTO does:** Fine-tunes the model **on the phone** from the user's typed text:
- LoRA r=16, α=16, 1 epoch, 128 iterations, 6 threads, n_ctx=64
- Training examples = user's typed text (trimmed + space-appended)
- Writes a full merged GGUF (LoRA applied + saved)
- Requires `lora_finetunable_v1` metadata flag

**Applicability to morpheus:** **MEDIUM — already in morpheus's future work (§7.3.5).**
Morpheus explicitly plans "on-device personalization from completion logs." FUTO's
implementation is a concrete reference: the hyperparameters (r=16, 128 iters, n_ctx=64)
are sensible defaults for a keyboard. Morpheus's completion logs (§5.4.6) already
capture (context, target) pairs — they're a ready-made personalization dataset. The
91M scale is large enough to meaningfully shift toward a single user's distribution
(15-40× larger than Gboard's federated LMs).

**Mamba-2 + LoRA feasibility:** LoRA adapts linear projections (q/k/v in transformers,
in_proj/out_proj in Mamba). Mamba-2's `in_proj` and `out_proj` are standard linears —
LoRA applies directly. No architectural barrier.

### Strategy 9: Personal Dictionary as Glossary Prompt (`addPersonalDictionary`) ◻ LOW IMPACT

**What FUTO does:** If the user has a personal dictionary, FUTO prepends a fake context:
```
(Glossary: word1, word2, word3)

<actual context>
```
The model sees these words and can suggest them. Zero-shot personalization — no
training needed.

**Applicability to morpheus:** **LOW-MEDIUM.** Simple and clever. For user-specific
terminology (names, jargon), prepending a glossary is a zero-cost way to bias
predictions. morpheus could add this to its demo server in ~10 lines. Less powerful
than LoRA personalization but zero-effort.

### Strategy 10: Exact-Match Boost + Length Sanity Check ◻ LOW IMPACT

**What FUTO does:**
- **Exact-match boost:** If a correction candidate exactly matches the typed partial word,
  boost it (subtract 1.0 from non-exact matches). Respects "user typed it right, keep it."
- **Length sanity:** If prediction is < half the typed word's length, force "clueless."
  Prevents absurd short predictions for long typed words.
- **Banned words:** User-configurable banned predictions, with hash-based wildcard
  sequence matching.

**Applicability to morpheus:** **LOW.** Simple sanity guards. The exact-match boost is
relevant for morpheus's word completion (if the model's continuation matches what the
user already typed, prefer it). The length sanity check prevents garbage. Trivial to add.

### Strategy 11: KV Cache Fast-Forward + Mix Caching ❌ N/A (architecture-specific)

**What FUTO does:** Caches the decoded prompt's KV state and only re-processes the delta
when context changes (`transformer_context_fastforward`). Also caches mixed embeddings
(`GetCachedMixAmount` checks which char-mixes haven't changed).

**Applicability to morpheus:** **N/A.** Mamba-2 has no KV cache — its SSM state is
fixed-size and updated incrementally. The concept ("don't recompute what hasn't
changed") is already inherent in Mamba's recurrent state. However, morpheus's demo
server could cache the SSM state across queries (if llama.cpp supports state
checkpointing for Mamba) to avoid re-prefilling the context each time.

---

## Part 3 — Summary: What Morpheus Should Adopt

### Tier 1: High-impact, low-effort (do first)

| # | Strategy | Effort | Why |
|---|----------|--------|-----|
| 1 | **Contextual logit banning** (capitalization, symbol→space) | Small | Directly improves candidate quality in `eval_utils.py` + demo |
| 2 | **Confidence-ratio modes** (autocorrect/uncertain/clueless) | Small | Suppresses noise, enables autocorrect mode, fixes repetition-loop UX |
| 3 | **Two-engine merge** (wordlist + LM rescoring) | Medium | Biggest gap — rescues OOV/rare words; FUTO's rescoring formula is reusable |

### Tier 2: High-impact, medium-effort (new capabilities)

| # | Strategy | Effort | Why |
|---|----------|--------|-----|
| 4 | **Keypress autocorrect format** (`<XBU>…<XEC>` + `<CHAR_X>`) | Large | Adds autocorrect capability morpheus entirely lacks; needs vocab slots + finetune |

### Tier 3: Already planned or situationally useful

| # | Strategy | Effort | Why |
|---|----------|--------|-----|
| 5 | On-device LoRA personalization | Medium | Already in morpheus §7.3.5; FUTO's hyperparameters are a reference |
| 6 | Char embedding mixing | Medium | Only if morpheus targets mobile touch keyboard |
| 7 | Beam search (lightweight variant) | Medium | Better multi-token completions; Mamba makes full beam costly |
| 8 | Glossary prompt / exact-match boost / length sanity | Small | Trivial sanity guards and zero-shot personalization |

### The reverse: what futo-basque should adopt from morpheus

For completeness — futo-basque's broken next-word (8.3%) could be fixed by adopting
morpheus's **inference engineering** (§5.4):

1. **Retokenization fallback** — query from progressively shorter prefixes to escape
   the tokenization trap (the reason futo-basque emits `<XEC>` is format overfit, but
   retokenization would also help genuine completion failures).
2. **Sticky merge** — carry forward previous candidates when switching from next-word
   to completion mode.
3. **A clean-text recovery finetune** — a small finetune on plain text (no `<XBU>`
   format) after Phase 4c, to teach the model the register shift between autocorrect
   mode and next-word mode.

---

## Part 4 — The Deeper Architectural Insight

The comparison reveals a fundamental **architecture-task alignment** question:

- **FUTO chose transformer + KV cache** because the keypress autocorrect format needs
  to attend to each `<CHAR_X>` token individually (the model must see *which keys were
  pressed*). The KV cache lets it incrementally process keystrokes. The cost: KV cache
  grows with context, requiring aggressive context trimming.

- **Morpheus chose Mamba-2 SSM** because next-word/ghost-text prediction needs constant
  latency over long sessions. The SSM state is fixed-size — context length doesn't
  affect decode speed. The cost: no easy way to do the per-keystroke attention that
  autocorrect needs (SSM compresses history into a fixed state, can't "look back" at
  individual keystrokes).

**The key question for morpheus:** Can Mamba-2 learn the `<XBU>…<CHAR_X>…<XEC>`
autocorrect format? Theoretically yes — the SSM state can encode "these keystrokes were
typed" — but it may struggle with long words (20+ keystrokes) where a transformer's
attention can directly attend to each `<CHAR_X>`. This is an **empirical question worth
testing** before committing to adding autocorrect to morpheus. A small experiment
(finetune morpheus on futo-basque's Phase 4a data) would answer it.

If Mamba-2 can learn the format, morpheus could become a **complete keyboard LM**
(next-word + autocorrect + O(1) latency) — strictly more capable than futo-basque in
every dimension except deployment reach (FUTO app vs custom demo).
