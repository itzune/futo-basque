# Keyboard Inference Engineering for Basque

This document preserves the inference-engineering strategies and evaluation
results developed for the **next-word keyboard paradigm** (whole-word
suggestion chips). These strategies were originally developed within the
Morpheus project (`~/Dev/itzune/morpheus-mamba`) and have been separated out
because the two projects target different interaction paradigms and
architectures:

| | Morpheus | futo-basque |
|---|---|---|
| **Architecture** | Mamba-2 (SSM, 91M) | Llama Transformer (25M) |
| **Paradigm** | Ghost-text autocomplete (desktop editor) | Next-word chips (mobile keyboard) |
| **Interaction** | Multi-token inline completion | Discrete whole-word suggestions |
| **Tokenization** | SentencePiece 4K (byte-fallback) | Llama BPE (32128) |
| **Deployment** | Obsidian plugin / web demo | FUTO Keyboard app |

The Morpheus write-up now documents the **ghost-text paradigm** strategies
(smart context, ghost suffix, digit-token repair, garbage filtering,
acceptance logging) in its §5.4. The strategies below are the **keyboard
paradigm** complement.

---

## 1. The Tokenization Trap

In an agglutinative language with a 4K subword vocabulary, the same word can
be reachable through multiple token paths — and the path the user's partial
input lands on may not reach the correct completion.

**Example:** The Basque word *Kaixo* ("hello") tokenizes as `[▁Ka, i, xo]`.
But when the user types *Kaix*, the tokenizer segments it as `[▁Ka, ix]` — a
different path that cannot reach the `xo` token. The model may know the word
perfectly, but the greedy continuation from `[▁Ka, ix]` produces *Kaixan*,
*Kaix-*, *Kaixko* — never *Kaixo*.

This is not a model deficiency; it is a structural artifact of subword
tokenization. The problem is especially acute in agglutinative languages
because long, morphologically complex words have many possible segmentation
paths, and short prefixes often land on the wrong one.

> **Note:** This strategy applies to the keyboard paradigm (word chips). The
> ghost-text paradigm (Morpheus §5.4) addresses a related but different
> problem: trailing subword fragments in the *context* (smart context
> stripping), not in the *completion target*.

---

## 2. Retokenization Fallback

**Strategy:** When generating word-completion candidates, query the model
from progressively shorter prefixes in parallel, then filter results by the
user's actual typed prefix.

For input *Kaix*, the system queries three paths simultaneously:

| Path | Prefix | Token IDs | Can reach *Kaixo*? |
|------|--------|-----------|---------------------|
| 0 | `Kaix` | `[▁Ka, ix]` | ✗ (wrong path) |
| 1 | `Kai` | `[▁Ka, i]` | ✓ (`xo` is reachable) |
| 2 | `Ka` | `[▁Ka]` | ✓ (but noisier) |

Path 1 reaches `[▁Ka, i]`, where the model predicts `xo` at 54.2% probability.
The result *Kaixo* passes the `startswith("Kaix")` filter and surfaces as a
candidate. All paths fire in parallel via `asyncio.gather`, keeping latency
at ~1× a single call rather than 3×.

A **from-scratch path** (empty prefix, predicting the next word from
preceding context only) rescues single-token whole words. For example,
*bezala* is a single token (`▁bezala`); when the user types *b*, the token
`▁b` cannot reach `▁bezala`. Querying from the preceding context (*"Ni ondo,
beti "*) surfaces *bezala* as a next-word prediction, which passes the
`startswith("b")` filter.

---

## 3. Sticky Merge: Candidate Carry-Forward

**Problem:** When the model predicts a next word (e.g., *izan* after
*idatzia*) and the user types the first letter (*i*), the system switches
from next-word prediction to word-completion mode. The token path changes
(`▁izan` is one token, but `▁i` + continuation is a different path), and the
previously good prediction vanishes from the candidate list — even though
the user is typing exactly the word that was predicted.

**Strategy:** Maintain a *sticky pool* of the previous render's candidates.
When new candidates arrive, filter the sticky pool by the current typed
prefix. Survivors are merged with fresh candidates, receiving a small
probability boost (+0.1) to compensate for the fact that cross-path
probabilities (next-word vs. word-completion) are not directly comparable.

```
State 1: "...idatzia" → [dago, dezakezu, daiteke, behar, izan]
  (izan at rank 5, not visible in top-3 chips, but stored in sticky pool)

State 2: "...idatzia i" → fresh: [iristeko, itxa, ikur, ...]
  Sticky survivor: izan (prob=0.087, boosted to 0.187)
  Merged result: [iristeko, izan ✓, itxa]
```

The sticky pool resets on chip acceptance and message send, preventing stale
candidates from persisting across word boundaries.

---

## 4. Top-k Exceeds Display-k

The keyboard displays 3 suggestion chips but fetches 5 candidates from the
server. The extra candidates populate the sticky pool, enabling
carry-forward of lower-ranked but relevant predictions. Without this, *izan*
(rank 5, prob=0.087) would never enter the sticky pool and could not be
rescued in State 2.

---

## 5. Next-Word Candidate Extraction

When the model's greedy continuation at a word-completion level begins with
a space (▁-prefixed token), it signals that the model considers the current
word complete and is predicting the *next* word. Rather than discarding
these tokens (as noise), we extract them as next-word candidates with an
`is_next_word` flag. The frontend handles these differently: inserting a
leading space before the word, matching the user's expectation that the
current word is finished.

This also handles the edge case where the user has typed a complete word
without a trailing space (e.g., *"Kaixo, zer"*): the model predicts *moduz*
as a next word, and the candidate appears despite no explicit word boundary.

---

## 6. Completion Logging and Replay

Every chip acceptance is logged to a JSONL file with: timestamp, model
checkpoint, context, smart context, accepted word and its probability, and
all candidates offered. This transforms real user sessions into an
evaluation dataset that can be replayed against any checkpoint:

```bash
python scripts/replay_completions.py --models step_0032000.Q4_K_M step_0054000.Q4_K_M
```

The replay script hot-reloads each checkpoint, queries the same contexts,
and checks whether the user-accepted word appears in the top-k.

The keyboard candidate algorithm (retokenization fallback, sticky merge,
top-k fetch, acceptance semantics) is also ported to PyTorch in
`src/eval_utils.py` (Morpheus repo) as `evaluate_next_word_csr`, enabling
training-time validation that faithfully reflects the deployed demo. This
runs natively on the GPU model (no llama.cpp dependency) during periodic
validation, reporting decomposed metrics (Top-1/Top-3/Top-5 accuracy,
acceptance rate, average prefix length, average confidence) alongside a
simulated CSR. It is used as a **secondary metric** — PPL remains primary
for checkpoint ranking — and the decomposed metrics avoid the CSR paradox
because they do not conflate model quality with morphological word length.

---

## 7. Evaluation: Next-Word CSR (NW-CSR)

The keyboard paradigm's headline result:

| Morpheus CSR | Simplified (raw) | Full Pipeline (with engineering) | Improvement |
|-------------|------------------|----------------------------------|-------------|
| Value | 0.094 | 0.362 | **3.9×** |

The full pipeline (retokenization fallback, sticky merge, top-k
alternatives, next-word extraction) improves CSR by **3.9×** over the raw
model. This is a larger engineering effect than the ghost-text paradigm
(+0.005, CSR-neutral) because the keyboard strategies change *which words
the model can reach* via multiple token paths, not just how they are
displayed.

### Comparison with Ghost-Text Paradigm

| Paradigm | Strategies | CSR effect | Why |
|----------|-----------|------------|-----|
| **Ghost-text** (Morpheus §5.4) | Smart context, ghost suffix, digit repair, garbage filter, confidence | +0.005 (CSR-neutral) | Cleans UX (no garbage) but junk never matches gold text anyway |
| **Keyboard** (this doc) | Retokenization fallback, sticky merge, top-k, next-word extraction | **3.9×** | Changes which words are reachable via multiple token paths |

The two paradigms share the same underlying model but require different
inference engineering. The keyboard paradigm's larger CSR effect reflects
the harder problem: producing discrete, complete words (which must navigate
the tokenization trap) rather than multi-token continuations (which can
ride the greedy path).

### Keyboard Simulation Metrics

Two variants:

1. **Frontend-faithful typing simulation** — char-by-char typing, sticky
   merge, top-3 chip display from a top-5 fetch, full acceptance semantics.
   15 sentences (5 Basque, 5 English, 5 Spanish) translated from the same
   semantic content.

2. **PyTorch-native port** (`evaluate_next_word_csr` in `src/eval_utils.py`)
   — runs during training validation on 30 Basque CSR test sentences,
   reporting:
   - **Top-1 accuracy** (was the correct word ever the #1 candidate?)
   - **Top-3 accuracy** (= acceptance rate, was it in the displayed chips?)
   - **Top-5 accuracy** (was it in the raw fetched pool?)
   - Average prefix length before acceptance
   - Average confidence

### futo-basque Results (25M Llama Transformer)

- 8.4% keystrokes saved (top-1)
- 28.9% keystrokes saved (top-5 bar)
- 50.0% next-word top-1 accuracy
- 41.7% next-word top-5 accuracy

---

## 8. Deployment Note: Mamba-2 `llama.cpp` Pinning

When deploying Mamba-2 models with `llama.cpp`, pin to a build that includes
commit `dc2187d48` (2025-07-04 or later). Earlier builds have a bug in the
SSM scan computation for Mamba-2 (`n_groups > 1`) that produces silently
incorrect greedy outputs — an apparent model regression (step 54K
"forgetting" *Kaixo*) was traced to a stale Docker cache running an older
build, not a model deficiency.

> This note applies to the Morpheus (Mamba-2) model. The futo-basque model
> is a standard Llama Transformer and does not require this pinning.

---

## Cross-References

- **Morpheus write-up §5.4** — ghost-text inference engineering (smart
  context, ghost suffix, digit-token repair, garbage filtering, confidence
  scoring)
- **Morpheus write-up §6.12** — CSR paradox (keyboard simulation findings)
- **Morpheus write-up §6.13** — metric inversion (keyboard simulation
  across checkpoints)
- **Morpheus `src/eval_utils.py`** — `evaluate_next_word_csr` PyTorch port
- **Morpheus `demo/server.py`** — `_keyboard_candidates()` endpoint
  (`/api/autocomplete/keyboard`), shared retokenization-fallback path
