# Configuration files

Every data-strategy decision lives in a declarative YAML file here, not in
script defaults or shell flags. This makes the decisions version-controlled,
diffable, and reviewable.

## Files

| File | Phase | What it declares |
|------|-------|-----------------|
| `datasets.yaml` | — | Dataset registry: named tiers (clean / conversational), paths, sources, cleaning params |
| `phase1_corpus.yaml` | 1 | Clean tier staging: token budget, shard size |
| `phase1b_bernat.yaml` | 1b | BSM cleaning: HF source, token budget |
| `phase2_tokenizer.yaml` | 2 | Tokenizer: vocab size, SP params, which corpus |
| `phase3_pretrain.yaml` | 3 | Pretrain: architecture, hyperparams, steps |
| `phase4a_dataprep.yaml` | 4a prep | Triple generation: n_synth, output paths |
| `phase4a_isolated.yaml` | 4a | Isolated autocorrect: mix ratio, PLW, steps |
| `phase4b_fulltext.yaml` | 4b | In-context autocorrect: typo rate, PLW, steps |
| `phase4c_conversational.yaml` | 4c | **Conversational adaptation**: BSM, low LR, steps |
| `phase4d_recovery.yaml` | 4d | **Clean-text recovery (legacy)**: tried to fix contamination from stage_c — failed |
| `phase4_multitask.yaml` | 4m | **Unified multi-task finetune (NEW)**: replaces 4a→4b→4c. 60/40 plain+triples from pretrain |
| `phase5_package.yaml` | 5 | GGUF packaging: features, metadata, output path |

## Mode system

Each config has a `modes:` section with `mini:` and `full:` overrides:

```yaml
total_steps: 24000          # full-mode default
lr: 3.0e-4
modes:
  mini:
    total_steps: 2000       # override for smoke-test runs
```

Pass `--mode mini` or `--mode full` to each script. CLI args always override
config values (for quick experiments).

## Usage

```bash
# With a config file (canonical — all decisions in YAML):
uv run python -m scripts.pretrain.train \
    --config configs/phase3_pretrain.yaml --mode full \
    --tokenizer tokenizer/spm_eu.model

# With CLI overrides (quick experiment):
uv run python -m scripts.pretrain.train \
    --config configs/phase3_pretrain.yaml --mode full \
    --tokenizer tokenizer/spm_eu.model \
    --total-steps 10000     # overrides config's 24000

# Without a config (pure CLI — backwards compatible):
uv run python -m scripts.pretrain.train \
    --tokenizer tokenizer/spm_eu.model --corpus corpora/clean \
    --total-steps 2000
```

The runbook (`run_server.sh`) passes `--config` and `--mode` to every phase.

## Key decisions encoded here

See [RESEARCH.md](../RESEARCH.md) §11.6 for the full analysis. Summary:

1. **Pretrain on clean tier only** (not BSM) — matches FUTO's own English
   pipeline (SlimPajama pretrain → conversational finetune).
2. **3B tokens, not 5B** — our 25M model's sweet spot is 80-120:1 ratio
   (between Chinchilla 20:1 and MiniCPM 192:1).
3. **Phase 4c added** — conversational adaptation on BERnaT BSM, which FUTO's
   wiki calls "an important step" and we were missing.
4. **Vocab 4096** — morpheme splitting for agglutinative Basque (§11.3.1).
5. **BSM excluded from tokenizer** — keeps UNIGRAM vocab morpheme-focused.
