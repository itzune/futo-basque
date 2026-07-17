#!/usr/bin/env bash
#
# run_server.sh — GPU-server runbook for futo-basque, Phases 1 → 5.
#
# All data-strategy decisions live in configs/*.yaml (see configs/README.md).
# This script just wires the right --config + --mode to each phase.
#
# REUSES morpheus-mamba's already-downloaded + already-cleaned corpus (no
# re-downloading from HuggingFace):
#   • Clean tier (tokenizer + pretrain base):
#       <MORPHEUS_DIR>/data/clean-v3/*.txt   — 11 Latxa v2 sources, ~4.77 B tokens
#   • Conversational tier (Phase 4c ONLY — not in pretrain per §11.6):
#       BERnaT BSMtime from HuggingFace (HiTZ/BERnaT-Diverse)
#
# Usage:
#   ./run_server.sh mini            # pipeline validation (~1-2h: 500M tok, 2k steps)
#   ./run_server.sh full            # real run (~8-12h: 3B tok, 24k pretrain steps)
#   ./run_server.sh full 1 1b 2 3   # run only corpus + tokenizer + pretrain
#   ./run_server.sh mini 4a 4b 4c 5 # run only finetune + package (resume)
#   ./run_server.sh full 4c         # run only Phase 4c (conversational adaptation)
#
# Phases:
#   1   = clean corpus (Latxa v2)     1b  = BSM cleaning
#   2   = tokenizer                    3   = pretrain (clean tier ONLY)
#   4a  = isolated autocorrect         4b  = in-context autocorrect
#   4c  = conversational adaptation    4d  = clean-text recovery (legacy)
#   4m  = unified multi-task finetune (NEW — replaces 4a/4b/4c/4d)
#   5   = package to GGUF
#
# Env overrides:
#   MORPHEUS_DIR=/root/morpheus-mamba   path to a morpheus-mamba checkout
#   WANDB_API_KEY=...                   or run `wandb login` once beforehand
#   WANDB_PROJECT=futo-eu               set empty to disable wandb logging
#   LLAMA_CPP=/root/llama.cpp           path to llama.cpp clone (Phase 5)
#
set -euo pipefail
cd "$(dirname "$0")"

# ─── paths ─────────────────────────────────────────────────────────────── #
MORPHEUS_DIR="${MORPHEUS_DIR:-../morpheus-mamba}"
CLEAN_OUT="corpora/clean"
CONV_OUT="corpora/conversational"
TOKENIZER="tokenizer/spm_eu.model"
PRETRAIN_OUT="pretrain"
STAGE_A="finetune/stage_a/final"
STAGE_B="finetune/stage_b/final"
STAGE_C="finetune/stage_c/final"
STAGE_D="finetune/stage_d/final"
STAGE_M="finetune/stage_m/final"
LLAMA_CPP="${LLAMA_CPP:-/root/llama.cpp}"
WANDB_PROJECT="${WANDB_PROJECT:-futo-eu}"

# ─── config files (the single source of truth for all decisions) ───────── #
CFG_PHASE1="configs/phase1_corpus.yaml"
CFG_PHASE1B="configs/phase1b_bernat.yaml"
CFG_PHASE2="configs/phase2_tokenizer.yaml"
CFG_PHASE3="configs/phase3_pretrain.yaml"
CFG_PHASE4A_DP="configs/phase4a_dataprep.yaml"
CFG_PHASE4A="configs/phase4a_isolated.yaml"
CFG_PHASE4B="configs/phase4b_fulltext.yaml"
CFG_PHASE4C="configs/phase4c_conversational.yaml"
CFG_PHASE4D="configs/phase4d_recovery.yaml"
CFG_PHASE4M="configs/phase4_multitask.yaml"
CFG_PHASE5="configs/phase5_package.yaml"

# ─── mode (mini = smoke test, full = real run) ─────────────────────────── #
MODE="${1:-full}"
shift || true
if [ $# -gt 0 ]; then PHASES=("$@"); else PHASES=(1 1b 2 3 4m 5); fi

case "$MODE" in
  mini|full) ;;
  *) echo "error: mode must be 'mini' or 'full' (got '$MODE')" >&2; exit 1 ;;
esac

# ─── helpers ───────────────────────────────────────────────────────────── #
count_shards() { find "$1" -name 'shard_*.txt' 2>/dev/null | wc -l; }
has_phase() { [[ " ${PHASES[*]} " == *" $1 "* ]]; }

# ─── preflight ─────────────────────────────────────────────────────────── #
echo "═══ futo-basque server runbook — mode=$MODE phases=${PHASES[*]} ═══"
echo

# morpheus clean-v3 (needed for phase 1)
if has_phase 1; then
  if [ ! -d "$MORPHEUS_DIR/data/clean-v3" ]; then
    echo "✗ clean-v3 not found at $MORPHEUS_DIR/data/clean-v3" >&2
    echo "  Fix: MORPHEUS_DIR=/path/to/morpheus-mamba ./run_server.sh ..." >&2
    exit 1
  fi
  n=$(find "$MORPHEUS_DIR/data/clean-v3" -name '*.txt' | wc -l)
  echo "  ✓ morpheus clean-v3: $MORPHEUS_DIR/data/clean-v3 ($n source files)"
fi

# uv + deps
command -v uv >/dev/null || { echo "✗ uv not found on PATH"; exit 1; }
echo "  ✓ uv: $(uv --version 2>&1)"

# Determine if any GPU phase is running
NEEDS_GPU=false
for p in 3 4a 4b 4c 4d 4m 4mr; do has_phase "$p" && NEEDS_GPU=true; done

if [ "$NEEDS_GPU" = true ]; then
  echo "  syncing deps (train group: torch/transformers/accelerate/wandb)..."
  uv sync --group train
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  ✓ GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  else
    echo "✗ nvidia-smi not found — GPU phases (3, 4a, 4b, 4c) need CUDA" >&2
    exit 1
  fi
else
  uv sync
fi

# wandb (GPU phases)
if [ "$NEEDS_GPU" = true ] && [ -n "$WANDB_PROJECT" ]; then
  if [ -z "${WANDB_API_KEY:-}" ]; then
    if grep -q 'WANDB_API_KEY=' "$HOME/.bashrc" 2>/dev/null; then
      export WANDB_API_KEY=$(grep -oP 'WANDB_API_KEY=\K.*' "$HOME/.bashrc")
      echo "  ✓ wandb key loaded from ~/.bashrc"
    elif [ -f "$HOME/.netrc" ]; then
      export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc('$HOME/.netrc').hosts['api.wandb.ai'][2])" 2>/dev/null)
      echo "  ⚠ wandb key loaded from ~/.netrc (may be stale — prefer .bashrc)"
    fi
  fi
  if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    echo "  ⚠ WANDB_API_KEY not found — run 'wandb login' first, or 'export WANDB_PROJECT=' to disable." >&2
  fi
fi
echo

# ─── Phase 1: clean corpus ─────────────────────────────────────────────── #
phase1() {
  echo "━━━ Phase 1: stage clean corpus (Latxa v2) → $CLEAN_OUT ━━━"
  uv run python -m scripts.corpus.build_corpus \
    --config "$CFG_PHASE1" --mode "$MODE" \
    --morpheus-dir "$MORPHEUS_DIR" \
    --out "$CLEAN_OUT"
  echo "  ✓ $(count_shards "$CLEAN_OUT") shards in $CLEAN_OUT"
  echo
}

# ─── Phase 1b: BERnaT BSM (conversational) ─────────────────────────────── #
phase1b() {
  echo "━━━ Phase 1b: clean + stage BERnaT BSM (conversational) → $CONV_OUT ━━━"
  uv run python -m scripts.corpus.clean_bernat \
    --config "$CFG_PHASE1B" --mode "$MODE" \
    --from-hf \
    --out "$CONV_OUT"
  echo "  ✓ $(count_shards "$CONV_OUT") shards in $CONV_OUT"
  echo
}

# ─── Phase 2: tokenizer (clean tier ONLY) ──────────────────────────────── #
phase2() {
  echo "━━━ Phase 2: train tokenizer (clean tier ONLY) → $TOKENIZER ━━━"
  if [ "$(count_shards "$CLEAN_OUT")" -eq 0 ]; then
    echo "✗ no shards in $CLEAN_OUT — run phase 1 first" >&2; exit 1
  fi
  uv run python -m scripts.tokenizer.train \
    --config "$CFG_PHASE2" --mode "$MODE" \
    --corpus "$CLEAN_OUT" \
    --out "${TOKENIZER%.model}"
  echo "  ✓ tokenizer at $TOKENIZER"
  echo
}

# ─── Phase 3: pretrain (clean tier ONLY — BSM moved to Phase 4c) ───────── #
phase3() {
  echo "━━━ Phase 3: pretrain (clean tier ONLY) → $PRETRAIN_OUT ━━━"
  if [ ! -f "$TOKENIZER" ]; then
    echo "✗ $TOKENIZER not found — run phase 2 first" >&2; exit 1
  fi
  if [ "$(count_shards "$CLEAN_OUT")" -eq 0 ]; then
    echo "✗ no shards in $CLEAN_OUT — run phase 1 first" >&2; exit 1
  fi
  echo "  NOTE: pretrain uses clean tier ONLY (BSM → Phase 4c, per §11.6)"
  local wb=()
  if [ -n "$WANDB_PROJECT" ]; then wb=(--wandb-project "$WANDB_PROJECT"); fi
  uv run python -m scripts.pretrain.train \
    --config "$CFG_PHASE3" --mode "$MODE" \
    --tokenizer "$TOKENIZER" \
    --corpus "$CLEAN_OUT" \
    --out "$PRETRAIN_OUT" \
    "${wb[@]}"
  echo "  ✓ base checkpoint at $PRETRAIN_OUT/base/"
  echo
}

# ─── Phase 4a: isolated autocorrect ────────────────────────────────────── #
phase4a() {
  echo "━━━ Phase 4a: isolated autocorrect finetune → $STAGE_A ━━━"
  if [ ! -d "$PRETRAIN_OUT/base" ]; then
    echo "✗ $PRETRAIN_OUT/base not found — run phase 3 first" >&2; exit 1
  fi
  # Data prep: wordfreq + triples
  if [ ! -f "notes/wordfreq.json" ]; then
    echo "  building wordfreq from $CLEAN_OUT..."
    uv run python -m scripts.finetune.build_wordfreq \
      --corpus "$CLEAN_OUT" --out notes/wordfreq.json
  fi
  uv run python -m scripts.finetune.generate_triples \
    --config "$CFG_PHASE4A_DP" --mode "$MODE" \
    --wordfreq notes/wordfreq.json \
    --out-synth notes/synth.json \
    --out-real notes/real.json
  # Train
  local wb=()
  if [ -n "$WANDB_PROJECT" ]; then wb=(--wandb-project "$WANDB_PROJECT"); fi
  uv run python -m scripts.finetune.isolated \
    --config "$CFG_PHASE4A" --mode "$MODE" \
    --base "$PRETRAIN_OUT/base" \
    --tokenizer "$TOKENIZER" \
    --synth-jsonl notes/synth.json \
    --real-jsonl notes/real.json \
    "${wb[@]}"
  echo "  ✓ Stage A checkpoint at $STAGE_A/"
  echo
}

# ─── Phase 4b: in-context autocorrect ──────────────────────────────────── #
phase4b() {
  echo "━━━ Phase 4b: in-context autocorrect finetune → $STAGE_B ━━━"
  if [ ! -d "$STAGE_A" ]; then
    echo "✗ $STAGE_A not found — run phase 4a first" >&2; exit 1
  fi
  if [ "$(count_shards "$CLEAN_OUT")" -eq 0 ]; then
    echo "✗ no shards in $CLEAN_OUT — run phase 1 first" >&2; exit 1
  fi
  local wb=()
  if [ -n "$WANDB_PROJECT" ]; then wb=(--wandb-project "$WANDB_PROJECT"); fi
  uv run python -m scripts.finetune.fulltext \
    --config "$CFG_PHASE4B" --mode "$MODE" \
    --base "$STAGE_A" \
    --tokenizer "$TOKENIZER" \
    --corpus "$CLEAN_OUT" \
    "${wb[@]}"
  echo "  ✓ Stage B checkpoint at $STAGE_B/"
  echo
}

# ─── Phase 4c: conversational adaptation (NEW) ─────────────────────────── #
phase4c() {
  echo "━━━ Phase 4c: conversational adaptation (BERnaT BSM) → $STAGE_C ━━━"
  if [ ! -d "$STAGE_B" ]; then
    echo "✗ $STAGE_B not found — run phase 4b first" >&2; exit 1
  fi
  if [ "$(count_shards "$CONV_OUT")" -eq 0 ]; then
    echo "✗ no conversational shards in $CONV_OUT — run phase 1b first" >&2; exit 1
  fi
  echo "  FUTO wiki calls this 'an important step' — shifts register toward chat (§11.6)"
  local wb=()
  if [ -n "$WANDB_PROJECT" ]; then wb=(--wandb-project "$WANDB_PROJECT"); fi
  uv run python -m scripts.finetune.conversational \
    --config "$CFG_PHASE4C" --mode "$MODE" \
    --base "$STAGE_B" \
    --tokenizer "$TOKENIZER" \
    --corpus "$CONV_OUT" \
    "${wb[@]}"
  echo "  ✓ Stage C checkpoint at $STAGE_C/"
  echo
}

# ─── Phase 4d: clean-text recovery (NEW) ────────────────────────────────── #
phase4d() {
  echo "━━━ Phase 4d: clean-text recovery (fix format contamination) → $STAGE_D ━━━"
  if [ ! -d "$STAGE_C" ]; then
    echo "✗ $STAGE_C not found — run phase 4c first" >&2; exit 1
  fi
  if [ "$(count_shards "$CLEAN_OUT")" -eq 0 ]; then
    echo "✗ no shards in $CLEAN_OUT — run phase 1 first" >&2; exit 1
  fi
  if [ ! -f "notes/synth.json" ] || [ ! -f "notes/real.json" ]; then
    echo "✗ notes/synth.json or notes/real.json not found — run phase 4a first" >&2; exit 1
  fi
  echo "  Fixes 100% format contamination: trains on 75% plain text + 25% triples"
  echo "  to break 'plain text → <XBU>' association while preserving autocorrect"
  local wb=()
  if [ -n "$WANDB_PROJECT" ]; then wb=(--wandb-project "$WANDB_PROJECT"); fi
  uv run python -m scripts.finetune.recovery \
    --config "$CFG_PHASE4D" --mode "$MODE" \
    --base "$STAGE_C" \
    --tokenizer "$TOKENIZER" \
    --corpus "$CLEAN_OUT" \
    --synth-jsonl notes/synth.json \
    --real-jsonl notes/real.json \
    "${wb[@]}"
  echo "  ✓ Stage D checkpoint at $STAGE_D/"
  echo
}

# ─── Phase 4m: unified multi-task finetune (NEW — replaces 4a/4b/4c/4d) ──── #
phase4m() {
  echo "━━━ Phase 4m: unified multi-task finetune (from pretrain) → $STAGE_M ━━━"
  if [ ! -d "$PRETRAIN_OUT/base" ]; then
    echo "✗ $PRETRAIN_OUT/base not found — run phase 3 first" >&2; exit 1
  fi
  if [ "$(count_shards "$CLEAN_OUT")" -eq 0 ]; then
    echo "✗ no shards in $CLEAN_OUT — run phase 1 first" >&2; exit 1
  fi
  if [ ! -f "notes/synth.json" ] || [ ! -f "notes/real.json" ]; then
    echo "✗ notes/synth.json or notes/real.json not found — run phase 4a dataprep first" >&2; exit 1
  fi
  echo "  Unified multi-task: 60% plain text (PLW=1.0) + 40% isolated triples"
  echo "  Starts from PRETRAIN (loss 4.33) — avoids format contamination"
  local wb=()
  if [ -n "$WANDB_PROJECT" ]; then wb=(--wandb-project "$WANDB_PROJECT"); fi
  uv run python -m scripts.finetune.multitask \
    --config "$CFG_PHASE4M" --mode "$MODE" \
    --base "$PRETRAIN_OUT/base" \
    --tokenizer "$TOKENIZER" \
    --corpus "$CLEAN_OUT" \
    --synth-jsonl notes/synth.json \
    --real-jsonl notes/real.json \
    "${wb[@]}"
  echo "  ✓ Stage M checkpoint at $STAGE_M/"
  echo
}

# ─── Phase 4mr: multi-task RECOVER (restart from checkpoint w/ lower LR) ── #
phase4mr() {
  local base_dir="finetune/stage_m"
  local ckpt="$base_dir/checkpoint-4000"
  if [ ! -d "$ckpt" ]; then
    ckpt="$base_dir/checkpoint-2000"
    echo "  ⚠ checkpoint-4000 not found, trying $ckpt" >&2
  fi
  if [ ! -d "$ckpt" ]; then
    echo "✗ no checkpoint in $base_dir — run phase 4m first" >&2; exit 1
  fi
  echo "━━━ Phase 4mr: multi-task RECOVER from $ckpt → $STAGE_M ━━━"
  echo "  Lower LR (2e-5) + explicit grad clipping — fixes loss spike"
  if [ "$(count_shards "$CLEAN_OUT")" -eq 0 ]; then
    echo "✗ no shards in $CLEAN_OUT — run phase 1 first" >&2; exit 1
  fi
  if [ ! -f "notes/synth.json" ] || [ ! -f "notes/real.json" ]; then
    echo "✗ notes/synth.json or notes/real.json not found — run phase 4a dataprep first" >&2; exit 1
  fi
  local wb=()
  if [ -n "$WANDB_PROJECT" ]; then wb=(--wandb-project "$WANDB_PROJECT"); fi
  uv run python -m scripts.finetune.multitask \
    --config "$CFG_PHASE4M" --mode recover \
    --base "$ckpt" \
    --tokenizer "$TOKENIZER" \
    --corpus "$CLEAN_OUT" \
    --synth-jsonl notes/synth.json \
    --real-jsonl notes/real.json \
    --progress-log finetune/stage_m/progress_recover.log \
    "${wb[@]}"
  echo "  ✓ Stage M (recovered) checkpoint at $STAGE_M/final/"
  echo
}

# ─── Phase 5: package to GGUF ──────────────────────────────────────────── #
phase5() {
  echo "━━━ Phase 5: package to FUTO-compatible GGUF ━━━"
  local ckpt="$STAGE_M"
  if [ ! -d "$ckpt" ]; then
    ckpt="$STAGE_D"
    echo "  ⚠ $STAGE_M not found — falling back to $STAGE_D (legacy recovery)" >&2
  fi
  if [ ! -d "$ckpt" ]; then
    ckpt="$STAGE_C"
    echo "  ⚠ $STAGE_D not found — falling back to $STAGE_C (skipping 4d)" >&2
  fi
  if [ ! -d "$ckpt" ]; then
    ckpt="$STAGE_B"
    echo "  ⚠ $STAGE_C not found — falling back to $STAGE_B (skipping 4c/4d)" >&2
  fi
  if [ ! -d "$ckpt" ]; then
    echo "✗ no finetune checkpoint found — run phase 4m (or 4a/4b/4c) first" >&2; exit 1
  fi
  if [ ! -f "$TOKENIZER" ]; then
    echo "✗ $TOKENIZER not found — run phase 2 first" >&2; exit 1
  fi
  if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    echo "✗ convert_hf_to_gguf.py not found in $LLAMA_CPP" >&2
    echo "  Fix: LLAMA_CPP=/path/to/llama.cpp ./run_server.sh ..." >&2
    exit 1
  fi
  uv run python -m scripts.package.to_gguf \
    --config "$CFG_PHASE5" --mode "$MODE" \
    --checkpoint "$ckpt" \
    --tokenizer "$TOKENIZER" \
    --llama-cpp "$LLAMA_CPP"
  echo
}

# ─── dispatch ──────────────────────────────────────────────────────────── #
for p in "${PHASES[@]}"; do
  case "$p" in
    1)  phase1  ;;
    1b) phase1b ;;
    2)  phase2  ;;
    3)  phase3  ;;
    4a) phase4a ;;
    4b) phase4b ;;
    4c) phase4c ;;
    4d) phase4d ;;
    4m) phase4m ;;
    4mr) phase4mr ;;
    5)  phase5  ;;
    *)  echo "unknown phase '$p' (use: 1 1b 2 3 4a 4b 4c 4d 4m 4mr 5)" >&2; exit 1 ;;
  esac
done

# ─── summary ───────────────────────────────────────────────────────────── #
echo "═══ Done — mode=$MODE phases=${PHASES[*]} ═══"
echo "  clean shards:        $(count_shards "$CLEAN_OUT")  ($CLEAN_OUT)"
echo "  conversational:      $(count_shards "$CONV_OUT")  ($CONV_OUT)"
echo "  tokenizer:           $([ -f "$TOKENIZER" ] && echo yes || echo no)  ($TOKENIZER)"
echo "  base checkpoint:     $([ -d "$PRETRAIN_OUT/base" ] && echo yes || echo no)  ($PRETRAIN_OUT/base)"
echo "  stage A (4a):        $([ -d "$STAGE_A" ] && echo yes || echo no)  ($STAGE_A)"
echo "  stage B (4b):        $([ -d "$STAGE_B" ] && echo yes || echo no)  ($STAGE_B)"
echo "  stage C (4c):        $([ -d "$STAGE_C" ] && echo yes || echo no)  ($STAGE_C)"
echo "  stage D (4d):        $([ -d "$STAGE_D" ] && echo yes || echo no)  ($STAGE_D)"
echo "  stage M (4m):        $([ -d "$STAGE_M" ] && echo yes || echo no)  ($STAGE_M)"
# Show GGUF output path from config
gguf_out=$(uv run python -c "from scripts.lib.runconfig import load_config, pick; cfg=load_config('$CFG_PHASE5','$MODE'); print(pick(None,cfg,'out','gguf/eu_futo_v2.gguf'))" 2>/dev/null || echo "gguf/eu_futo_v2.gguf")
echo "  GGUF:                $([ -f "$gguf_out" ] && echo yes || echo no)  ($gguf_out)"
echo
if has_phase 5; then
  echo "Next: side-load $gguf_out via FUTO → Languages & Models → Import from file."
fi
