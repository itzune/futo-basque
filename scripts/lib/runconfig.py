"""
Lightweight YAML run-config loader with mini/full mode support.

Every data-strategy decision (which corpus, what token budget, how many
steps, typo rate, learning rate, …) lives in a declarative YAML file under
``configs/`` so the decisions are version-controlled and diffable — not
buried in script defaults or shell flags.

Config file structure
---------------------
Each ``configs/phaseN_*.yaml`` file has flat top-level keys (the defaults)
plus an optional ``modes:`` section whose ``mini:`` / ``full:`` sub-mappings
override those defaults for the given mode::

    # configs/phase3_pretrain.yaml
    dataset: clean          # clean tier only (BSM → Phase 4c, see §11.6)
    total_steps: 24000      # full-mode default
    lr: 3.0e-4
    modes:
      mini:
        total_steps: 2000   # override for smoke-test runs
        lr: 3.0e-4

``load_config("configs/phase3_pretrain.yaml", "mini")`` returns a flat dict
with mode overrides applied: ``{"dataset": "clean", "total_steps": 2000, "lr": 3.0e-4}``.

Usage in scripts
----------------
Scripts add ``--config`` and ``--mode`` args, set all other argparse defaults
to ``None``, then resolve each value with :func:`pick` (CLI > config > hard
default)::

    from scripts.lib.runconfig import load_config, pick

    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", default="full", choices=["mini", "full"])
    ap.add_argument("--total-steps", type=int, default=None)   # None = use config
    args = ap.parse_args()

    cfg = load_config(args.config, args.mode)
    total_steps = pick(args.total_steps, cfg, "total_steps", 150000)

This keeps CLI overrides working (for quick experiments) while making the
*canonical* decisions live in YAML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None, mode: str = "full") -> dict[str, Any]:
    """Load a YAML config and apply ``modes[mode]`` overrides.

    Returns an empty dict if *path* is ``None`` (config is always optional —
    scripts still work with pure CLI args).
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    # Start with all top-level keys except "modes".
    cfg: dict[str, Any] = {k: v for k, v in raw.items() if k != "modes"}

    # Deep-merge mode-specific overrides on top (mode keys win).
    modes = raw.get("modes", {})
    if mode in modes and isinstance(modes[mode], dict):
        cfg = _deep_merge(cfg, modes[mode])

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override values win)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def pick(cli_value: Any, cfg: dict, key: str, hard_default: Any = None) -> Any:
    """Resolve a parameter: **CLI arg > config file > hardcoded default**.

    ``cli_value`` should be ``None`` when the user did not pass the flag
    (i.e. the argparse default is ``None``), so we fall through to the config.
    """
    if cli_value is not None:
        return cli_value
    if key in cfg:
        return cfg[key]
    return hard_default
