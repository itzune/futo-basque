"""
Typo synthesis for the FUTO autocorrect format `<XBU>typo<XBC>correct<XEC>`.

Used by Phase 4a (isolated triples) and Phase 4b (in-context corrections).

Strategies (mixed per-word with weighted probabilities):
  1. Keyboard-adjacency typos using the standard QWERTY layout (the layout
     Basque speakers overwhelmingly use on mobile and desktop).
  2. Missing-diacritic / ñ loss: drop accents via NFD — covers ñ→n and the rare
     acute/ü loss in Basque names/dialects (e.g. Iñaki → Inaki).
  3. Transposed adjacent letters (kaixo → kaixo? → kabxo).
  4. Single-char insertion (random letter).
  5. Single-char deletion.
  6. Doubled char (e.g. kaixo → kaixoo).
  7. Known shortcut substitutions (config.eu.EU_SHORTCUTS) — these are *known*
     corrections, weighted higher than synthetic typos.

Note on Basque vs Portuguese: standard Basque (batua) barely uses acute
accents, so accent-drop is NOT the dominant typo class here (as it is for
Portuguese). Keyboard-adjacency + transposition/doubling dominate instead.
The synthesis is deterministic given a seed so we can regenerate the same
training set, and respects a `log(word_freq + 1)` dampener so common words
don't flood the dataset.
"""
from __future__ import annotations
import math
import random
import re
import unicodedata
from typing import Callable

from config.eu import EU_SHORTCUTS as SHORTCUTS

# Standard QWERTY adjacency map for the alpha rows. Each key maps to its
# left/right/up/down neighbours. Basque speakers use the standard (Spanish/
# international) QWERTY layout; the ñ key (right of L on the Spanish layout)
# is intentionally NOT in the map — ñ typos are handled by the accent-drop
# rule (ñ→n), which is the realistic path since ñ needs a separate keystroke.
ADJ = {
    "q": "wa12", "w": "qe23sa", "e": "wr34ds", "r": "et45fd", "t": "ry56gf",
    "y": "tu67hg", "u": "yi78jh", "i": "uo89kj", "o": "ip90lk", "p": "o0-l",
    "a": "qsz",   "s": "awxd",   "d": "secxf",  "f": "drcvg",  "g": "ftvbh",
    "h": "gybnj", "j": "hubmnk", "k": "jimol",  "l": "kop",
    "z": "asx",   "x": "zsdc",   "c": "xdvfg",  "v": "cfbg",   "b": "vghn",
    "n": "bhjmk", "m": "njkl",
}


def _strip_accents(s: str) -> str:
    """á → a, é → e, ñ → n, ü → u. NFD-decompose then drop combining marks.
    (ñ decomposes to n + combining tilde, which is dropped → n.)"""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _adj_typo(w: str, rng: random.Random) -> str:
    if not w:
        return w
    chars = list(w)
    candidates = [i for i, c in enumerate(chars) if c.lower() in ADJ]
    if not candidates:
        return w
    i = rng.choice(candidates)
    c = chars[i].lower()
    neighbours = ADJ[c]
    if not neighbours:
        return w
    new_c = rng.choice(neighbours)
    if chars[i].isupper():
        new_c = new_c.upper()
    chars[i] = new_c
    return "".join(chars)


def _drop_accent(w: str, rng: random.Random) -> str:
    """Targeted diacritic drop: only fires if the word has at least one accented
    char (covers ñ and rare acute/ü)."""
    has_accent = any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", w))
    if not has_accent:
        return w
    return _strip_accents(w)


def _transpose(w: str, rng: random.Random) -> str:
    if len(w) < 3:
        return w
    i = rng.randrange(len(w) - 1)
    return w[:i] + w[i+1] + w[i] + w[i+2:]


def _insert(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randrange(len(w) + 1)
    extra = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return w[:i] + extra + w[i:]


def _delete(w: str, rng: random.Random) -> str:
    if len(w) <= 2:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i+1:]


def _double(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i] + w[i:]


def _shortcut(w: str, rng: random.Random) -> str | None:
    """If we have a known shortcut for this word, return one of the shortcut forms."""
    forms = SHORTCUTS.get(w.lower())
    if not forms:
        return None  # caller should fall back to another rule
    pick = rng.choice(forms)
    if w[0:1].isupper() and pick:
        pick = pick[0].upper() + pick[1:]
    return pick


# Per-rule weight: chosen to overweight realistic Basque typo classes.
# Adjacency + transposition/doubling dominate; accent-drop is lighter than in
# Portuguese because standard Basque uses few diacritics.
RULES: list[tuple[Callable, int]] = [
    (_drop_accent, 20),    # ñ→n and rare acute/ü loss
    (_adj_typo,    30),    # the dominant Basque typo class
    (_transpose,   15),
    (_delete,      12),
    (_insert,      11),
    (_double,      12),
]


def synth_typo(word: str, rng: random.Random) -> str | None:
    """Generate one plausible typo for `word`. Returns None if word is too short or noise.

    Strategy:
      - Try a known shortcut (high quality) ~25% of the time, if available.
      - Otherwise pick a synthetic rule by weight.
      - If the synthetic rule produces the original (no-op), retry once.
    """
    if len(word) < 2 or not re.match(r"^[A-Za-zÀ-ÿ'-]+$", word):
        return None
    # 25% known-shortcut bias
    if rng.random() < 0.25:
        s = _shortcut(word, rng)
        if s is not None and s != word:
            return s
    # Synthetic rule
    rules, weights = zip(*RULES)
    for _ in range(2):
        rule = rng.choices(rules, weights=weights, k=1)[0]
        out = rule(word, rng)
        if out != word:
            return out
    return None


def freq_weight_log(freq: int) -> float:
    """log(freq + 1) frequency dampener (per FUTO wiki recommendation)."""
    return math.log(freq + 1)


def to_keypress_chars(typed: str) -> list[str]:
    """Convert a string to <CHAR_X> tokens (ASCII A-Z only).
    Strips diacritics (á→A, ñ→N, ü→U) and case via NFD. Non-letter chars dropped.
    This is the FUTO Keyboard's keypress-token format — what the model is
    actually trained on (verified via reference English model inference).
    """
    out = []
    for ch in typed:
        decomposed = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        for c in base.upper():
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out


def make_xbu_triple(typo: str, correct: str) -> str:
    """FUTO autocorrect format: <XBU><CHAR_*>...<CHAR_*><XBC>correct<XEC>.
    The TYPED part is keypresses (one <CHAR_X> per stroke); the CORRECTION
    part is the actual word in plain text. Verified against the reference
    English model via inference."""
    chars = "".join(to_keypress_chars(typo))
    return f"<XBU>{chars}<XBC>{correct}<XEC>"


def make_inline_corrected(text: str, rng: random.Random, typo_rate: float = 0.33) -> str:
    """Phase 4b format: take a sentence, randomly replace ~typo_rate of words
    with <XBU><CHAR_*>...<CHAR_*><XBC>correct<XEC>. Words too short / non-alphabetic are skipped."""
    out_words: list[str] = []
    for word in text.split():
        # Strip leading/trailing punctuation; keep it around the corrected form.
        m = re.match(r"^([^A-Za-zÀ-ÿ']*)([A-Za-zÀ-ÿ']+)([^A-Za-zÀ-ÿ']*)$", word)
        if not m or rng.random() > typo_rate:
            out_words.append(word)
            continue
        prefix, core, suffix = m.groups()
        typo = synth_typo(core, rng)
        if typo is None:
            out_words.append(word)
            continue
        out_words.append(prefix + make_xbu_triple(typo, core) + suffix)
    return " ".join(out_words)


# ---------------------------- Self-test ----------------------------

def _selftest() -> None:
    rng = random.Random(42)
    samples = [
        "Egun on, nola zaude gaur?",
        "Euskara ikasten ari naiz eta oso gustatzen zait.",
        "Bihar goizean etorriko naiz zure etxera.",
        "Eskerrik asko laguntzagatik, benetan miler esker.",
        "Gaur arratsaldean futbola ikusiko dugu estadioan.",
    ]
    print("=== Phase 4b inline format ===")
    for s in samples:
        out = make_inline_corrected(s, rng, typo_rate=0.4)
        print(f"  {out}")
    print()
    print("=== Phase 4a isolated triples (sample 30) ===")
    for word in ["kaixo", "agur", "mesedez", "barkatu", "eskerrik", "egun",
                 "orain", "gaur", "bihar", "etxe", "urte", "mutila", "neska",
                 "liburua", "katua", "txakurra", "ikusi", "etorri", "egin",
                 "izan", "handi", "txikia", "eder", "berri", "zahar",
                 "gazte", "lan", "hitz", "dira", "iñaki", "naiz"]:
        typo = synth_typo(word, rng)
        if typo is None:
            print(f"  [skip] {word}")
            continue
        print(f"  {make_xbu_triple(typo, word)}")


if __name__ == "__main__":
    _selftest()
