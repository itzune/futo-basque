"""
Basque (euskara, `eu`) language configuration for the FUTO keyboard model.

Every piece of *language-specific* data lives here so the pipeline scripts stay
generic:

  - Language constants (BCP-47 tag, tokenizer prefix, model name, wandb project)
  - Corpus dataset registry (HuggingFace ids/configs) + a light language filter
  - Tokenizer content-slot word lists (the 146 + 56 user-defined-symbol fillers)
  - Emoji set for the 40 emoji slots
  - Typo-synthesis shortcut dictionary
  - Eval test sets (autocorrect + next-word)

Basque orthography notes (relevant to the choices below):
  * Standard Basque (euskara batua) uses the Latin-26 alphabet + ñ. Acute
    accents (á é í ó ú) and ü appear only rarely (mainly in names / dialects),
    so the dominant real-world typo class is NOT accent-drop (as in Portuguese)
    but keyboard-adjacency, transposition, doubling, and ñ→n loss.
  * Basque is agglutinative: case is expressed by suffixes, so there are few
    standalone prepositions. The slot fillers below prioritise high-frequency
    function words, pronouns, auxiliaries, and common verbs/nouns.
  * NFD-decomposition handles ñ→n and any acute/ü loss exactly as it does for
    Portuguese — no app patches, no new tokens.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Language constants
# --------------------------------------------------------------------------- #

LANGUAGE = "eu"                  # BCP-47 tag; matches FUTO's locales/eu.json
TOKENIZER_PREFIX = "spm_eu"      # → spm_eu.model / spm_eu.vocab
MODEL_NAME = "Euskara v1"        # general.name in the GGUF
WANDB_PROJECT = "futo-eu"        # default wandb project for training runs

# Approx chars/token for the corpus token budget. Basque is agglutinative so
# words run a little longer than Romance languages; ~4.5 chars/token.
CHARS_PER_TOKEN = 4

# --------------------------------------------------------------------------- #
# Corpus strategy
# --------------------------------------------------------------------------- #
# Two tiers, kept in SEPARATE directories so the tokenizer can be trained on
# the clean tier only (see RESEARCH.md §11.4 for the full rationale):
#
#   PRIMARY (corpora/clean/)       → tokenizer training + pretrain base
#     Reuse the morpheus-mamba project's cleaned Latxa corpus v2
#     (data/clean-v3/). HiTZ already did the curation, deduplication, and
#     per-source LLM quality audit that morpheus documents. 11 sources,
#     ~15 GB, ~4.77 B tokens, avg quality 4.6/5.
#
#   CONVERSATIONAL (corpora/conversational/)  → Phase 4c adaptation ONLY
#     BERnaT BSMtime (Basque Social Media): 11 M individual posts,
#     ~250 M tokens. Re-included despite morpheus's a-priori
#     exclusion because the BERnaT paper (Azurmendi et al. 2025,
#     arXiv:2512.03903 — the SAME HiTZ group) shows diverse data helps
#     without hurting standard-form accuracy, and FUTO is a phone keyboard
#     (deployment distribution = chat, not Wikipedia). Aggressively cleaned
#     (emoji/URL/mention/code-switch stripping) by clean_bernat.py.
#     REVISED STRATEGY (§11.6): excluded from pretrain (Phase 3) and used
#     only for Phase 4c conversational adaptation finetune, matching FUTO's
#     own English pipeline (SlimPajama pretrain → small conversational
#     finetune at the end).
#
# HF streaming fallback: if the morpheus repo is not available locally, both
# scripts can stream the same sources from HuggingFace (HiTZ/latxa-corpus-v2
# and HiTZ/BERnaT-Diverse) and apply lighter cleaning. See build_corpus.py
# --from-hf and clean_bernat.py --from-hf.

# Approx chars/token for the corpus token budget. Basque is agglutinative so
# words run a little longer than Romance languages; ~4.5 chars/token.
CHARS_PER_TOKEN = 4

# --- Morpheus clean-v3: the 11 approved Latxa sources --------------------- #
# Files live at <morpheus_repo>/data/clean-v3/HiTZ_latxa-corpus-v2_<source>.txt
# (one file per source, post-exclusion). build_corpus.py globs this dir and
# stages all .txt it finds into shard_*.txt. This list is the reference set +
# allows --sources subsetting; quality notes from morpheus's LLM audit.
LATXA_SOURCES = [
    "euscrawl-v2",     # Q5.0 — news/media crawl, best source (56 % of v1)
    "parleus",         # Q4.9 — parliament transcriptions
    "zelaihandi",      # Q4.9 — curated diverse corpus
    "bopv",            # Q4.8 — Basque Government official gazette
    "botha",           # Q4.8 — Álava provincial gazette
    "colossal-oscar",  # Q4.7 — cleaned Common Crawl
    "wikipedia",       # Q4.6 — Basque Wikipedia dump (Sep 2025)
    "cultura-x",       # Q4.6 — cleaned web (CulturaX)
    "hplt-v2",         # Q4.4 — HPLT v2 crawl
    "fineweb2",        # Q4.3 — FineWeb2
    "finepdfs",        # Q3.7 — FinePDFs (digit-filtered in morpheus Phase 3)
]

# Sources morpheus excluded and WHY (documented for reference). BERnaT BSM is
# re-included separately as the conversational tier — see below.
LATXA_EXCLUDED_SOURCES = {
    "hplt-v1":     "83.8 % duplicates, only 4.9 % Basque signal — net negative",
    "bog":         "Phase-2 sentence splitting fragmented legal text (36/40 lines incomplete)",
    "aldizkariak": "35 % boilerplate (author lists, English titles, citation numbers)",
}

# --- BERnaT BSM (conversational supplement) ------------------------------- #
# Use BSMtime (NOT BSMauthor). Both contain the SAME ~11M posts, but organized
# differently: BSMtime = one post per row (11M rows, 34-280 chars each);
# BSMauthor = posts grouped by author into 13K giant timeline documents (~91KB
# each). Our line-level cleaning (is_strictly_eu, min-words, per-line dedup)
# only works on individual posts, so BSMtime is required. Using both would be
# pure duplication (same textual content).
BERNAT_REPO = "HiTZ/BERnaT-Diverse"
BERNAT_CONFIG = "BSMtime"
BERNAT_TEXT_KEY = "text"
# Lightly-cleaned BSM on the morpheus server (Phase 1, pre-exclusion):
BERNAT_LOCAL_FILE = "data/clean/HiTZ_BERnaT-Diverse_BSMtime.txt"

# BSM cleaning thresholds (consumed by clean_bernat.py).
BERNAT_MAX_SPANISH_RATIO = 0.15   # drop lines whose Spanish-function-word ratio exceeds this
BERNAT_MIN_WORDS = 3              # drop lines with fewer words after stripping noise
BERNAT_TARGET_TOKENS = 250_000_000  # ~5–10 % of a 3–5 B token pretrain corpus

# --- HuggingFace fallback (when morpheus clean-v3 is unavailable) --------- #
# Latxa corpus v2 is published per-source on HF. Config names are the display
# names from the dataset card; text field is "text".
LATXA_HF_REPO = "HiTZ/latxa-corpus-v2"
LATXA_HF_TEXT_KEY = "text"
LATXA_HF_CONFIGS = {
    "euscrawl-v2":    "Euscrawl v2",
    "parleus":        "ParlEus",
    "zelaihandi":     "ZelaiHandi",
    "bopv":           "BOPV",
    "botha":          "BOTHA",
    "colossal-oscar": "Colossal OSCAR",
    "wikipedia":      "Wikipedia",
    "cultura-x":      "CulturaX",
    "hplt-v2":        "HPLT v2",
    "fineweb2":       "FineWeb2",
    "finepdfs":       "FinePDFs",
}
# HF configs to NEVER stream (morpheus-excluded; BSM handled via BERNAT_*).
LATXA_HF_EXCLUDED_CONFIGS = {"HPLT v1", "BOG", "Aldizkariak"}

# --------------------------------------------------------------------------- #
# Language filters
# --------------------------------------------------------------------------- #
# Basque shares the Latin alphabet with Spanish, but Spanish's most frequent
# function words (que, de, el, la, los, las, un, una, para, con, por, más,
# pero, como) are NOT Basque words, while Basque's (eta, ez, da, baina,
# direla, egun, gaur, …) are NOT Spanish.

_SPANISH_MARKERS = re.compile(
    r"\b(que|de|el|la|los|las|un|una|unos|unas|para|con|por|más|pero|como|"
    r"este|esta|eso|esa|su|sus|del|al|lo|le|se|me|te|nos|ya|hay|fue|son)\b"
)
_BASQUE_MARKERS = re.compile(
    r"\b(eta|ez|baina|dira|dela|direla|zein|nola|egun|orain|gaur|bihar|"
    r"baita|nahiz|dago|dut|naiz|zara|gara|izan|egin|joan|etorri|ikusi|"
    r"eskerrik|mesedez|barkatu|agur|kaixo|handi|etxe|urte|mutil|neska|"
    r"maite|nahi|badago|horrela|beraz|ordea|ostera|adibidez|gustoko|"
    r"halaber|bertan|hemen|oraintxe)\b",
    re.IGNORECASE,
)


def is_likely_eu(text: str) -> bool:
    """Soft filter: drop only docs that are strongly Spanish with zero Basque.

    Used by the HF-streaming fallback in build_corpus.py (Latxa sources are
    already eu-curated, so this only catches rare mislabelling).
    """
    eu = len(_BASQUE_MARKERS.findall(text))
    es = len(_SPANISH_MARKERS.findall(text))
    if es > 5 and eu == 0:
        return False
    return True


def spanish_ratio(text: str) -> float:
    """Fraction of word tokens that are high-frequency Spanish function words.

    A stronger, quantitative signal than is_likely_eu() — used by
    clean_bernat.py to reject code-switched social-media lines (morpheus's
    audit found 6.43 % mixed-language in BSM).
    """
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    return len(_SPANISH_MARKERS.findall(text)) / len(words)


def is_strictly_eu(text: str, max_es_ratio: float = BERNAT_MAX_SPANISH_RATIO) -> bool:
    """Strict filter for social-media lines: reject code-switched content.

    Unlike is_likely_eu() (which only drops all-Spanish docs), this rejects any
    line whose Spanish-function-word ratio exceeds `max_es_ratio`. Used by
    clean_bernat.py on BSM posts where mixed-language lines are common.
    """
    if not is_likely_eu(text):
        return False
    return spanish_ratio(text) <= max_es_ratio


# --------------------------------------------------------------------------- #
# Tokenizer content-slot word lists
# --------------------------------------------------------------------------- #
# These fill the 300 user-defined-symbol slots (IDs 4..303). The STRUCTURAL
# slots (XBU/XBC/XEC, XC0-4, CHAR_A-Z) are added by build_user_defined_symbols()
# in scripts/tokenizer/train.py and MUST NOT change. Only the content slots
# below are language-specific. Padded with <FUTO> filler to the exact count.

# --- 146 slots (indices 28..173) — high-frequency words / function words ---
EU_FREQ_SHORT = [
    # very-high-frequency short words & particles
    "ta", "bai", "ez", "da", "oso", "ere", "ea", "oi", "ba", "ni",
    "zu", "gu", "hi", "ze", "neu", "bera", "berak", "bere", "nire", "zure",
    "gure", "honek", "hauek", "horiek", "haiek", "hau", "hori", "hura", "zuek", "euren",
]

EU_INTERROGATIVES_CONNECTIVES = [
    "nor", "zer", "non", "nola", "noiz", "zergatik", "zenbat", "zein", "nongo", "eta",
    "edo", "baina", "ordea", "beraz", "hala", "baita", "baizik", "nahiz", "besterik", "gainera",
    "ostera", "alegia", "hots", "adibidez", "berriz", "berriro", "beti", "inoiz", "orain", "halaber",
]

EU_AUX_VERBS = [
    # Basque auxiliaries / conditionals — among the most frequent tokens
    "naiz", "zara", "gara", "zarete", "dira", "zen", "ziren", "zineten", "nintzen", "zinen",
    "dut", "du", "dugu", "dute", "ditu", "ditut", "ditugu", "dituzte", "nuen", "zuen",
    "genuen", "zuten", "nuke", "luke", "genuke", "lukete", "litzateke", "balitz", "balu", "banu",
]

EU_COMMON_VERBS = [
    "egin", "joan", "etorri", "izan", "esan", "jakin", "ikusi", "eman", "hartu", "jarri",
    "hasi", "aurkitu", "erabili", "gorde", "irabazi", "galdu", "bizi", "ibili", "irakurri", "idatzi",
    "ikasi", "pentsatu", "sentitu", "nahi", "maite", "atera", "sartu", "eraman", "ekarri", "erosi",
]

EU_COMMON_WORDS = [
    "gaur", "bihar", "atzo", "etxe", "urte", "lan", "hitz", "gizon", "emakume", "mutil",
    "neska", "haur", "ikasle", "irakasle", "liburu", "eskola", "kale", "hiri", "herri", "mundu",
    "egun", "gau", "goiz", "arratsalde", "gabon", "aste",
]

# Combined list for slots 28..173 (tokenizer/train.py pads/truncates to 146).
SLOT_28_173 = (
    EU_FREQ_SHORT
    + EU_INTERROGATIVES_CONNECTIVES
    + EU_AUX_VERBS
    + EU_COMMON_VERBS
    + EU_COMMON_WORDS
)

# --- 56 slots (indices 208..263) — adjectives / quantifiers ---
EU_ADJECTIVES = [
    "handi", "txiki", "on", "txar", "eder", "berri", "zahar", "gazte", "goxo", "polita",
    "itsusi", "garbi", "zikin", "argi", "ilun", "bero", "hotz", "lehor", "heze", "gogor",
    "bigun", "astun", "arin", "motz", "luze", "altu", "baxu", "zabal", "lodi", "mehar",
    "sendo", "ahul", "indartsu", "azkar", "motel", "ezagun", "ezezagun", "anitz", "ugari",
    "leun", "latz", "zoragarri", "nabarmen", "berezia", "garrantzitsu", "zorion", "pozik",
    "triste", "harro", "bakar", "bikoitz", "zorrotz", "ohiko", "arraro", "moderno", "guzti", "dena",
]
SLOT_208_263 = EU_ADJECTIVES

# --- 40 emoji slots (indices 264..303) ---
# Language-neutral; mirrors the reference English model's set for max compat.
EMOJI = [
    "😂", "❤", "😍", "😭", "😘", "🙏", "👌", "👍", "💕", "🔥",
    "👀", "💯", "✨", "🥺", "😊", "💗", "🎂", "🎁", "🌟", "🎈",
    "💜", "💙", "✅", "😢", "😳", "💪", "💖", "🎶", "🙌", "⬅",
    "😋", "🙈", "💀", "😄", "🌹", "✌", "👉", "😞", "💛", "😜",
]

# --------------------------------------------------------------------------- #
# Typo-synthesis shortcuts
# --------------------------------------------------------------------------- #
# token -> plausible "wrong" forms a typist might write. Treated as a
# high-quality synthetic typo class (maps to a single recognised correction).
# Basque has fewer SMS abbreviations than Portuguese, so this is lighter; the
# bulk of synthetic typos come from the generic rules in typo_synthesis.py.

EU_SHORTCUTS: dict[str, list[str]] = {
    "eskerrik":  ["eskerik", "eskerik", "eskrrik"],
    "asko":      ["asko", "askko", "ask"],
    "mesedez":   ["mesedez", "mesede", "messedez"],
    "barkatu":   ["barkat", "barkatu", "barcatu"],
    "agur":      ["agur", "agru", "agr"],
    "kaixo":     ["kaixo", "kaix", "kaixp", "kaixoo"],
    "egun":      ["egn", "egun", "egn"],
    "orain":     ["orain", "oraim", "orin"],
    "gaur":      ["gaur", "garu", "gaut"],
    "bihar":     ["bihar", "bihr", "biar"],
    "baina":     ["baina", "bsina", "bina"],
    "beraz":     ["beraz", "neraz", "bera"],
    "hala":      ["hala", "hla", "hala"],
    "ere":       ["ere", "erew", "ers"],
    "oso":       ["oso", "os", "osoo"],
    "milesker":  ["milesker", "mileskr", "milesker"],
    "eta":       ["eta", "ta", "et"],
    "ez":        ["ez", "ezz", "ez"],
}


# --------------------------------------------------------------------------- #
# Eval test sets
# --------------------------------------------------------------------------- #

# (typo, expected_correction) — hand-curated Basque autocorrect cases.
# The typo is what the user *types* (encoded as <CHAR_*> keypresses); the
# correction is the plain-text word the model should emit after <XBC>.
AUTOCORRECT_TESTS = [
    # Keyboard-adjacency typos (the dominant Basque typo class)
    ("kaixp", "kaixo"),        # o→p   (kaixo = hello)
    ("agut", "agur"),          # r→t   (agur = hi/bye)
    ("mesedex", "mesedez"),    # z→x   (mesedez = please)
    ("oraim", "orain"),        # n→m   (orain = now)
    ("gaut", "gaur"),          # r→t   (gaur = today)
    ("bihsr", "bihar"),        # a→s   (bihar = tomorrow)
    ("dagp", "dago"),          # o→p   (dago = is/stands)
    ("narkatu", "barkatu"),    # b→n   (barkatu = sorry/excuse)
    ("ikuso", "ikusi"),        # i→o   (ikusi = to see)
    ("etprri", "etorri"),      # o→p   (etorri = to come)
    ("joab", "joan"),          # n→b   (joan = to go)
    ("ezket", "ezker"),        # r→t   (ezker = left)
    ("eskuim", "eskuin"),      # n→m   (eskuin = right)
    ("mutika", "mutila"),      # l→k   (mutila = boy)
    ("nesks", "neska"),        # a→s   (neska = girl)
    ("lam", "lan"),            # n→m   (lan = work)
    ("etxr", "etxe"),          # e→r   (etxe = house)
    ("irte", "urte"),          # u→i   (urte = year)
    ("hamdi", "handi"),        # n→m   (handi = big)
    ("txikis", "txikia"),      # a→s   (txikia = small)
    ("huta", "hura"),          # r→t   (hura = that)
    ("bsina", "baina"),        # a→s   (baina = but)
    ("neraz", "beraz"),        # b→n   (beraz = therefore)
    ("oraindil", "oraindik"),  # k→l   (oraindik = still)
    ("behim", "behin"),        # n→m   (behin = once)
    ("beto", "beti"),          # i→o   (beti = always)
    ("inois", "inoiz"),        # z→s   (inoiz = ever)
    ("nais", "naiz"),          # z→s   (naiz = I am)
    ("zars", "zara"),          # a→s   (zara = you are)
    ("izam", "izan"),          # n→m   (izan = to be)
    ("egim", "egin"),          # n→m   (egin = to do)
    ("hits", "hitz"),          # z→s   (hitz = word)
    ("latua", "katua"),        # k→l   (katua = cat)
    ("liborua", "liburua"),    # u→o   (liburua = book)
    ("txakirra", "txakurra"),  # u→i   (txakurra = dog)
    # ñ loss (ñ→n) — the main Basque diacritic typo
    ("inaki", "iñaki"),        # Iñaki (common name)
    # Doubling
    ("eskkerrik", "eskerrik"), # rr→rrr (eskerrik, in eskerrik asko)
    ("kaixoo", "kaixo"),       # o→oo
    # Transposition
    ("garu", "gaur"),          # ur→ru  (gaur = today)
    ("eskerirk", "eskerrik"),  # ri→ir
]

# (prefix, plausible-next-words) for next-word evaluation.
NEXT_WORD_TESTS = [
    ("Egun on, zer", ["moduz", "berri", "da", "nola"]),
    ("Ni euskara", ["ikasten", "hitzen", "dakit", "maite"]),
    ("Bai, gustatu", ["zait", "zaizu", "da"]),
    ("Ez dut", ["ahaztu", "dakit", "maite", "nahi", "ikusi"]),
    ("Zein da zure", ["izena", "adina", "etxea"]),
    ("Bihar goizean", ["etorriko", "joango", "izango", "izanen"]),
    ("Eskerrik asko", ["guztiaz", "laguntzagatik", "denagatik", "gu"]),
    ("Non dago", ["etxea", "trena", "garagardoa", "jana"]),
    ("Zer", ["da", "esan", "egin", "norena"]),
    ("Nola", ["zaude", "dago", "da", "esaten"]),
    ("Gaur ezin", ["dut", "naiz", "da", "dugu"]),
    ("Barkatu, ez", ["dakit", "nahi", "dut", "da"]),
]
