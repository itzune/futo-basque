#!/usr/bin/env bash
# Compare our Basque dictionary against the one FUTO's dictionaries page
# references for eu-ES (Helium314's community AOSP dict on Codeberg).
#
# Finding: FUTO does NOT ship its own Basque dictionary — its dictionaries page
# (https://keyboard.futo.tech/dictionaries?locale=eu-ES) links to Helium314's
# main_eu.dict. So this is a direct apples-to-apples AOSP-v2 vs AOSP-v2 compare.
#
# Usage: ./scripts/package/compare_dicts.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TMP="${TMPDIR:-/tmp}"; mkdir -p "$TMP"
FUTO="$TMP/futo_eu.combined"
if [ ! -f "$FUTO" ]; then
  echo "Downloading FUTO-referenced (Helium314) Basque source combined …"
  curl -fsSL "https://codeberg.org/Helium314/aosp-dictionaries/raw/branch/main/wordlists/main_eu.combined" -o "$FUTO"
fi
uv run python - <<'PY'
import gzip
def load(path, gz=None):
    if gz is None: gz = path.endswith('.gz')
    op = (gzip.open if gz else open)(path,'rt',encoding='utf-8',errors='replace')
    words={}; header=None
    with op as f:
        for line in f:
            line=line.rstrip('\n')
            if header is None: header=line; continue
            if line.startswith(' word='):
                _,rest=line[1:].split('=',1); w,f=rest.rsplit(',f=',1); words[w]=int(f)
    return header,words
import os
TMP=os.environ.get("TMPDIR","/tmp")
hf,wf=load(f"{TMP}/futo_eu.combined")
ho,wo=load("dictionaries/eu_wordlist.combined.gz")
print(f"FUTO/H314: {len(wf):>7} words  | OURS: {len(wo):>7} words")
print(f"overlap {len(set(wf)&set(wo))} | only-FUTO {len(set(wf)-set(wo))} | only-OURS {len(set(wo)-set(wf))}")
f_da=wf.get('da',0)
print(f"'da' (#1 Basque word) rank: FUTO/H314 f={f_da}, {sum(1 for v in wf.values() if v>f_da)} words above it | OURS f={wo['da']}, {sum(1 for v in wo.values() if v>wo['da'])} above")
PY
