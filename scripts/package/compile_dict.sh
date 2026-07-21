#!/usr/bin/env bash
# Compile the Basque wordlist (AOSP "combined" text) into a binary .dict that
# FUTO Keyboard can side-load via Settings → Languages & Models → Import.
#
# The .combined.gz is the *source* format (what FUTO compiles into the app at
# build time, and what this project ships). The .dict is the *binary* artifact
# the app's import UI actually accepts (it detects the 0x9bc13afe magic +
# locale header — see RESEARCH.md §11.2 and DictionaryFactory.java).
#
# We use the AOSP dicttool (java_binary_host) — a standard AOSP v2 (202) binary
# dict is readable by FUTO's AOSP-forked dictionary engine. The dicttool jar is
# a community prebuild (Helium314/aosp-dictionaries); it bundles test classes
# that need junit on the classpath, so we fetch junit too.
#
# Usage:
#   ./scripts/package/compile_dict.sh                         # dictionaries/*.combined.gz → eu.dict
#   ./scripts/package/compile_dict.sh path/to/in.combined.gz out.dict
#
# Requires: java (JRE 11+). Network access on first run (downloads ~360 KB of jars).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACHE_DIR="$REPO_ROOT/dictionaries/.dicttool_cache"
mkdir -p "$CACHE_DIR"

DICTTOOL_JAR="$CACHE_DIR/dicttool_aosp.jar"
JUNIT_JAR="$CACHE_DIR/junit.jar"
DICTTOOL_URL="https://raw.githubusercontent.com/vaimalaviya1233/aosp-dictionaries/main/dicttool_aosp.jar"
JUNIT_URL="https://repo1.maven.org/maven2/junit/junit/3.8.2/junit-3.8.2.jar"
MAIN_CLASS="com.android.inputmethod.latin.dicttool.Dicttool"

IN="${1:-$REPO_ROOT/dictionaries/eu_wordlist.combined.gz}"
OUT="${2:-$REPO_ROOT/dictionaries/eu.dict}"

command -v java >/dev/null || { echo "ERROR: java not found (need JRE 11+)" >&2; exit 1; }

# 1. Fetch the dicttool + junit jars (cached in dictionaries/.dicttool_cache/)
if [ ! -f "$DICTTOOL_JAR" ]; then
  echo "Downloading AOSP dicttool jar …"
  curl -fsSL "$DICTTOOL_URL" -o "$DICTTOOL_JAR"
fi
if [ ! -f "$JUNIT_JAR" ]; then
  echo "Downloading junit jar (dicttool bundles test classes) …"
  curl -fsSL "$JUNIT_URL" -o "$JUNIT_JAR"
fi

# 2. Decompress .combined.gz → .combined (dicttool takes raw combined text)
TMP_COMBINED="$(mktemp --suffix=.combined)"
trap 'rm -f "$TMP_COMBINED"' EXIT
case "$IN" in
  *.gz) gunzip -c "$IN" > "$TMP_COMBINED" ;;
  *)    cp "$IN" "$TMP_COMBINED" ;;
esac
echo "Input: $IN ($(wc -l < "$TMP_COMBINED") lines)"

# 3. Compile combined → binary dict (v2 / version 202)
echo "Compiling .combined → $OUT …"
java -cp "$DICTTOOL_JAR:$JUNIT_JAR" "$MAIN_CLASS" makedict \
  -s "$TMP_COMBINED" -d "$OUT"

# 4. Verify the magic + locale header
MAGIC="$(xxd -l 4 -p "$OUT" 2>/dev/null || true)"
echo
echo "Done: $OUT ($(stat -c%s "$OUT") bytes)"
echo "  magic: ${MAGIC:-?}  (expect 9bc13afe — FUTO's import trigger)"
echo "  header:" 
strings -n 3 "$OUT" | grep -A0 -E '^(main:eu|locale|dictionary|description)$' | head -4
echo
echo "Side-load: transfer $OUT to your phone and open it with FUTO Keyboard, or"
echo "import via Settings → Languages & Models → Import. The app detects the"
echo "0x9bc13afe magic + locale=eu header and registers it as the Basque main"
echo "dictionary (DictionaryFactory.tryOpeningCustomMainDictionaryForLocale)."
