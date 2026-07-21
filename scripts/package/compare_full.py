"""Field-by-field comparison: our Basque dict vs FUTO-referenced (Helium314) dict."""
import gzip, os, sys, urllib.request
from collections import Counter

sys.path.insert(0, ".")
from config.eu import AUTOCORRECT_TESTS


def load(path, gz=None):
    if gz is None:
        gz = path.endswith(".gz")
    op = (gzip.open if gz else open)(path, "rt", encoding="utf-8", errors="replace")
    uni = {}; big = 0; header = None
    with op as f:
        for line in f:
            line = line.rstrip("\n")
            if header is None:
                header = line; continue
            if line.startswith("  bigram="):
                big += 1
            elif line.startswith(" word="):
                _, rest = line[1:].split("=", 1)
                w, fr = rest.rsplit(",f=", 1)
                uni[w] = int(fr)
    return header, uni, big


if not os.path.exists("/tmp/futo_eu.combined"):
    urllib.request.urlretrieve(
        "https://codeberg.org/Helium314/aosp-dictionaries/raw/branch/main/wordlists/main_eu.combined",
        "/tmp/futo_eu.combined",
    )

hf, wf, bf = load("/tmp/futo_eu.combined")
ho, wo, bo = load("dictionaries/eu_wordlist.combined.gz")

print("=" * 64)
print("FIELD-BY-FIELD: ours vs FUTO-referenced (Helium314) Basque dict")
print("=" * 64)
hdr = "field".ljust(28) + "H314 (FUTO ref)".rjust(18) + "OURS".rjust(18)
print(hdr)
print("-" * 64)
print("unigrams".ljust(28) + f"{len(wf):>18,}" + f"{len(wo):>18,}")
print("bigrams".ljust(28) + f"{bf:>18,}" + f"{bo:>18,}")
print("f range".ljust(28) + f"{min(wf.values())}-{max(wf.values())}".rjust(18) + f"{min(wo.values())}-{max(wo.values())}".rjust(18))
tg = [c.lower() for _, c in AUTOCORRECT_TESTS]
pf = sum(1 for w in tg if w in wf); po = sum(1 for w in tg if w in wo)
print("autocorrect targets".ljust(28) + f"{pf}/{len(tg)}".rjust(18) + f"{po}/{len(tg)}".rjust(18))
cf = sum(1 for w in wf if w[:1].isupper()); co = sum(1 for w in wo if w[:1].isupper())
print("proper nouns (cap.)".ljust(28) + f"{cf:>18,}" + f"{co:>18,}")
print("-" * 64)

f_da = wf.get("da", 0); o_da = wo.get("da", 0)
print("\nFrequency quality (da = #1 Basque word):")
print(f"  H314: da f={f_da}, {sum(1 for v in wf.values() if v > f_da):,} words rank above it")
print(f"  OURS: da f={o_da}, {sum(1 for v in wo.values() if v > o_da):,} words rank above it")
print(f"  H314 words clamped at f=255: {sum(1 for v in wf.values() if v == 255):,}")
print(f"  OURS words clamped at f=255: {sum(1 for v in wo.values() if v == 255):,}")

sf, so = set(wf), set(wo)
print(f"\nOverlap: {len(sf & so):,} in both, {len(sf - so):,} only-H314, {len(so - sf):,} only-OURS")

# sample top bigrams from ours
print("\nTop-10 our bigrams (contextual next-word pairs):")
bigrams = Counter()
with gzip.open("dictionaries/eu_wordlist.combined.gz", "rt", encoding="utf-8") as f:
    cur = None
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("  bigram="):
            _, rest = line[2:].split("=", 1)
            w, fr = rest.rsplit(",f=", 1)
            bigrams[(cur, w)] = int(fr)
        elif line.startswith(" word="):
            _, rest = line[1:].split("=", 1)
            cur, _ = rest.rsplit(",f=", 1)
for (w1, w2), c in bigrams.most_common(10):
    print(f"  {w1} {w2:10} f={c}")
