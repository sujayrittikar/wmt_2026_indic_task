#!/usr/bin/env python3
"""
Canonical data-prep pipeline for WMT-2026 Indic MT.

This is the AUTHORITATIVE producer of `data_clean/` — both the baselines and the CA
model consume its output. The EDA notebook's inline cleaning is exploratory only.

Steps (all auditable, nothing dropped silently):
  1. Robust load (Mizo has no header; filenames have typos -> glob match).
  2. NFC normalization + whitespace squash.
  3. Drop null/empty either side.
  4. Drop source==target copy-through.
  5. Exact dedup (per language, on the pair).
  6. Near-duplicate dedup (per language, on a normalized casefold/punct-stripped key).
  7. Length filtering: max words, max chars, and a word length-ratio bound.
  8. Deterministic train/dev/test split (seed=42), then de-leak: no test/dev SOURCE
     sentence may appear in train.
  9. Log per-language token counts, type-token ratio (TTR) and char-vocab size.

Outputs:
  data_clean/<lang>/{train,dev,test}.tsv      (columns: en, tgt)
  data_clean/prep_report.md                   (human-readable audit)
  data_clean/prep_report.json                 (machine-readable)

Run:  python prep_data.py
"""
import os, glob, re, json, unicodedata
from collections import Counter
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, "category_1")
CLEAN_DIR = os.path.join(ROOT, "data_clean")
SEED      = 42

# ── filtering thresholds (tuned to EDA: keep short phrases, kill outliers/misalignment) ──
MAX_WORDS   = 100     # drop pairs with > this many whitespace tokens on either side
MAX_CHARS   = 1000    # drop pairs with > this many chars on either side
WORD_RATIO  = 3.0     # drop if longer/shorter word count exceeds this (only when both >= 5 words)
CHAR_RATIO  = 9.0     # drop gross char-length misalignment (applies to all, incl. short)

# ── split config ──
DEV_FRAC, TEST_FRAC, SPLIT_CAP = 0.05, 0.05, 2000

# key -> (filename substring, has_header, target script)
REGISTRY = {
    "Nyishi":       ("*Nyishi*",        True,  "Latin"),
    "Assamese":     ("*Assamese*",      True,  "Bengali"),
    "Mizo":         ("*Mizo*",          False, "Latin"),       # no header row
    "Khasi":        ("*Khasi*",         True,  "Latin"),
    "Manipuri":     ("*Manipuri*",      True,  "Bengali"),     # Meiteilon, Bengali script
    "Meitei-Mayek": ("*Meitei-Mayek*",  True,  "MeiteiMayek"), # Meiteilon, Meitei script
}

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS    = re.compile(r"\s+")

def nfc(s):        return unicodedata.normalize("NFC", s) if isinstance(s, str) else ""
def squash(s):     return _WS.sub(" ", s).strip() if isinstance(s, str) else ""
def norm_key(s):   return _WS.sub(" ", _PUNCT.sub("", s.casefold())).strip()
def wcount(s):     return len(s.split())


def load_pair(key):
    pattern, has_header, script = REGISTRY[key]
    hits = glob.glob(os.path.join(DATA_DIR, pattern))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern}")
    path = hits[0]
    rd = pd.read_csv if path.lower().endswith(".csv") else pd.read_excel
    df = rd(path, header=0 if has_header else None).iloc[:, :2]
    df.columns = ["en", "tgt"]
    df["lang"], df["script"] = key, script
    return df, os.path.basename(path)


def clean_lang(df):
    """Apply steps 2-7 to one language; return (clean_df, per_step_counts)."""
    steps = [("0_raw", len(df))]

    df = df.copy()
    df["en"]  = df["en"].map(nfc).map(squash)
    df["tgt"] = df["tgt"].map(nfc).map(squash)

    df = df[(df.en != "") & (df.tgt != "")];                         steps.append(("2_drop_empty", len(df)))
    df = df[df.en.str.casefold() != df.tgt.str.casefold()];          steps.append(("3_drop_copy", len(df)))
    df = df.drop_duplicates(subset=["en", "tgt"]);                   steps.append(("4_exact_dedup", len(df)))

    df["_k"] = (df.en.map(norm_key) + " ||| " + df.tgt.map(norm_key))
    df = df.drop_duplicates(subset="_k").drop(columns="_k");         steps.append(("5_near_dedup", len(df)))

    enw, tgw = df.en.map(wcount), df.tgt.map(wcount)
    enc, tgc = df.en.str.len(), df.tgt.str.len()
    longer  = pd.concat([enw, tgw], axis=1).max(axis=1)
    shorter = pd.concat([enw, tgw], axis=1).min(axis=1).clip(lower=1)
    wratio  = longer / shorter
    cratio  = pd.concat([enc, tgc], axis=1).max(axis=1) / pd.concat([enc, tgc], axis=1).min(axis=1).clip(lower=1)

    keep = ((enw <= MAX_WORDS) & (tgw <= MAX_WORDS) &
            (enc <= MAX_CHARS) & (tgc <= MAX_CHARS) &
            (cratio <= CHAR_RATIO) &
            ~((shorter >= 5) & (wratio > WORD_RATIO)))
    df = df[keep];                                                   steps.append(("7_length_filter", len(df)))
    return df.reset_index(drop=True), steps


def split_and_deleak(df):
    """Deterministic split, then remove any train pair whose SOURCE appears in dev/test."""
    d = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(d)
    n_dev  = min(SPLIT_CAP, round(n * DEV_FRAC))
    n_test = min(SPLIT_CAP, round(n * TEST_FRAC))
    test  = d.iloc[:n_test]
    dev   = d.iloc[n_test:n_test + n_dev]
    train = d.iloc[n_test + n_dev:]
    # Each unique SOURCE sentence must live in exactly one split: test > dev > train.
    before = len(dev) + len(train)
    dev   = dev[~dev.en.isin(set(test.en))]
    train = train[~train.en.isin(set(dev.en) | set(test.en))]
    deleaked = before - len(dev) - len(train)
    return train, dev, test, deleaked


def lang_stats(train, dev, test):
    """Token counts, TTR and char-vocab on the TRAIN split (what models learn from)."""
    en_tok  = [w for s in train.en  for w in s.lower().split()]
    tg_tok  = [w for s in train.tgt for w in s.lower().split()]
    en_types, tg_types = len(set(en_tok)), len(set(tg_tok))
    char_vocab = len(set("".join(train.tgt)))
    return {
        "train": len(train), "dev": len(dev), "test": len(test),
        "en_tokens": len(en_tok), "en_types": en_types,
        "en_TTR": round(en_types / max(len(en_tok), 1), 4),
        "tgt_tokens": len(tg_tok), "tgt_types": tg_types,
        "tgt_TTR": round(tg_types / max(len(tg_tok), 1), 4),
        "tgt_char_vocab": char_vocab,
    }


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    report = {"config": {"MAX_WORDS": MAX_WORDS, "MAX_CHARS": MAX_CHARS,
                         "WORD_RATIO": WORD_RATIO, "CHAR_RATIO": CHAR_RATIO,
                         "seed": SEED, "dev_frac": DEV_FRAC, "test_frac": TEST_FRAC,
                         "split_cap": SPLIT_CAP},
              "languages": {}}

    print(f"{'lang':14s} {'raw':>7s} {'clean':>7s} {'train':>7s} {'dev':>5s} {'test':>5s} "
          f"{'deleak':>6s} {'tgtTTR':>7s} {'charV':>6s}")
    print("-" * 78)

    for key in REGISTRY:
        raw, fname = load_pair(key)
        clean, steps = clean_lang(raw)
        train, dev, test, deleaked = split_and_deleak(clean)

        outdir = os.path.join(CLEAN_DIR, key)
        os.makedirs(outdir, exist_ok=True)
        for name, part in [("train", train), ("dev", dev), ("test", test)]:
            part[["en", "tgt"]].to_csv(os.path.join(outdir, f"{name}.tsv"),
                                       sep="\t", index=False)

        stats = lang_stats(train, dev, test)
        report["languages"][key] = {
            "source_file": fname,
            "script": REGISTRY[key][2],
            "steps": dict(steps),
            "deleaked_from_train": deleaked,
            **stats,
        }
        print(f"{key:14s} {steps[0][1]:>7,} {steps[-1][1]:>7,} {stats['train']:>7,} "
              f"{stats['dev']:>5,} {stats['test']:>5,} {deleaked:>6,} "
              f"{stats['tgt_TTR']:>7.3f} {stats['tgt_char_vocab']:>6,}")

    # ── write reports ──
    with open(os.path.join(CLEAN_DIR, "prep_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = ["# Data-prep report\n",
             f"_Generated by `prep_data.py` (seed={SEED}). Filters: "
             f"max_words={MAX_WORDS}, max_chars={MAX_CHARS}, word_ratio<={WORD_RATIO}, "
             f"char_ratio<={CHAR_RATIO}._\n",
             "## Cleaning funnel (rows remaining after each step)\n"]
    step_names = [s for s, _ in next(iter(report["languages"].values()))["steps"].items()]
    lines.append("| lang | " + " | ".join(step_names) + " | deleak |")
    lines.append("|" + "---|" * (len(step_names) + 2))
    for k, v in report["languages"].items():
        row = " | ".join(f"{v['steps'][s]:,}" for s in step_names)
        lines.append(f"| {k} | {row} | {v['deleaked_from_train']:,} |")

    lines.append("\n## Splits & sparsity (measured on TRAIN)\n")
    lines.append("| lang | train | dev | test | en_TTR | tgt_TTR | tgt_types | tgt_char_vocab |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, v in report["languages"].items():
        lines.append(f"| {k} | {v['train']:,} | {v['dev']:,} | {v['test']:,} | "
                     f"{v['en_TTR']} | {v['tgt_TTR']} | {v['tgt_types']:,} | {v['tgt_char_vocab']:,} |")
    lines.append("\n> Lower TTR = more repetition (less lexical diversity). "
                 "tgt_char_vocab is the byte/char-model alphabet size — relevant for the CA front-end.\n")
    lines.append("> **Splits are PLACEHOLDERS** — replace dev/test with the official WMT sets when available.\n")

    with open(os.path.join(CLEAN_DIR, "prep_report.md"), "w") as f:
        f.write("\n".join(lines))

    print("-" * 78)
    print("wrote:", os.path.join(CLEAN_DIR, "prep_report.md"),
          "+ prep_report.json + <lang>/{train,dev,test}.tsv")


if __name__ == "__main__":
    main()
