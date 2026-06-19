"""Single source of truth for WMT-2026 Indic datasets + model↔language assignment.

Imported by notebooks 03 (NLLB), 04 (CA pretrain), 05 (CA translate) and `_predict_gold.py` so every
file trains/evaluates on the SAME data and routes the SAME way.

Training policy (chosen 2026-06-15): **use ALL parallel data** — `data_clean/<lang>/{train,dev,test}.tsv`
PLUS the 2025 gold pairs — merged, NFC-normalised, deduped. No held-out split (final checkpoint is used).
The real evaluation is the blind competition gold (`GOLD_2026`, filled in when it arrives).
"""
import os, glob, unicodedata
import pandas as pd

ROOT     = "/data/sujay/wmt_2026_indic"
DATA_DIR = os.path.join(ROOT, "data_clean")
MONO_DIR = os.path.join(ROOT, "mono_data")
SPLITS   = ("train", "dev", "test")

LANGS = ["Assamese", "Khasi", "Manipuri", "Meitei-Mayek", "Mizo", "Nyishi"]

# Which system produces the final submission for each language (the user's plan):
#   NLLB-200 was stronger on the languages it covers; the CA byte-LM is the only system for
#   Khasi/Nyishi, and also handles Meitei-Mayek (Meitei script, which NLLB does not cover).
MODEL_FOR = {
    "Assamese": "nllb", "Manipuri": "nllb", "Mizo": "nllb",
    "Khasi": "ca", "Nyishi": "ca", "Meitei-Mayek": "ca",
}

# NLLB-200 FLORES-200 codes (None = not covered by NLLB)
NLLB_CODE = {
    "English": "eng_Latn", "Assamese": "asm_Beng", "Manipuri": "mni_Beng", "Mizo": "lus_Latn",
    "Meitei-Mayek": None, "Khasi": None, "Nyishi": None,
}

# ── 2025 gold (parallel, WITH references) — folded into training ──
# spec: (path_relative_to_ROOT, sheet, source_col, target_col)
GOLD_2025 = {
    "Assamese": ("Test Data Gold 2025/English-Assamese  WMT 2025 Test Set Gold.xlsx.xlsx", "EN - AS Correction", "Source Sentence", "Target Sentence"),
    "Manipuri": ("Test Data Gold 2025/English-Manipuri WMT 2025 Test Set Gold.xlsx", "Final Data", "Source Sentence", "Target Sentence"),
    "Mizo":     ("Test Data Gold 2025/English-Mizo WMT 2025 Test Set Gold.xlsx", "output", "Source Sentence", "Target Sentences"),
    "Khasi":    ("Test Data Gold 2025/English-Khasi WMT 2025 Test Set Gold.xlsx", "Final Data", "Source Sentence", "Target Sentences"),
    "Nyishi":   ("Test Data Gold 2025/English-Nyshi WMT 2025 Test Set Gold.xlsx", "Final Data", "Source Sentence", "Target Sentence"),
    # Meitei-Mayek: no 2025 gold set exists.
}

# ── REAL competition gold — FILL THIS IN WHEN IT ARRIVES ──
# Drop the files in GOLD_2026_DIR and add one entry per language below. Same shape as GOLD_2025, but the
# target column may be missing (blind set) — set it to None and the pipeline will only *predict* (no score).
GOLD_2026_DIR = os.path.join(ROOT, "Test Data Gold 2026")
GOLD_2026 = {
    # "Assamese": ("Test Data Gold 2026/English-Assamese ....xlsx", "<sheet>", "Source Sentence", "Target Sentence"),
    # "Khasi":    ("Test Data Gold 2026/English-Khasi ....xlsx",    "<sheet>", "Source Sentence", None),   # None ref = blind
    # ... add the languages the competition actually scores ...
}

def _nfc(s):
    return unicodedata.normalize("NFC", str(s)).strip()

def parallel_pairs(lang, include_gold2025=True):
    """All (en, tgt) pairs for `lang`: data_clean train+dev+test (+ 2025 gold), NFC + deduped."""
    pairs = []
    for sp in SPLITS:
        p = os.path.join(DATA_DIR, lang, f"{sp}.tsv")
        if os.path.exists(p):
            df = pd.read_csv(p, sep="\t").dropna()
            pairs += list(zip(df["en"].map(_nfc), df["tgt"].map(_nfc)))
    if include_gold2025 and lang in GOLD_2025:
        df = load_gold(GOLD_2025[lang])
        if "ref" in df:
            pairs += list(zip(df["src"], df["ref"]))
    seen, out = set(), []
    for en, tg in pairs:
        if en and tg and (en, tg) not in seen:
            seen.add((en, tg)); out.append((en, tg))
    return out

def english_pairs_pool(langs=None, include_gold2025=True):
    """Deduped English sentences across the given languages' pairs (for CA pretraining)."""
    seen = set()
    for lang in (langs or LANGS):
        for en, _ in parallel_pairs(lang, include_gold2025):
            if en not in seen:
                seen.add(en)
    return list(seen)

def mono_texts(lang):
    """Extra monolingual lines for `lang` (mono_data/<lang>/*.txt), stripped."""
    out = []
    for fp in sorted(glob.glob(os.path.join(MONO_DIR, lang, "*.txt"))):
        with open(fp, encoding="utf-8") as f:
            out += [ln.strip() for ln in f if ln.strip()]
    return out

def load_gold(spec, root=ROOT):
    """spec = (file, sheet, src_col, tgt_col_or_None). Returns df with 'src' (+ 'ref' if available)."""
    fn, sheet, sc, tc = spec
    path = os.path.join(root, fn)
    df = pd.read_excel(path, sheet_name=sheet) if path.lower().endswith((".xls", ".xlsx")) \
        else pd.read_csv(path)
    out = pd.DataFrame({"src": df[sc].map(_nfc)})
    if tc and tc in df.columns:
        out["ref"] = df[tc].map(_nfc)
    return out.dropna(subset=["src"]).reset_index(drop=True)

def stats():
    """Quick sanity table of how many pairs each language contributes."""
    rows = []
    for lang in LANGS:
        n = len(parallel_pairs(lang))
        rows.append({"lang": lang, "model": MODEL_FOR[lang], "all_pairs": n})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print(stats().to_string(index=False))
    print("\nGOLD_2026 configured:", bool(GOLD_2026), "| dir:", GOLD_2026_DIR)
