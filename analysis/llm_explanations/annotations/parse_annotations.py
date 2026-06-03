#!/usr/bin/env python3
"""
Parse annotation responses (burgers + juristen, all wetten) and
link each response to LLM explanation metadata from the thesis JSONL output.

Expected filename convention:
  LLM Uitleg Evaluatie -- <Wet> -- <Burgers|Juristen> (Responses).xlsx
  (em-dashes or regular dashes, case-insensitive)

Burger form (GROUP_SIZE=4, data_start auto-detected via [BURGER] prefix):
  duidelijkheid  — numeric 0-5
  actionability  — Ja/Deels/Nee -> 1.0/0.5/0.0
  leesbaarheid   — numeric 1-5
  toelichting    — free text

Jurist form (GROUP_SIZE=5, data_start=4):
  juridische_correctheid — numeric 0-5
  toelichting_q1         — free text
  ontbrekende_informatie — Ja/Deels/Nee -> 1.0/0.5/0.0
  toelichting_q2         — free text
  extra_aandacht         — Ja/Nee -> 1.0/0.0

Output:
  results/annotations_long.csv      one row per respondent x uitleg
  results/annotations_scores.csv    mean per uitleg (approach/model/profiel)
  results/annotations_model.csv     mean per model x approach x wet x annotator_type

Usage (run from project root):
    uv run python analysis/llm_explanations/annotations/parse_annotations.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ANNOTATIONS_DIR = Path(__file__).parent / "input"
THESIS_DIR      = Path(__file__).parent.parent / "output" / "thesis_20260428_091358"
OUTPUT_DIR      = Path(__file__).parent / "results"

MODELS    = ["haiku", "llama3.1", "mistral", "deepseek", "gpt4"]
APPROACHES = ["graph", "open"]

LAW_SLUGS = {
    "zorgtoeslag": "zorgtoeslag",
    "bijstand":    "participatiewet_bijstand",
    "alcoholwet":  "alcoholwet_vergunning",
}

WET_KEYWORDS = {
    "zorgtoeslag": "zorgtoeslag",
    "bijstand":    "bijstand",
    "alcohol":     "alcoholwet",
}
ANNOTATOR_KEYWORDS = {
    "burger":  "burger",
    "jurist":  "jurist",
}

BURGER_GROUP_SIZE = 4
JURIST_GROUP_SIZE = 5


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def detect_meta(path: Path) -> tuple[str, str]:
    """Return (wet, annotator_type) from filename, or raise ValueError."""
    name = re.sub(r"[—–\-]+", " ", path.stem.lower())
    wet = next((v for k, v in WET_KEYWORDS.items() if k in name), None)
    ann = next((v for k, v in ANNOTATOR_KEYWORDS.items() if k in name), "burger")
    if wet is None:
        raise ValueError(f"Cannot detect wet from filename: {path.name}")
    return wet, ann


# ---------------------------------------------------------------------------
# JSONL mapping builder
# ---------------------------------------------------------------------------

def build_uitleg_map(wet_slug: str) -> dict[int, dict]:
    """Build {uitleg_nr: metadata} from all JSONL files for a wet."""
    mapping: dict[int, dict] = {}
    idx = 1
    for approach in APPROACHES:
        for model in MODELS:
            fname = THESIS_DIR / model / f"{approach}_{model}_{wet_slug}.jsonl"
            if not fname.exists():
                continue
            with open(fname, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("record_type") != "explanation":
                        continue
                    mapping[idx] = {
                        "uitleg_nr":        idx,
                        "approach":         approach,
                        "model":            model,
                        "profile_name":     r.get("profile_name", ""),
                        "profile_bsn":      r.get("profile", ""),
                        "requirements_met": r.get("requirements_met"),
                    }
                    idx += 1
    return mapping


# ---------------------------------------------------------------------------
# Data start detection
# ---------------------------------------------------------------------------

def detect_data_start(df_raw: pd.DataFrame, annotator_type: str) -> int:
    """Find the column index where the first rating question starts."""
    if annotator_type == "burger":
        for i, col in enumerate(df_raw.columns):
            if "[BURGER]" in str(col).upper():
                return i
        # Fallback: check if col 4 is email
        sample_col4 = str(df_raw.columns[4]) if len(df_raw.columns) > 4 else ""
        if any(k in sample_col4.lower() for k in ("e-mail", "email", "@", "mailadres")):
            return 5
        return 4
    else:
        # Jurist: fixed header cols (Timestamp, Jaren ervaring, Specialisatie, Email)
        return 4


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

def _parse_numeric(val) -> float | None:
    if pd.isna(val):
        return None
    try:
        return float(str(val).strip().split()[0])
    except (ValueError, IndexError):
        return None


def _parse_ja_nee(val) -> float | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s.startswith("ja"):
        return 1.0
    if s.startswith("nee"):
        return 0.0
    if s.startswith("deels"):
        return 0.5
    return None


def _parse_text(val) -> str:
    return str(val).strip() if pd.notna(val) else ""


# ---------------------------------------------------------------------------
# Group parsers per form type
# ---------------------------------------------------------------------------

def _parse_burger_group(resp: pd.Series, col_base: int) -> dict:
    return {
        "duidelijkheid":          _parse_numeric(resp.iloc[col_base]),
        "actionability":          _parse_ja_nee(resp.iloc[col_base + 1]),
        "leesbaarheid":           _parse_numeric(resp.iloc[col_base + 2]),
        "toelichting":            _parse_text(resp.iloc[col_base + 3]) if col_base + 3 < len(resp) else "",
        "juridische_correctheid": None,
        "ontbrekende_informatie": None,
        "extra_aandacht":         None,
        "toelichting_q1":         "",
        "toelichting_q2":         "",
    }


def _parse_jurist_group(resp: pd.Series, col_base: int) -> dict:
    return {
        "duidelijkheid":          None,
        "actionability":          None,
        "leesbaarheid":           None,
        "toelichting":            "",
        "juridische_correctheid": _parse_numeric(resp.iloc[col_base]),
        "toelichting_q1":         _parse_text(resp.iloc[col_base + 1]) if col_base + 1 < len(resp) else "",
        "ontbrekende_informatie": _parse_ja_nee(resp.iloc[col_base + 2]) if col_base + 2 < len(resp) else None,
        "toelichting_q2":         _parse_text(resp.iloc[col_base + 3]) if col_base + 3 < len(resp) else "",
        "extra_aandacht":         _parse_ja_nee(resp.iloc[col_base + 4]) if col_base + 4 < len(resp) else None,
    }


# ---------------------------------------------------------------------------
# Form parser
# ---------------------------------------------------------------------------

def parse_form(xlsx_path: Path) -> pd.DataFrame:
    """Parse one Google Form response Excel into long-format rows."""
    wet, annotator_type = detect_meta(xlsx_path)
    wet_slug = LAW_SLUGS[wet]
    uitleg_map = build_uitleg_map(wet_slug)

    if not uitleg_map:
        print(f"  WARNING: no JSONL data for wet={wet_slug}", file=sys.stderr)
        return pd.DataFrame()

    df_raw = pd.read_excel(xlsx_path)
    data_start = detect_data_start(df_raw, annotator_type)
    group_size = BURGER_GROUP_SIZE if annotator_type == "burger" else JURIST_GROUP_SIZE
    n_uitleggen = len(uitleg_map)
    parse_group = _parse_burger_group if annotator_type == "burger" else _parse_jurist_group

    rows = []
    for resp_idx, (_, resp) in enumerate(df_raw.iterrows()):
        for uitleg_nr in range(1, n_uitleggen + 1):
            col_base = data_start + (uitleg_nr - 1) * group_size
            if col_base >= len(df_raw.columns):
                break
            meta = uitleg_map.get(uitleg_nr, {})
            rows.append({
                "respondent":       resp_idx + 1,
                "annotator_type":   annotator_type,
                "wet":              wet,
                "uitleg_nr":        uitleg_nr,
                "approach":         meta.get("approach"),
                "model":            meta.get("model"),
                "profile_name":     meta.get("profile_name"),
                "profile_bsn":      meta.get("profile_bsn"),
                "requirements_met": meta.get("requirements_met"),
                **parse_group(resp, col_base),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    burger_metrics = ["duidelijkheid", "actionability", "leesbaarheid"]
    jurist_metrics = ["juridische_correctheid", "ontbrekende_informatie", "extra_aandacht"]
    all_metrics = [c for c in burger_metrics + jurist_metrics if c in df.columns]

    per_uitleg = (
        df.groupby(["annotator_type", "wet", "uitleg_nr",
                    "approach", "model", "profile_name", "requirements_met"])[all_metrics]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )
    per_uitleg.columns = ["_".join(c).rstrip("_") for c in per_uitleg.columns]

    per_model = (
        df.groupby(["annotator_type", "wet", "approach", "model"])[all_metrics]
        .mean()
        .round(3)
        .reset_index()
    )
    per_model.columns = (
        ["annotator_type", "wet", "approach", "model"]
        + [f"{c}_mean" for c in all_metrics]
    )

    return per_uitleg, per_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(ANNOTATIONS_DIR.glob("*.xlsx"))
    if not xlsx_files:
        print("No .xlsx files found in", ANNOTATIONS_DIR)
        return

    all_long: list[pd.DataFrame] = []

    for xlsx in xlsx_files:
        try:
            wet, ann = detect_meta(xlsx)
        except ValueError as e:
            print(f"Skipping {xlsx.name}: {e}")
            continue

        print(f"Parsing: {xlsx.name}")
        print(f"  wet={wet}  annotator={ann}")
        df_long = parse_form(xlsx)
        if df_long.empty:
            print("  Skipped (empty)")
            continue
        n_resp = df_long["respondent"].nunique()
        n_uitl = df_long["uitleg_nr"].nunique()
        print(f"  {len(df_long)} rows ({n_resp} respondenten x {n_uitl} uitleggen)")
        all_long.append(df_long)

    if not all_long:
        print("No data parsed.")
        return

    combined = pd.concat(all_long, ignore_index=True)

    long_path = OUTPUT_DIR / "annotations_long.csv"
    combined.to_csv(long_path, index=False, encoding="utf-8")
    print(f"\nLong format  -> {long_path}  ({len(combined)} rijen)")

    per_uitleg, per_model = aggregate(combined)

    uitleg_path = OUTPUT_DIR / "annotations_scores.csv"
    per_uitleg.to_csv(uitleg_path, index=False, encoding="utf-8")
    print(f"Per uitleg   -> {uitleg_path}")

    model_path = OUTPUT_DIR / "annotations_model.csv"
    per_model.to_csv(model_path, index=False, encoding="utf-8")
    print(f"Per model    -> {model_path}")

    # Toelichtingen
    toel_mask = (
        combined.get("toelichting", pd.Series("", index=combined.index)).str.len().gt(0) |
        combined.get("toelichting_q1", pd.Series("", index=combined.index)).str.len().gt(0) |
        combined.get("toelichting_q2", pd.Series("", index=combined.index)).str.len().gt(0)
    )
    toelichtingen_rows = combined[toel_mask]
    if not toelichtingen_rows.empty:
        toel_cols = [c for c in ["annotator_type", "wet", "uitleg_nr", "approach", "model",
                                  "profile_name", "toelichting", "toelichting_q1", "toelichting_q2"]
                     if c in combined.columns]
        toel_path = OUTPUT_DIR / "annotations_toelichtingen.csv"
        toelichtingen_rows[toel_cols].to_csv(toel_path, index=False, encoding="utf-8")
        print(f"Toelichtingen -> {toel_path}  ({len(toelichtingen_rows)} reacties)")

    print("\n-- Scores per model x approach x wet (burgers) --")
    burgers = per_model[per_model["annotator_type"] == "burger"]
    if not burgers.empty:
        cols = [c for c in per_model.columns if c in
                ["annotator_type", "wet", "approach", "model",
                 "duidelijkheid_mean", "actionability_mean", "leesbaarheid_mean"]]
        print(burgers[cols].to_string(index=False))

    print("\n-- Scores per model x approach x wet (juristen) --")
    juristen = per_model[per_model["annotator_type"] == "jurist"]
    if not juristen.empty:
        cols = [c for c in per_model.columns if c in
                ["annotator_type", "wet", "approach", "model",
                 "juridische_correctheid_mean", "ontbrekende_informatie_mean", "extra_aandacht_mean"]]
        print(juristen[cols].to_string(index=False))
    else:
        print("  (geen juristendata)")


if __name__ == "__main__":
    main()
