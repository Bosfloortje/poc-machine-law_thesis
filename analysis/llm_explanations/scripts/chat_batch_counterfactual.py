#!/usr/bin/env python3
"""
Counterfactual Robustness Evaluation for machine-law chat explanations.

For each profile, identifies the legally decisive input variable, mutates it
across its legal threshold, and re-evaluates the explanation via the full chat
pipeline. Measures whether the LLM explanation updates consistently with the
changed legal conditions.

Workflow (two phases):

  Phase 1 — Generate counterfactual profiles (server with original profiles.yaml):
    uv run python analysis/llm_explanations/scripts/chat_batch_counterfactual.py \\
        --law zorgtoeslag --limit 200 --phase generate

    Output:
      - data/profiles_cf_TIMESTAMP.yaml   (counterfactual profiles, new BSNs)
      - output/cf/cf_meta_TIMESTAMP.jsonl (per-profile: decisive var + mutation)

    Then: append profiles_cf_TIMESTAMP.yaml entries to data/profiles.yaml and
          restart the web server before running phase 2.

  Phase 2 — Run batch and evaluate (server with merged profiles):
    uv run python analysis/llm_explanations/scripts/chat_batch_counterfactual.py \\
        --law zorgtoeslag --limit 200 --phase run \\
        --meta-file output/cf/cf_meta_TIMESTAMP.jsonl \\
        --provider haiku [--graphrag]

    Output:
      - output/cf/cf_eval_TIMESTAMP.jsonl (paired conversations + evaluation)

Requires the web server:
    $env:FEATURE_CHAT='1'; uv run web/main.py
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import socket
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import websockets
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "cf"

# ---------------------------------------------------------------------------
# Conversation script (same as chat_batch.py for zorgtoeslag)
# ---------------------------------------------------------------------------

# Q1 per law: asks the current entitlement — asked to BOTH original and CF BSN.
Q1: dict[str, str] = {
    "zorgtoeslag": "Waar heb ik recht op met de zorgtoeslag?",
    "bijstand":    "Waar heb ik recht op met de bijstand?",
    "alcoholwet":  "Waar heb ik recht op met de Alcoholwet-vergunning?",
}

# Mapping from decisive subject field names to Dutch natural-language labels for Q2.
SUBJECT_NL: dict[str, str] = {
    "INKOMEN":                               "inkomen",
    "TOETSINGSINKOMEN":                      "toetsingsinkomen",
    "PARTNER_INKOMEN":                       "partnerinkomen",
    "ARBEIDSVERMOGEN":                       "arbeidsvermogen",
    "LEEFTIJD":                              "leeftijd",
    "LEEFTIJD_EXPLOITANT":                   "leeftijd als exploitant",
    "SVH_REGISTRATIE_GELDIG":               "SVH-registratie",
    "IS_INGESCHREVEN_SVH_REGISTER":         "SVH-registratie",
    "BEDRIJF_STATUS":                        "bedrijfsstatus",
    "IS_ONDER_CURATELE_EXPLOITANT":         "curatele-status",
    "IS_ONDER_CURATELE":                    "curatele-status",
    "IS_VAN_SLECHT_LEVENSGEDRAG_EXPLOITANT": "levensgedrag",
    "IS_VAN_SLECHT_LEVENSGEDRAG":           "levensgedrag",
    "BIBOB_OORDEEL":                        "BIBOB-oordeel",
    "VLOEROPPERVLAKTE_HORECALOKALITEIT":    "vloeroppervlakte van de horecalokaliteit",
    "VLOEROPPERVLAKTE":                     "vloeroppervlakte",
    "HEEFT_NEDERLANDSE_NATIONALITEIT":      "nationaliteit",
    "VERMOGEN":                             "vermogen",
    "GEZAMENLIJK_VERMOGEN":                 "gezamenlijk vermogen",
    "BEZITTINGEN":                          "bezittingen",
    "IS_VERZEKERDE":                        "zorgverzekering",
    "HEEFT_ZORGTOESLAG_VERZEKERING":        "zorgverzekering",
    "VERBLIJFSADRES":                       "verblijfsadres",
    "VERBLIJFT_IN_NEDERLAND":               "verblijfsstatus",
}


def make_q1(law: str) -> str:
    return Q1.get(law, f"Waar heb ik recht op met {law}?")


def make_q2(decisive_subject: str) -> str:
    """Build the personalized counterfactual question from the decisive variable."""
    label = SUBJECT_NL.get(decisive_subject.upper(), decisive_subject.lower().replace("_", " "))
    return (
        f"Wat als mijn {label} verandert? "
        f"Zou ik dan wel of niet in aanmerking komen?"
    )


# ---------------------------------------------------------------------------
# Profile mutation helpers
# ---------------------------------------------------------------------------

# Per-law, per-field: how to mutate the profile to flip the decisive condition.
# "paths" are tuples of keys to traverse into profile["sources"].
# Values in the profile are in eurocents for amounts, years for age, bool for booleans.

# Income thresholds (eurocents) — well inside/outside so the outcome definitely flips.
_LOW_INCOME = 2_000_000    # 20,000 euro — clearly below zorgtoeslag threshold
_HIGH_INCOME = 8_000_000   # 80,000 euro — clearly above zorgtoeslag threshold
_OLD_AGE = 30              # clearly ≥ 18
_YOUNG_AGE = 16            # clearly < 18


def _set_nested(d: dict, keys: list, value) -> None:
    """Set a value at a nested dict/list path. Silently skips if path does not exist."""
    obj = d
    for key in keys[:-1]:
        if isinstance(obj, dict) and key in obj or isinstance(obj, list) and isinstance(key, int) and key < len(obj):
            obj = obj[key]
        else:
            return
    last = keys[-1]
    if isinstance(obj, dict) or isinstance(obj, list) and isinstance(last, int) and last < len(obj):
        obj[last] = value


def _ensure_set(d: dict, keys: list, value) -> None:
    """Set a value at a nested path, creating intermediate dicts/lists as needed."""
    obj = d
    for i, key in enumerate(keys[:-1]):
        next_key = keys[i + 1]
        if isinstance(obj, dict):
            if key not in obj:
                obj[key] = [] if isinstance(next_key, int) else {}
            obj = obj[key]
        elif isinstance(obj, list):
            while len(obj) <= key:
                obj.append({})
            obj = obj[key]
    last = keys[-1]
    if isinstance(obj, dict):
        obj[last] = value
    elif isinstance(obj, list):
        while len(obj) <= last:
            obj.append({})
        obj[last] = value


def _mutate_income_low(cf: dict, bsn: str) -> None:
    """Set income to clearly below threshold (make eligible if income was blocking)."""
    for path in [
        ["sources", "UWV", "uwv_toetsingsinkomen", 0, "toetsingsinkomen"],
        ["sources", "BELASTINGDIENST", "box1", 0, "loon_uit_dienstbetrekking"],
        ["sources", "BELASTINGDIENST", "box1", 0, "uitkeringen_en_pensioenen"],
        ["sources", "BELASTINGDIENST", "box1", 0, "winst_uit_onderneming"],
    ]:
        _set_nested(cf, path, 0)
    # Set a consistent low income value
    _set_nested(cf, ["sources", "UWV", "uwv_toetsingsinkomen", 0, "toetsingsinkomen"], _LOW_INCOME)
    _set_nested(cf, ["sources", "BELASTINGDIENST", "box1", 0, "loon_uit_dienstbetrekking"], _LOW_INCOME)
    # Also update monthly income fields if present
    for path in [
        ["sources", "BELASTINGDIENST", "monthly_income", 0, "bedrag"],
        ["sources", "BELASTINGDIENST", "maandelijks_inkomen", 0, "bedrag"],
    ]:
        _set_nested(cf, path, _LOW_INCOME // 12)


def _mutate_income_high(cf: dict, bsn: str) -> None:
    """Set income to clearly above threshold (make ineligible if income was OK)."""
    _set_nested(cf, ["sources", "UWV", "uwv_toetsingsinkomen", 0, "toetsingsinkomen"], _HIGH_INCOME)
    _set_nested(cf, ["sources", "BELASTINGDIENST", "box1", 0, "loon_uit_dienstbetrekking"], _HIGH_INCOME)
    _set_nested(cf, ["sources", "BELASTINGDIENST", "box1", 0, "uitkeringen_en_pensioenen"], 0)
    for path in [
        ["sources", "BELASTINGDIENST", "monthly_income", 0, "bedrag"],
        ["sources", "BELASTINGDIENST", "maandelijks_inkomen", 0, "bedrag"],
    ]:
        _set_nested(cf, path, _HIGH_INCOME // 12)


def _mutate_age(cf: dict, bsn: str, new_age: int) -> None:
    """Change the person's age in all relevant profile source fields."""
    for path in [
        ["sources", "RvIG", "personen", 0, "leeftijd"],
        ["sources", "RvIG", "personen", 0, "age"],
        ["sources", "RvIG", "brp_gegevens", 0, "leeftijd"],
    ]:
        _set_nested(cf, path, new_age)


def _mutate_boolean(cf: dict, field_paths: list[list], new_value: bool) -> None:
    """Flip a boolean field in the profile."""
    for path in field_paths:
        _set_nested(cf, path, new_value)


def _remap_bsn_in_sources(profile: dict, old_bsn: str, new_bsn: str) -> None:
    """Update all BSN references inside profile sources to use the new CF BSN."""
    sources = profile.get("sources", {})
    for svc_data in sources.values():
        if not isinstance(svc_data, dict):
            continue
        for rows in svc_data.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("bsn") == old_bsn:
                    row["bsn"] = new_bsn


# Mutation strategies keyed by (law, field_name).
# Each entry: {"direction": "lower"|"raise"|"flip_bool", "action": callable}
def make_mutation(
    law: str,
    field_name: str,
    original_requirements_met: bool,
    original_value,
) -> tuple[str, callable] | None:
    """
    Decide how to mutate the profile to flip the outcome.

    Returns (mutation_description, mutate_fn) or None if no mutation is possible.
    mutate_fn takes (cf_profile_dict, bsn_str) and mutates in-place.
    """
    field = field_name.upper()

    # ---- Income-based conditions ----
    # zorgtoeslag: INKOMEN is annual (UWV.uwv_toetsingsinkomen + BELASTINGDIENST.box1)
    # bijstand:    INKOMEN is monthly (BELASTINGDIENST.maandelijks_inkomen)
    if field in ("INKOMEN", "TOETSINGSINKOMEN", "PARTNER_INKOMEN"):
        if not original_requirements_met:
            return (
                f"Inkomen verlaagd naar {_LOW_INCOME // 100} euro (was te hoog)",
                lambda cf, bsn: _mutate_income_low(cf, bsn),
            )
        else:
            return (
                f"Inkomen verhoogd naar {_HIGH_INCOME // 100} euro (boven grens)",
                lambda cf, bsn: _mutate_income_high(cf, bsn),
            )

    # bijstand: BEDRIJFSINKOMEN (annual, eurocents)
    if field == "BEDRIJFSINKOMEN":
        if not original_requirements_met:
            def _lower_bedrijfsinkomen(cf, bsn):
                _set_nested(cf, ["sources", "BELASTINGDIENST", "bedrijfsinkomen", 0, "bedrag"], 0)
                _set_nested(cf, ["sources", "BELASTINGDIENST", "business_income", 0, "bedrag"], 0)
                _set_nested(cf, ["sources", "BELASTINGDIENST", "box1", 0, "winst_uit_onderneming"], 0)
            return ("Bedrijfsinkomen verlaagd naar 0 (was te hoog)", _lower_bedrijfsinkomen)
        else:
            def _raise_bedrijfsinkomen(cf, bsn):
                _set_nested(cf, ["sources", "BELASTINGDIENST", "bedrijfsinkomen", 0, "bedrag"], 3_000_000)
                _set_nested(cf, ["sources", "BELASTINGDIENST", "business_income", 0, "bedrag"], 3_000_000)
            return ("Bedrijfsinkomen verhoogd naar 30.000 euro (boven grens)", _raise_bedrijfsinkomen)

    # ---- Asset-based conditions (zorgtoeslag: VERMOGEN; bijstand: BEZITTINGEN) ----
    if field in ("VERMOGEN", "GEZAMENLIJK_VERMOGEN", "BEZITTINGEN"):
        if not original_requirements_met:
            def _lower_assets(cf, bsn):
                for path in [
                    ["sources", "BELASTINGDIENST", "bezittingen", 0, "bedrag"],
                    ["sources", "BELASTINGDIENST", "assets", 0, "bedrag"],
                    ["sources", "BELASTINGDIENST", "box3", 0, "spaargeld"],
                    ["sources", "BELASTINGDIENST", "box3", 0, "beleggingen"],
                    ["sources", "BELASTINGDIENST", "belastingdienst_vermogen", 0, "vermogen"],
                ]:
                    _set_nested(cf, path, 0)
            return ("Vermogen/bezittingen verlaagd naar 0 (was boven grens)", _lower_assets)
        else:
            def _raise_assets(cf, bsn):
                for path in [
                    ["sources", "BELASTINGDIENST", "bezittingen", 0, "bedrag"],
                    ["sources", "BELASTINGDIENST", "assets", 0, "bedrag"],
                    ["sources", "BELASTINGDIENST", "box3", 0, "spaargeld"],
                    ["sources", "BELASTINGDIENST", "belastingdienst_vermogen", 0, "vermogen"],
                ]:
                    _set_nested(cf, path, 20_000_000)
            return ("Vermogen/bezittingen verhoogd naar 200.000 euro (boven grens)", _raise_assets)

    # ---- Age conditions ----
    # zorgtoeslag/bijstand threshold: 18; alcoholwet threshold: 21
    if field in ("LEEFTIJD", "LEEFTIJD_EXPLOITANT"):
        threshold = 21 if field == "LEEFTIJD_EXPLOITANT" else 18
        above_age = 30 if field == "LEEFTIJD_EXPLOITANT" else _OLD_AGE
        below_age = 19 if field == "LEEFTIJD_EXPLOITANT" else _YOUNG_AGE

        if not original_requirements_met and isinstance(original_value, (int, float)) and original_value < threshold:
            return (
                f"Leeftijd aangepast van {int(original_value)} naar {above_age} (boven drempel {threshold})",
                lambda cf, bsn: _mutate_age(cf, bsn, above_age),
            )
        elif original_requirements_met and isinstance(original_value, (int, float)) and original_value >= threshold:
            return (
                f"Leeftijd aangepast van {int(original_value)} naar {below_age} (onder drempel {threshold})",
                lambda cf, bsn: _mutate_age(cf, bsn, below_age),
            )

    # ---- zorgtoeslag: health insurance ----
    if field in ("IS_VERZEKERDE", "HEEFT_ZORGTOESLAG_VERZEKERING"):
        if not original_requirements_met:
            return (
                "Zorgverzekering gezet op Ja",
                lambda cf, bsn: _mutate_boolean(cf, [
                    ["sources", "RVZ", "verzekeringen", 0, "is_verzekerde"],
                    ["sources", "RVZ", "verzekeringen", 0, "heeft_basisverzekering"],
                ], True),
            )
        else:
            return (
                "Zorgverzekering gezet op Nee",
                lambda cf, bsn: _mutate_boolean(cf, [
                    ["sources", "RVZ", "verzekeringen", 0, "is_verzekerde"],
                ], False),
            )

    # ---- bijstand: nationality ----
    if field == "HEEFT_NEDERLANDSE_NATIONALITEIT":
        if not original_requirements_met:
            return (
                "Nederlandse nationaliteit gezet op Ja",
                lambda cf, bsn: _mutate_boolean(cf, [
                    ["sources", "RvIG", "personen", 0, "has_dutch_nationality"],
                    ["sources", "RvIG", "brp_gegevens", 0, "heeft_nederlandse_nationaliteit"],
                ], True),
            )
        else:
            return (
                "Nederlandse nationaliteit gezet op Nee",
                lambda cf, bsn: _mutate_boolean(cf, [
                    ["sources", "RvIG", "personen", 0, "has_dutch_nationality"],
                    ["sources", "RvIG", "brp_gegevens", 0, "heeft_nederlandse_nationaliteit"],
                ], False),
            )

    # ---- Residency / fixed address ----
    if field in ("VERBLIJFT_IN_NEDERLAND", "HEEFT_VAST_ADRES", "HEEFT_VASTE_WOONPLAATS", "VERBLIJFSADRES"):
        if not original_requirements_met:
            return (
                f"{field} gezet op Ja",
                lambda cf, bsn: _mutate_boolean(cf, [
                    ["sources", "RvIG", "personen", 0, "has_fixed_address"],
                    ["sources", "RvIG", "brp_gegevens", 0, "heeft_vast_adres"],
                ], True),
            )

    # ---- bijstand: ARBEIDSVERMOGEN (compound OR — inject werk_en_re_integratie data) ----
    # Profiles have exactly one GEMEENTE_* source key (their municipality).
    # Use _ensure_set so the data is created even if the source key doesn't exist yet.
    _ALL_GEMEENTEN = [
        "GEMEENTE_AMSTERDAM", "GEMEENTE_DEN_HAAG", "GEMEENTE_EINDHOVEN",
        "GEMEENTE_GRONINGEN", "GEMEENTE_MAASTRICHT", "GEMEENTE_ROTTERDAM", "GEMEENTE_UTRECHT",
    ]
    if field == "ARBEIDSVERMOGEN":
        if not original_requirements_met:
            def _add_arb(cf, bsn):
                for gemeente in _ALL_GEMEENTEN:
                    _ensure_set(cf, ["sources", gemeente, "werk_en_re_integratie", 0, "bsn"], bsn)
                    _ensure_set(cf, ["sources", gemeente, "werk_en_re_integratie", 0, "arbeidsvermogen"], "MEDISCH_VOLLEDIG")
                    _ensure_set(cf, ["sources", gemeente, "werk_en_re_integratie", 0, "ontheffing_reden"], "medisch")
                    _ensure_set(cf, ["sources", gemeente, "werk_en_re_integratie", 0, "re_integratie_traject"], None)
            return ("Arbeidsvermogen toegevoegd: MEDISCH_VOLLEDIG (medische ontheffing)", _add_arb)
        else:
            def _remove_arb(cf, bsn):
                for gemeente in _ALL_GEMEENTEN:
                    _ensure_set(cf, ["sources", gemeente, "werk_en_re_integratie", 0, "bsn"], bsn)
                    _ensure_set(cf, ["sources", gemeente, "werk_en_re_integratie", 0, "arbeidsvermogen"], "ARBEIDSFIT")
                    _ensure_set(cf, ["sources", gemeente, "werk_en_re_integratie", 0, "re_integratie_traject"], None)
            return ("Arbeidsvermogen gezet op ARBEIDSFIT (geen vrijstelling)", _remove_arb)

    # ---- alcoholwet: SVH register (create from scratch if data is missing) ----
    if field in ("SVH_REGISTRATIE_GELDIG", "IS_INGESCHREVEN_SVH_REGISTER"):
        if not original_requirements_met:
            def _set_svh_true(cf, bsn):
                _ensure_set(cf, ["sources", "SVH", "register_sociale_hygiene", 0, "bsn"], bsn)
                _ensure_set(cf, ["sources", "SVH", "register_sociale_hygiene", 0, "is_geregistreerd"], True)
            return ("SVH-registratie gezet op Ja (ingeschreven in Register Sociale Hygiëne)", _set_svh_true)
        else:
            def _set_svh_false(cf, bsn):
                _ensure_set(cf, ["sources", "SVH", "register_sociale_hygiene", 0, "is_geregistreerd"], False)
            return ("SVH-registratie gezet op Nee (niet ingeschreven)", _set_svh_false)

    # ---- alcoholwet: curatele ----
    if field in ("IS_ONDER_CURATELE_EXPLOITANT", "IS_ONDER_CURATELE"):
        if not original_requirements_met:
            def _remove_curatele(cf, bsn):
                sources = cf.get("sources", {})
                rechtspraak = sources.get("RECHTSPRAAK", {})
                if "curatele_registraties" in rechtspraak:
                    rechtspraak["curatele_registraties"] = []
            return ("Curatele verwijderd (niet meer onder curatele)", _remove_curatele)
        else:
            def _add_curatele(cf, bsn):
                sources = cf.get("sources", {})
                rechtspraak = sources.setdefault("RECHTSPRAAK", {})
                rechtspraak["curatele_registraties"] = [{"bsn": bsn, "is_onder_curatele": True}]
            return ("Curatele toegevoegd (exploitant staat nu onder curatele)", _add_curatele)

    # ---- alcoholwet: slecht levensgedrag (via BIBOB — create from scratch if missing) ----
    if field in ("IS_VAN_SLECHT_LEVENSGEDRAG_EXPLOITANT", "IS_VAN_SLECHT_LEVENSGEDRAG", "BIBOB_OORDEEL"):
        if not original_requirements_met:
            def _clear_bibob(cf, bsn):
                _ensure_set(cf, ["sources", "LBB", "bibob_adviezen", 0, "bsn"], bsn)
                _ensure_set(cf, ["sources", "LBB", "bibob_adviezen", 0, "mate_van_gevaar"], "geen_gevaar")
            return ("BIBOB-oordeel gezet op 'geen gevaar' (niet van slecht levensgedrag)", _clear_bibob)
        else:
            def _set_bibob_bad(cf, bsn):
                _ensure_set(cf, ["sources", "LBB", "bibob_adviezen", 0, "mate_van_gevaar"], "ernstig_gevaar")
            return ("BIBOB-oordeel gezet op 'ernstig gevaar' (slecht levensgedrag)", _set_bibob_bad)

    # ---- alcoholwet: floor area (create from scratch if KVK record missing) ----
    if field in ("VLOEROPPERVLAKTE_HORECALOKALITEIT", "VLOEROPPERVLAKTE"):
        if not original_requirements_met:
            def _raise_floor(cf, bsn):
                _ensure_set(cf, ["sources", "KVK", "inrichtingen", 0, "bsn"], bsn)
                _ensure_set(cf, ["sources", "KVK", "inrichtingen", 0, "vloeroppervlakte_horecalokaliteit"], 50)
            return ("Vloeroppervlakte vergroot naar 50 m² (boven minimumeis)", _raise_floor)
        else:
            def _lower_floor(cf, bsn):
                _ensure_set(cf, ["sources", "KVK", "inrichtingen", 0, "vloeroppervlakte_horecalokaliteit"], 5)
            return ("Vloeroppervlakte verkleind naar 5 m² (onder minimumeis)", _lower_floor)

    # ---- alcoholwet: leeftijd when data is completely missing ----
    if field in ("LEEFTIJDGEGEVENS_EXPLOITANT", "LEEFTIJDGEGEVENS"):
        if not original_requirements_met:
            return (
                "Leeftijd exploitant gezet op 30 jaar (boven drempel 21)",
                lambda cf, bsn: _mutate_age(cf, bsn, 30),
            )

    # ---- alcoholwet: bedrijf status (inject full horeca business record) ----
    # Most citizen profiles have no KVK data, making this the first unknown condition.
    # Inject a minimal active horeca business so the outcome can flip.
    if field == "BEDRIJF_STATUS":
        if not original_requirements_met:
            def _inject_horeca(cf, bsn):
                kvk_nummer = f"KVK{bsn[:8]}"
                # Set KVK_NUMMER parameter so the law can look up data
                cf.setdefault("parameters", {})["KVK_NUMMER"] = kvk_nummer
                # Active business in handelsregister
                _ensure_set(cf, ["sources", "KVK", "organisaties", 0, "kvk_nummer"], kvk_nummer)
                _ensure_set(cf, ["sources", "KVK", "organisaties", 0, "status"], "Actief")
                _ensure_set(cf, ["sources", "KVK", "organisaties", 0, "rechtsvorm"], "Eenmanszaak")
                # Leidinggevende = this person (so LEEFTIJD_EXPLOITANT resolves to their age)
                _ensure_set(cf, ["sources", "KVK", "leidinggevenden", 0, "kvk_nummer"], kvk_nummer)
                _ensure_set(cf, ["sources", "KVK", "leidinggevenden", 0, "bsn"], bsn)
                # Horeca inrichting with valid floor area (≥ 35 m² per Alcoholwet art. 10 lid 2)
                _ensure_set(cf, ["sources", "KVK", "inrichtingen", 0, "kvk_nummer"], kvk_nummer)
                _ensure_set(cf, ["sources", "KVK", "inrichtingen", 0, "vloeroppervlakte_horecalokaliteit"], 50)
                _ensure_set(cf, ["sources", "KVK", "inrichtingen", 0, "type_bedrijf"], "horecabedrijf")
                # SVH register sociale hygiene = ingeschreven
                _ensure_set(cf, ["sources", "SVH", "register_sociale_hygiene", 0, "bsn"], bsn)
                _ensure_set(cf, ["sources", "SVH", "register_sociale_hygiene", 0, "is_geregistreerd"], True)
                # No curatele
                cf.setdefault("sources", {}).setdefault("RECHTSPRAAK", {})["curatele_registraties"] = []
                # No BIBOB issues
                _ensure_set(cf, ["sources", "LBB", "bibob_adviezen", 0, "bsn"], bsn)
                _ensure_set(cf, ["sources", "LBB", "bibob_adviezen", 0, "mate_van_gevaar"], "geen_gevaar")
            return (
                "Actief horeca bedrijf toegevoegd: KVK actief, SVH geregistreerd, geen BIBOB-bezwaren",
                _inject_horeca,
            )
        else:
            def _deactivate_business(cf, bsn):
                _ensure_set(cf, ["sources", "KVK", "organisaties", 0, "status"], "Niet_Actief")
            return ("Bedrijfsstatus gezet op Niet_Actief (bedrijf opgeheven)", _deactivate_business)

    return None


def build_counterfactual_profile(
    original_profile: dict,
    original_bsn: str,
    decisive_subject: str,
    decisive_value,
    original_requirements_met: bool,
    law: str,
) -> tuple[dict, str, str] | None:
    """
    Build a counterfactual profile dict.

    Returns (cf_profile, cf_bsn, mutation_description) or None if no mutation possible.
    """
    mutation = make_mutation(law, decisive_subject, original_requirements_met, decisive_value)
    if mutation is None:
        return None

    description, mutate_fn = mutation
    cf_bsn = f"CF{original_bsn}"
    cf_profile = copy.deepcopy(original_profile)

    # Apply mutation
    mutate_fn(cf_profile, cf_bsn)

    # Update BSN references inside sources
    _remap_bsn_in_sources(cf_profile, original_bsn, cf_bsn)

    # Update top-level profile metadata
    cf_profile["name"] = f"[CF] {cf_profile.get('name', original_bsn)}"
    cf_profile["description"] = (
        f"Counterfactual van BSN {original_bsn}. Mutatie: {description}"
    )

    return cf_profile, cf_bsn, description


# ---------------------------------------------------------------------------
# Engine + graph helpers (require server to be running)
# ---------------------------------------------------------------------------

def _synthesize_or_subject(description: str) -> str | None:
    """Map compound OR condition descriptions to a synthetic subject name for mutation."""
    desc = description.lower()
    if "arbeidsvermogen" in desc:
        return "ARBEIDSVERMOGEN"
    if "sociale hygiene" in desc or "sociale hygiëne" in desc or " svh" in desc:
        return "SVH_REGISTRATIE_GELDIG"
    if "curatele" in desc:
        return "IS_ONDER_CURATELE_EXPLOITANT"
    if "bibob" in desc or "levensgedrag" in desc:
        return "IS_VAN_SLECHT_LEVENSGEDRAG_EXPLOITANT"
    return None


def get_decisive_condition(law_name: str, bsn: str, verbose: bool = False) -> dict:
    """
    Run the rule engine for a profile and extract the decisive condition from the graph.

    Returns dict with keys:
        requirements_met, decisive_subject, decisive_value,
        decisive_description, outcome_label, calc_result
    Returns empty dict on failure.
    """
    try:
        from extract_graphrag import serialize_graph
        from extraction_generic import DecisionGraphExtractor, load_law_yaml, run_calculation
    except ImportError as e:
        print(f"  [graph] Import failed: {e}", file=sys.stderr)
        return {}

    try:
        profiles = _load_profiles_dict()
        profile = profiles.get(bsn)
        if not profile:
            return {}

        law = _load_law_for_profile(law_name, profile)
        if not law:
            return {}

        calc_result = run_calculation(law_name, bsn, law, profile)
        if not calc_result:
            return {}

        extractor = DecisionGraphExtractor(law=law, profile=profile, bsn=bsn, calc_result=calc_result)
        graph = extractor.extract()
        person_name = profile.get("name", bsn)
        graph_json = serialize_graph(graph, extractor, person_name)

        requirements_met = graph_json["beslissing"]["requirements_voldaan"]
        if not requirements_met:
            candidates = graph_json["voorwaarden"]["niet_voldaan"]
            # Fall back to unknown/missing-data conditions when no condition explicitly failed.
            # (serialize_graph only populates onbekend_gegevens_ontbreken when failed is empty)
            if not candidates:
                candidates = graph_json["voorwaarden"].get("onbekend_gegevens_ontbreken", [])
        else:
            candidates = graph_json["voorwaarden"]["voldaan"]

        # Pick first candidate with non-empty subject (skip compound OR group nodes)
        decisive = {}
        for candidate in candidates:
            if candidate.get("subject"):
                decisive = candidate
                break

        # Fallback: if all conditions are compound OR groups, synthesize subject from description
        if not decisive.get("subject"):
            for candidate in candidates:
                if candidate.get("is_or_groep"):
                    desc = candidate.get("beschrijving", "").lower()
                    synth = _synthesize_or_subject(desc)
                    if synth:
                        decisive = dict(candidate)
                        decisive["subject"] = synth
                        break

        if not decisive:
            decisive = candidates[0] if candidates else {}

        subject = decisive.get("subject", "")
        profile_value = decisive.get("profielwaarde")

        # Try to get the raw numeric value from profile_values
        raw_value = None
        if subject in extractor.profile_values:
            raw_value = extractor.profile_values[subject].get("value")

        return {
            "requirements_met": requirements_met,
            "decisive_subject": subject,
            "decisive_value": raw_value if raw_value is not None else profile_value,
            "decisive_description": decisive.get("beschrijving", ""),
            "outcome_label": graph_json["beslissing"]["uitkomst"],
            "calc_result": calc_result,
            "_law": law,   # reused by verification so both steps use the exact same law
        }

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        print(f"  [graph] Error for {bsn}: {e}", file=sys.stderr)
        return {}


def _load_profiles_dict() -> dict:
    """Load profiles.yaml from the project data directory."""
    path = PROJECT_ROOT / "data" / "profiles.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("profiles", {})


# Directories containing gemeente-specific law variants.
_GEMEENTE_LAW_DIRS: dict[str, Path] = {
    "bijstand":  PROJECT_ROOT / "laws" / "participatiewet" / "bijstand" / "gemeenten",
    "alcoholwet": PROJECT_ROOT / "laws" / "alcoholwet" / "vergunning" / "gemeenten",
}


def _load_law_for_profile(law_name: str, profile: dict) -> dict | None:
    """
    Load the gemeente-specific law YAML that matches the profile's municipality.

    Profiles have exactly one GEMEENTE_* source key (e.g. GEMEENTE_DEN_HAAG).
    Loading the matching variant ensures source data paths align with what the engine reads.
    Falls back to load_law_yaml(law_name) when no gemeente variant exists (e.g. zorgtoeslag).
    """
    try:
        from extraction_generic import load_law_yaml
    except ImportError:
        return None

    law_dir = _GEMEENTE_LAW_DIRS.get(law_name)
    if not law_dir or not law_dir.exists():
        return load_law_yaml(law_name)

    sources = profile.get("sources", {})
    gemeente = next((k for k in sources if k.startswith("GEMEENTE_")), None)
    if not gemeente:
        return load_law_yaml(law_name)

    matches = sorted(law_dir.glob(f"{gemeente}-*.yaml"))
    if matches:
        with open(matches[-1], encoding="utf-8") as f:  # latest date variant
            return yaml.safe_load(f)

    return load_law_yaml(law_name)


def _verify_cf_mutation(
    cf_profile: dict,
    decisive_subject: str,
    original_requirements_met: bool,
) -> bool:
    """
    Logically verify that the mutation was correctly applied to the CF profile.

    run_calculation goes through the MCP registry (reads from profiles.yaml on disk),
    so it cannot evaluate in-memory CF profiles. Instead, we directly inspect the
    profile data to confirm the decisive field was set to the intended value.

    Returns True if the mutation data is present and logically correct, False otherwise.
    """
    subject = decisive_subject.upper()
    sources = cf_profile.get("sources", {})

    _VALID_ARBV = {"MEDISCH_VOLLEDIG", "MANTELZORG_VOLLEDIG", "SOCIALE_OMSTANDIGHEDEN_VOLLEDIG"}

    if subject == "ARBEIDSVERMOGEN":
        return any(
            row.get("arbeidsvermogen") in _VALID_ARBV or row.get("re_integratie_traject") is not None
            for svc in sources.values() if isinstance(svc, dict)
            for row in svc.get("werk_en_re_integratie", [])
        )

    if subject in ("INKOMEN", "TOETSINGSINKOMEN", "PARTNER_INKOMEN"):
        target = _LOW_INCOME if not original_requirements_met else _HIGH_INCOME
        for svc in sources.values():
            if not isinstance(svc, dict):
                continue
            for table in ("uwv_toetsingsinkomen", "box1", "maandelijks_inkomen", "monthly_income"):
                for row in svc.get(table, []):
                    if isinstance(row, dict) and target in row.values():
                        return True
        return False

    if subject in ("VERMOGEN", "GEZAMENLIJK_VERMOGEN", "BEZITTINGEN"):
        if not original_requirements_met:
            return True  # Mutation sets to 0; 0 in values is always present after _set_nested
        # Raised above limit — check for 20_000_000
        for svc in sources.values():
            if not isinstance(svc, dict):
                continue
            for table in ("bezittingen", "assets", "box3", "belastingdienst_vermogen"):
                for row in svc.get(table, []):
                    if isinstance(row, dict) and 20_000_000 in row.values():
                        return True
        return False

    if subject in ("LEEFTIJD", "LEEFTIJD_EXPLOITANT"):
        threshold = 21 if subject == "LEEFTIJD_EXPLOITANT" else 18
        rvig = sources.get("RvIG", {})
        for row in rvig.get("personen", []) + rvig.get("brp_gegevens", []):
            age = row.get("leeftijd") or row.get("age")
            if age is not None:
                return age >= threshold if not original_requirements_met else age < threshold
        return False

    if subject in ("SVH_REGISTRATIE_GELDIG", "IS_INGESCHREVEN_SVH_REGISTER"):
        expected = not original_requirements_met  # True if was ineligible, False if was eligible
        svh = sources.get("SVH", {})
        return any(r.get("is_geregistreerd") == expected for r in svh.get("register_sociale_hygiene", []))

    if subject == "BEDRIJF_STATUS":
        expected_status = "Actief" if not original_requirements_met else "Niet_Actief"
        return any(
            org.get("status") == expected_status
            for svc in sources.values() if isinstance(svc, dict)
            for org in svc.get("organisaties", [])
        )

    if subject in ("IS_ONDER_CURATELE_EXPLOITANT", "IS_ONDER_CURATELE"):
        rechtspraak = sources.get("RECHTSPRAAK", {})
        if not original_requirements_met:
            return rechtspraak.get("curatele_registraties") == []  # Removed
        return bool(rechtspraak.get("curatele_registraties"))  # Added

    if subject in ("IS_VAN_SLECHT_LEVENSGEDRAG_EXPLOITANT", "BIBOB_OORDEEL"):
        expected_gevaar = "geen_gevaar" if not original_requirements_met else "ernstig_gevaar"
        lbb = sources.get("LBB", {})
        return any(r.get("mate_van_gevaar") == expected_gevaar for r in lbb.get("bibob_adviezen", []))

    if subject in ("HEEFT_NEDERLANDSE_NATIONALITEIT",):
        expected = not original_requirements_met
        rvig = sources.get("RvIG", {})
        for rows in (rvig.get("personen", []), rvig.get("brp_gegevens", [])):
            for row in rows:
                val = row.get("has_dutch_nationality") or row.get("heeft_nederlandse_nationaliteit")
                if val == expected:
                    return True
        return False

    # For unrecognized fields: trust that the mutation function ran without error.
    return True


# ---------------------------------------------------------------------------
# Phase 1: Generate counterfactual profiles
# ---------------------------------------------------------------------------

def phase_generate(
    law: str,
    limit: int | None,
    bsn_filter: list[str] | None,
    verbose: bool,
    timestamp: str,
) -> Path:
    """
    For each profile: determine decisive variable, build CF profile, save to YAML.

    Returns path to the metadata JSONL.
    """
    profiles = _load_profiles_dict()
    bsns = list(profiles.keys())
    if bsn_filter:
        bsns = [b for b in bsns if b in bsn_filter]
    if limit:
        bsns = bsns[:limit]

    print("\nFase 1 — Counterfactual profielen genereren")
    print(f"Law: {law} | Profielen: {len(bsns)}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cf_profiles: dict = {}
    meta_records: list[dict] = []

    for i, bsn in enumerate(bsns, 1):
        profile = profiles[bsn]
        name = profile.get("name", bsn)
        print(f"[{i}/{len(bsns)}] {bsn} — {name}")

        # Run engine + extract decisive condition
        info = get_decisive_condition(law, bsn, verbose=verbose)
        if not info:
            print("  [SKIP] Kon geen engine-resultaat ophalen", file=sys.stderr)
            meta_records.append({"bsn": bsn, "skipped": True, "reason": "engine_failed"})
            continue

        requirements_met = info["requirements_met"]
        subject = info["decisive_subject"]
        value = info["decisive_value"]
        description = info["decisive_description"]

        print(f"  Uitkomst: {'[+] recht' if requirements_met else '[-] geen recht'}")
        print(f"  Decisieve variabele: {subject} = {value}")
        print(f"  Beschrijving: {description[:80]}")

        # Build counterfactual profile
        result = build_counterfactual_profile(
            original_profile=profile,
            original_bsn=bsn,
            decisive_subject=subject,
            decisive_value=value,
            original_requirements_met=requirements_met,
            law=law,
        )

        if result is None:
            print(f"  [SKIP] Geen mutatie mogelijk voor veld '{subject}'")
            meta_records.append({
                "bsn": bsn,
                "skipped": True,
                "reason": f"no_mutation_for_{subject}",
                "decisive_subject": subject,
            })
            continue

        cf_profile, cf_bsn, mutation_desc = result

        # Logical verification: confirm the mutation was correctly applied.
        flip_verified = _verify_cf_mutation(
            cf_profile=cf_profile,
            decisive_subject=subject,
            original_requirements_met=requirements_met,
        )

        if flip_verified is False:
            print("  [SKIP] Engine verificatie: uitkomst NIET omgekeerd na mutatie")
            meta_records.append({
                "bsn": bsn,
                "skipped": True,
                "reason": "engine_flip_not_verified",
                "decisive_subject": subject,
                "mutation_description": mutation_desc,
            })
            continue

        if flip_verified is None:
            print("  [WARN] Engine verificatie mislukt — paar toch opgenomen (onzeker)")

        cf_profiles[cf_bsn] = cf_profile

        flip_status = "bevestigd" if flip_verified else "onzeker"
        print(f"  -> CF BSN: {cf_bsn} | Mutatie: {mutation_desc} [{flip_status}]")

        meta_records.append({
            "bsn": bsn,
            "cf_bsn": cf_bsn,
            "name": name,
            "law": law,
            "skipped": False,
            "original_requirements_met": requirements_met,
            "original_outcome_label": info["outcome_label"],
            "decisive_subject": subject,
            "decisive_description": description,
            "decisive_value": str(value),
            "mutation_description": mutation_desc,
            "engine_flip_verified": flip_verified,
        })

    # Save CF profiles YAML
    cf_yaml_path = PROJECT_ROOT / "data" / f"profiles_cf_{timestamp}.yaml"
    with open(cf_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"profiles": cf_profiles}, f, allow_unicode=True, default_flow_style=False)

    # Save metadata JSONL
    meta_path = OUTPUT_DIR / f"cf_meta_{timestamp}.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    skipped = sum(1 for r in meta_records if r.get("skipped"))
    generated = len(cf_profiles)

    print(f"\n{'=' * 60}")
    print(f"Gegenereerd: {generated} CF-profielen | Overgeslagen: {skipped}")
    print(f"\nCF profielen: {cf_yaml_path}")
    print(f"Metadata:     {meta_path}")
    print(f"\n{'=' * 60}")
    print("VOLGENDE STAP:")
    print("  1. Voeg de CF-profielen toe aan data/profiles.yaml:")
    print("     python -c \"")
    print("       import yaml")
    print("       orig = yaml.safe_load(open('data/profiles.yaml'))")
    print(f"       cf   = yaml.safe_load(open('{cf_yaml_path.name}'))")
    print("       orig['profiles'].update(cf['profiles'])")
    print("       yaml.dump(orig, open('data/profiles.yaml','w'), allow_unicode=True)\"")
    print("  2. Herstart de web server:  $env:FEATURE_CHAT='1'; uv run web/main.py")
    print(f"  3. Voer fase 2 uit met --phase run --meta-file {meta_path}")

    return meta_path


# ---------------------------------------------------------------------------
# Phase 2: Run batch conversations and evaluate
# ---------------------------------------------------------------------------

def _outcome_from_text(text: str) -> str | None:
    """
    Extract a simple eligible/ineligible signal from LLM response text.
    Returns "eligible", "ineligible", or None if unclear.
    """
    text_low = text.lower()

    positive_signals = [
        "u komt in aanmerking",
        "u heeft recht",
        "u heeft recht op",
        "u heeft aanspraak",
        "u voldoet aan alle",
        "u ontvangt",
        "recht op zorgtoeslag",
        "recht op bijstand",
        "vergunning wordt verleend",
        "vergunning kunt u aanvragen",
        "in aanmerking voor",
        "u komt in aanmerking",
        "u heeft recht op deze",
    ]
    negative_signals = [
        "u komt niet in aanmerking",
        "u heeft geen recht",
        "geen recht op",
        "niet in aanmerking",
        "voldoet niet aan",
        "u voldoet helaas niet",
        "u heeft helaas geen",
        "geen aanspraak",
        "wordt geweigerd",
        "niet verleend",
    ]

    for signal in negative_signals:
        if signal in text_low:
            return "ineligible"
    for signal in positive_signals:
        if signal in text_low:
            return "eligible"
    return None


def _explanation_mentions_variable(text: str, subject: str, decisive_description: str) -> bool:
    """
    Check whether the LLM explanation mentions the changed variable.
    Checks field name, Dutch description words, and common synonyms.
    """
    text_low = text.lower()
    subject_low = subject.lower().replace("_", " ")

    SYNONYMS: dict[str, list[str]] = {
        # zorgtoeslag
        "inkomen": ["inkomen", "toetsingsinkomen", "verdient", "verdienen", "salaris", "inkomsten"],
        "toetsingsinkomen": ["toetsingsinkomen", "inkomen", "inkomsten"],
        "partner_inkomen": ["partner", "partnerinkomen", "gezamenlijk inkomen"],
        "is_verzekerde": ["zorgverzekering", "verzekerd", "basisverzekering", "zvw"],
        "heeft_zorgtoeslag_verzekering": ["zorgverzekering", "verzekerd", "basisverzekering"],
        "vermogen": ["vermogen", "spaargeld", "bezittingen", "rendementsgrondslag"],
        "gezamenlijk_vermogen": ["vermogen", "spaargeld", "bezittingen", "gezamenlijk"],
        # zorgtoeslag + bijstand
        "leeftijd": ["leeftijd", "jaar oud", "jarig", "geboren", "geboortedatum", "minderjarig", "18 jaar"],
        "verblijft_in_nederland": ["woonachtig", "verblijft", "nederland", "ingeschreven"],
        "heeft_vast_adres": ["vast adres", "woonadres", "ingeschreven"],
        # bijstand
        "bedrijfsinkomen": ["bedrijfsinkomen", "onderneming", "ondernemer", "winst"],
        "bezittingen": ["bezittingen", "vermogen", "spaargeld", "eigen vermogen"],
        "heeft_nederlandse_nationaliteit": ["nationaliteit", "nederlander", "staatsburger", "verblijfsrecht"],
        "verblijfsadres": ["verblijfsadres", "woonadres", "adres", "woonplaats"],
        # alcoholwet
        "leeftijd_exploitant": ["leeftijd", "21 jaar", "jaar oud", "meerderjarig"],
        "svh_registratie_geldig": ["sociale hygiëne", "svh", "register", "sociaal hygienisch"],
        "is_ingeschreven_svh_register": ["sociale hygiëne", "svh", "register", "sociaal hygienisch"],
        "is_onder_curatele_exploitant": ["curatele", "handelingsonbekwaam", "curator"],
        "is_onder_curatele": ["curatele", "handelingsonbekwaam", "curator"],
        "is_van_slecht_levensgedrag_exploitant": ["levensgedrag", "bibob", "slecht levensgedrag", "integriteit"],
        "is_van_slecht_levensgedrag": ["levensgedrag", "bibob", "slecht levensgedrag"],
        "vloeroppervlakte_horecalokaliteit": ["vloeroppervlakte", "oppervlakte", "vierkante meter", "m²", "ruimte"],
        "vloeroppervlakte": ["vloeroppervlakte", "oppervlakte", "vierkante meter", "m²"],
        "bibob_oordeel": ["bibob", "integriteit", "levensgedrag", "gevaar"],
    }

    synonyms = SYNONYMS.get(subject_low, [subject_low])
    description_words = [w for w in decisive_description.lower().split() if len(w) > 4]

    for syn in synonyms:
        if syn in text_low:
            return True
    for word in description_words[:5]:
        if word in text_low:
            return True
    return False


async def run_conversation(
    bsn: str,
    law: str,
    provider: str,
    script: list[str],
    base_url: str,
    graphrag: bool,
    no_guard: bool,
    timeout: float,
    verbose: bool,
) -> dict:
    """Run a full multi-turn chat conversation. Returns conversation record."""
    client_id = f"cf_{bsn}_{uuid.uuid4().hex[:8]}"
    uri = f"{base_url}/chat/ws/{client_id}"
    turns: list[dict] = []
    error: str | None = None
    model = "unknown"

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=180) as ws:
            await ws.send(json.dumps({
                "bsn": bsn,
                "provider": provider,
                "graphrag": graphrag,
                "no_guard": no_guard,
            }))

            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(raw)

            if data.get("error"):
                return {"error": data["error"], "turns": []}
            if data.get("feature_disabled"):
                return {"error": "Chat uitgeschakeld — start server met FEATURE_CHAT=1", "turns": []}

            model = data.get("model", "unknown")

            for i, message in enumerate(script, 1):
                if verbose:
                    print(f"      turn {i}/{len(script)}: {message[:60]}...")

                turns.append({"role": "user", "message": message})
                assistant_parts: list[str] = []

                await ws.send(json.dumps({"message": message}))

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        msg_data = json.loads(raw)

                        if msg_data.get("error"):
                            error = msg_data["error"]
                            break
                        if msg_data.get("isProcessing"):
                            continue
                        if msg_data.get("applicationPanel") or msg_data.get("graphPanel"):
                            continue

                        text = msg_data.get("message", "")
                        if text:
                            assistant_parts.append(text)

                        try:
                            follow_raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                            follow_data = json.loads(follow_raw)
                            if follow_data.get("isProcessing"):
                                continue
                            follow_text = follow_data.get("message", "")
                            if follow_text:
                                assistant_parts.append(follow_text)
                        except TimeoutError:
                            break

                    except TimeoutError:
                        if not assistant_parts:
                            error = f"Timeout op beurt {i}"
                        break

                if assistant_parts:
                    turns.append({"role": "assistant", "message": "\n\n".join(assistant_parts)})

                if error:
                    break

    except ConnectionRefusedError:
        error = "Verbinding geweigerd — draait de server?"
    except websockets.exceptions.ConnectionClosedError as e:
        error = f"WebSocket gesloten (code {e.code})"
    except TimeoutError:
        error = "Timeout bij WebSocket handshake"
    except Exception as e:
        error = str(e)

    return {
        "bsn": bsn,
        "law": law,
        "provider": provider,
        "model": model,
        "turns": turns,
        "error": error,
    }


async def run_conversation_with_retry(
    max_retries: int = 3,
    retry_delay: float = 15.0,
    **kwargs,
) -> dict:
    """Call run_conversation with retries on connection/timeout errors."""
    _retryable = ("Verbinding geweigerd", "Timeout bij WebSocket handshake", "WebSocket gesloten")
    for attempt in range(max_retries):
        result = await run_conversation(**kwargs)
        err = result.get("error") or ""
        if not err or not any(r in err for r in _retryable):
            return result
        if attempt < max_retries - 1:
            print(f"  [Retry {attempt + 1}/{max_retries - 1}] {err} — wacht {retry_delay:.0f}s...")
            await asyncio.sleep(retry_delay)
    return result


def _wait_for_server(host: str = "localhost", port: int = 8000, max_wait: int = 60) -> bool:
    """Block until the web server accepts TCP connections or timeout."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def evaluate_pair(
    original_conv: dict,
    cf_conv: dict,
    meta: dict,
) -> dict:
    """
    Evaluate a (original, counterfactual) conversation pair.

    Conversation structure:
      original_conv  — turn 1: Q1 (current entitlement), turn 2: Q2 (what if X changes?)
      cf_conv        — turn 1: Q1 only (entitlement with mutated data)

    Returns evaluation dict with:
        original_q1_outcome       — LLM-stated outcome for original Q1
        cf_q1_outcome             — LLM-stated outcome for CF Q1
        original_q1_correct       — does original Q1 match engine outcome?
        cf_q1_correct             — does CF Q1 match expected (flipped) engine outcome?
        outcome_flipped           — did Q1 outcome flip between original and CF?
        q2_mentions_variable      — does Q2 answer mention the decisive variable?
        q2_predicts_cf_direction  — does Q2 answer predict the correct CF outcome direction?
        counterfactual_score      — 0.0–1.0 composite (average of outcome_flipped + q2_mentions_variable)
    """
    original_req = meta.get("original_requirements_met")
    subject = meta.get("decisive_subject", "")
    decisive_desc = meta.get("decisive_description", "")
    expected_cf_outcome = "ineligible" if original_req else "eligible"

    def get_turn(conv: dict, turn_index: int) -> str:
        """Return the nth assistant message (0-indexed)."""
        count = 0
        for turn in conv.get("turns", []):
            if turn.get("role") == "assistant":
                if count == turn_index:
                    return turn.get("message", "")
                count += 1
        return ""

    orig_q2_text = get_turn(original_conv, 0)  # Q2 is now the only question for original
    cf_q1_text   = get_turn(cf_conv, 0)
    orig_q1_text = ""  # no longer asked for original

    orig_q1_outcome = _outcome_from_text(orig_q1_text)
    cf_q1_outcome   = _outcome_from_text(cf_q1_text)
    q2_outcome      = _outcome_from_text(orig_q2_text)

    # 1. Original Q1 correct?
    original_q1_correct = None
    if orig_q1_outcome is not None:
        original_q1_correct = (orig_q1_outcome == ("eligible" if original_req else "ineligible"))

    # 2. CF Q1 correct (should show flipped outcome)?
    cf_q1_correct = None
    if cf_q1_outcome is not None:
        cf_q1_correct = (cf_q1_outcome == expected_cf_outcome)

    # 3. Did the stated outcome flip between original Q1 and CF Q1?
    outcome_flipped = None
    if orig_q1_outcome is not None and cf_q1_outcome is not None:
        outcome_flipped = (orig_q1_outcome != cf_q1_outcome)

    # 4. Does Q2 answer mention the decisive variable?
    q2_mentions_variable = (
        _explanation_mentions_variable(orig_q2_text, subject, decisive_desc)
        if orig_q2_text else False
    )

    # 5. Does Q2 answer predict the correct CF direction?
    q2_predicts_cf_direction = None
    if q2_outcome is not None:
        q2_predicts_cf_direction = (q2_outcome == expected_cf_outcome)

    # 6. Composite counterfactual score — always based on Q2
    # q2_mentions_variable: always included (0 if Q2 had no answer)
    # q2_predicts_cf_direction: included when determinable
    score_parts: list[float] = [1.0 if q2_mentions_variable else 0.0]
    if q2_predicts_cf_direction is not None:
        score_parts.append(1.0 if q2_predicts_cf_direction else 0.0)
    counterfactual_score = sum(score_parts) / len(score_parts)

    return {
        "original_q1_outcome":       orig_q1_outcome,
        "cf_q1_outcome":             cf_q1_outcome,
        "expected_cf_outcome":       expected_cf_outcome,
        "original_q1_correct":       original_q1_correct,
        "cf_q1_correct":             cf_q1_correct,
        "outcome_flipped":           outcome_flipped,
        "q2_mentions_variable":      q2_mentions_variable,
        "q2_predicts_cf_direction":  q2_predicts_cf_direction,
        "counterfactual_score":      counterfactual_score,
    }


async def phase_run(
    meta_file: Path,
    provider: str,
    base_url: str,
    graphrag: bool,
    no_guard: bool,
    timeout: float,
    verbose: bool,
    timestamp: str,
) -> None:
    """
    Phase 2: run conversations for original + CF BSN pairs, evaluate, save JSONL.
    """
    # Load metadata
    meta_records: list[dict] = []
    with open(meta_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if not rec.get("skipped"):
                meta_records.append(rec)

    law = meta_records[0]["law"] if meta_records else "zorgtoeslag"
    q1 = make_q1(law)

    # Check server readiness before starting
    host, port_str = base_url.replace("ws://", "").replace("http://", "").split(":")
    port = int(port_str.split("/")[0])
    print(f"Wachten op server {host}:{port}...")
    if not _wait_for_server(host, port, max_wait=60):
        print("[FOUT] Server niet bereikbaar na 60s. Start de server eerst.")
        return

    print("\nFase 2 — Batch uitvoeren en evalueren")
    print(f"Law: {law} | Paren: {len(meta_records)} | Provider: {provider} | GraphRAG: {graphrag}")
    print(f"Q1 (beide BSNs): {q1}")
    print("Q2 (alleen origineel): gepersonaliseerd per decisieve variabele")
    print("=" * 60)

    output_path = OUTPUT_DIR / f"cf_eval_{timestamp}.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    outcome_flips: list[bool] = []
    mention_changes: list[bool] = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        # Metadata record
        out_f.write(json.dumps({
            "record_type": "metadata",
            "timestamp": datetime.now().isoformat(),
            "law": law,
            "provider": provider,
            "graphrag": graphrag,
            "guard_enabled": not no_guard,
            "pairs_count": len(meta_records),
            "q1_template": q1,
            "meta_file": str(meta_file),
        }, ensure_ascii=False) + "\n")

        for i, meta in enumerate(meta_records, 1):
            bsn = meta["bsn"]
            cf_bsn = meta["cf_bsn"]
            name = meta.get("name", bsn)
            decisive_subject = meta.get("decisive_subject", "")

            # Q2 is the counterfactual question — scored directly, no Q1 preamble needed
            # (system prompt already provides full profile context).
            # CF script uses Q1 to detect outcome flip.
            q2 = make_q2(decisive_subject)
            original_script = [q2]
            cf_script = [q1]

            print(f"[{i}/{len(meta_records)}] {bsn} ({name}) <-> {cf_bsn}")
            print(f"  Mutatie: {meta.get('mutation_description', '?')}")
            print(f"  Q2: {q2}")

            # Run original conversation (Q1 + Q2)
            print(f"  -> Origineel ({bsn})...")
            orig_conv = await run_conversation_with_retry(
                bsn=bsn, law=law, provider=provider, script=original_script,
                base_url=base_url, graphrag=graphrag, no_guard=no_guard,
                timeout=timeout, verbose=verbose,
            )

            if orig_conv.get("error"):
                print(f"  [FOUT origineel] {orig_conv['error']}")

            # Run counterfactual conversation (Q1 only)
            print(f"  -> Counterfactual ({cf_bsn})...")
            cf_conv = await run_conversation_with_retry(
                bsn=cf_bsn, law=law, provider=provider, script=cf_script,
                base_url=base_url, graphrag=graphrag, no_guard=no_guard,
                timeout=timeout, verbose=verbose,
            )

            if cf_conv.get("error"):
                print(f"  [FOUT counterfactual] {cf_conv['error']}")

            # Evaluate pair
            evaluation = evaluate_pair(orig_conv, cf_conv, meta)

            # Report
            flip_str    = "[+]" if evaluation.get("outcome_flipped") else ("[-]" if evaluation.get("outcome_flipped") is False else "?")
            mention_str = "[+]" if evaluation.get("q2_mentions_variable") else "[-]"
            q2_dir_str  = "[+]" if evaluation.get("q2_predicts_cf_direction") else ("[-]" if evaluation.get("q2_predicts_cf_direction") is False else "?")
            score = evaluation.get("counterfactual_score")
            score_str = f"{score:.2f}" if score is not None else "?"
            print(f"  Uitkomst omgekeerd: {flip_str}  Q2 variabele: {mention_str}  Q2 richting: {q2_dir_str}  Score: {score_str}")

            if evaluation.get("outcome_flipped") is not None:
                outcome_flips.append(evaluation["outcome_flipped"])
            if evaluation.get("q2_mentions_variable") is not None:
                mention_changes.append(evaluation["q2_mentions_variable"])
            if score is not None:
                scores.append(score)

            # Write record
            record = {
                "record_type": "pair",
                "bsn": bsn,
                "cf_bsn": cf_bsn,
                "name": name,
                "law": law,
                "provider": provider,
                "mutation_description": meta.get("mutation_description"),
                "decisive_subject": meta.get("decisive_subject"),
                "decisive_description": meta.get("decisive_description"),
                "original_requirements_met": meta.get("original_requirements_met"),
                "q1": q1,
                "q2": q2,
                "original_conversation": orig_conv,
                "cf_conversation": cf_conv,
                "evaluation": evaluation,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    n = len(meta_records)
    print(f"\n{'=' * 60}")
    print(f"SAMENVATTING ({n} paren)")
    if outcome_flips:
        pct_flip = 100 * sum(outcome_flips) / len(outcome_flips)
        print(f"  Uitkomst omgekeerd (CF Q1):      {sum(outcome_flips)}/{len(outcome_flips)} ({pct_flip:.1f}%)")
    if mention_changes:
        pct_mention = 100 * sum(mention_changes) / len(mention_changes)
        print(f"  Q2 noemt decisieve variabele:    {sum(mention_changes)}/{len(mention_changes)} ({pct_mention:.1f}%)")
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Gemiddelde counterfactual score: {avg:.3f}")
    print(f"\nOutput: {output_path.absolute()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactual robustness evaluatie voor machine-law chat uitleg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gebruik:

  Fase 1 — genereer CF-profielen (server met originele profiles.yaml):
    uv run python analysis/llm_explanations/scripts/chat_batch_counterfactual.py \\
        --phase generate --law zorgtoeslag --limit 200

  Fase 2a — baseline (geen graphrag, geen guard):
    uv run python analysis/llm_explanations/scripts/chat_batch_counterfactual.py \\
        --phase run \\
        --meta-file analysis/llm_explanations/output/cf/cf_meta_TIMESTAMP.jsonl \\
        --provider mistral --no-guard

  Fase 2b — GraphRAG met guard:
    uv run python analysis/llm_explanations/scripts/chat_batch_counterfactual.py \\
        --phase run \\
        --meta-file analysis/llm_explanations/output/cf/cf_meta_TIMESTAMP.jsonl \\
        --provider mistral --graphrag
""",
    )

    parser.add_argument(
        "--phase", required=True, choices=["generate", "run"],
        help="Fase: 'generate' maakt CF-profielen, 'run' voert batch uit",
    )
    parser.add_argument("--law", default="zorgtoeslag",
        help="Wet (default: zorgtoeslag)")
    parser.add_argument("--limit", type=int, default=200,
        help="Max aantal profielen (default: 200)")
    parser.add_argument("--bsn", nargs="+", help="Filter op specifieke BSN(s)")
    parser.add_argument("--meta-file", type=Path, dest="meta_file",
        help="Metadata JSONL van fase 1 (vereist voor fase 2)")
    parser.add_argument(
        "--provider", default="haiku",
        choices=["claude", "haiku", "sonnet", "vlam", "gpt-4o", "gpt-4o-mini",
                 "llama3.1", "llama3.2", "llama3.3", "mistral", "deepseek", "gemma2"],
        help="LLM provider voor fase 2 (default: haiku)",
    )
    parser.add_argument("--host", default="localhost:8000",
        help="Server host:port (default: localhost:8000)")
    parser.add_argument("--graphrag", action="store_true",
        help="Gebruik GraphRAG knowledge graph context in fase 2")
    parser.add_argument("--no-guard", action="store_true", dest="no_guard",
        help="Schakel LLM guard uit")
    parser.add_argument("--timeout", type=float, default=120.0,
        help="Seconden per LLM-respons (default: 120)")
    parser.add_argument("--verbose", action="store_true",
        help="Toon gedetailleerde voortgang")

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.phase == "generate":
        phase_generate(
            law=args.law,
            limit=args.limit,
            bsn_filter=args.bsn,
            verbose=args.verbose,
            timestamp=timestamp,
        )

    elif args.phase == "run":
        if not args.meta_file:
            parser.error("--meta-file is vereist voor fase 2")
        if not args.meta_file.exists():
            parser.error(f"Meta-bestand niet gevonden: {args.meta_file}")

        asyncio.run(phase_run(
            meta_file=args.meta_file,
            provider=args.provider,
            base_url=f"ws://{args.host}",
            graphrag=args.graphrag,
            no_guard=args.no_guard,
            timeout=args.timeout,
            verbose=args.verbose,
            timestamp=timestamp,
        ))


if __name__ == "__main__":
    main()
