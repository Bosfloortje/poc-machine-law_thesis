#!/usr/bin/env python3
"""Valideer generated eligible profiles door ze door de engine te sturen.

Laadt onze custom profiles in de engine en test of bijstand + alcoholwet
requirements_met=True geven.

Gebruik (vanuit poc-machine-law root):
    uv run python validate_eligible_profiles.py

Accepteert bijstand + alcoholwet YAML bestanden uit poc-machine-thesis/data/profielen/.
"""

import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent
THESIS_DATA = PROJECT_ROOT.parent / "poc-machine-thesis" / "data" / "profielen"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

logging.disable(logging.CRITICAL)


# ── Hulpfuncties ──────────────────────────────────────────────────────────────

def find_latest(pattern: str) -> Path:
    files = sorted(THESIS_DATA.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Geen bestand gevonden: {THESIS_DATA}/{pattern}")
    return files[-1]


def load_sample(yaml_path: Path, n: int = 5) -> list[tuple[str, dict]]:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = list(data["profiles"].items())
    step = max(1, len(items) // n)
    return items[::step][:n]


# ── Engine met custom profiles ────────────────────────────────────────────────

def make_engine_for(profiles_yaml: Path):
    """Injecteer profiles in de bestaande Services singleton en geef engine + profiles terug.

    De Services klasse (eventsourcing) staat maar één instantie toe per process.
    We gebruiken de al-gecreëerde singleton uit factory.py en voegen onze profielen toe.
    """
    # Importeer de bestaande singleton (factory.py maakt hem aan bij import)
    from web.engines import factory as _factory
    from web.engines.py_engine import PythonMachineService

    svcs = _factory.services  # bestaande singleton

    with open(profiles_yaml, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Voeg profiles' data toe aan bestaande dataframes
    profiles = raw.get("profiles", {})
    for bsn, profile_data in profiles.items():
        for svc_name, tables in profile_data.get("sources", {}).items():
            if svc_name not in svcs.services:
                continue
            rule_svc = svcs.services[svc_name]
            if not isinstance(tables, dict):
                continue
            for table_name, data in tables.items():
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    continue
                if table_name in rule_svc.source_dataframes:
                    rule_svc.source_dataframes[table_name] = pd.concat(
                        [rule_svc.source_dataframes[table_name], df], ignore_index=True
                    )
                else:
                    rule_svc.source_dataframes[table_name] = df

    return PythonMachineService(svcs), profiles


def eval_profile(engine, bsn: str, profile: dict, law: str, today: str) -> dict:
    sources = profile.get("sources", {})

    # Detecteer GEMEENTE service
    service = None
    gemeente_keys = [k for k in sources if k.startswith("GEMEENTE_")]
    if gemeente_keys:
        service = gemeente_keys[0]

    if law == "zorgtoeslag":
        service = "TOESLAGEN"
        engine_law = "zorgtoeslagwet"
    else:
        engine_law = law

    if service is None:
        return {"error": "Geen GEMEENTE_ key gevonden in profiel"}

    # Bouw params
    params: dict = {"BSN": bsn}
    kvk_data = sources.get("KVK", {})
    if isinstance(kvk_data, dict):
        for rows in kvk_data.values():
            if isinstance(rows, list):
                for row in rows:
                    kvk = row.get("kvk_nummer")
                    if kvk:
                        params["KVK_NUMMER"] = str(kvk)
                        break

    try:
        result = engine.evaluate(
            service=service,
            law=engine_law,
            parameters=params,
            reference_date=today,
            approved=True,
        )
        if result is None:
            return {"error": "Engine returned None"}
        return {
            "requirements_met": result.requirements_met,
            "missing_required": result.missing_required,
            "output": result.output or {},
            "service": service,
            "law": engine_law,
        }
    except Exception as e:
        return {"error": str(e), "service": service, "law": engine_law}


def print_result(bsn: str, profile: dict, result: dict):
    name = profile.get("name", bsn)
    ok = result.get("requirements_met")
    missing = result.get("missing_required")
    err = result.get("error")
    svc = result.get("service", "?")
    law = result.get("law", "?")
    output = result.get("output", {})

    if err:
        symbol = "ERR"
    elif missing:
        symbol = "???"
    elif ok:
        symbol = " OK"
    else:
        symbol = " NO"

    miss_str = " [MISSING DATA]" if missing else ""
    print(f"  [{symbol}] {name} ({bsn}){miss_str}")
    if err:
        print(f"       ERROR: {err}")
    else:
        print(f"       service={svc}, law={law}")
        if output:
            for k, v in list(output.items())[:3]:
                print(f"       {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = datetime.today().strftime("%Y-%m-%d")
    print("=" * 65)
    print("PROFILE VALIDATIE — poc-machine-thesis eligible profiles")
    print("=" * 65)

    # ── BIJSTAND ──────────────────────────────────────────────────
    bijstand_file = find_latest("profiles_200_bijstand_*.yaml")
    print(f"\n[BIJSTAND] {bijstand_file.name}")
    print(f"  Initialiseer engine met {bijstand_file.name}...")
    engine_b, all_b = make_engine_for(bijstand_file)
    print(f"  Engine klaar. {len(all_b)} profielen geladen.")
    print("-" * 65)

    sample_b = load_sample(bijstand_file, n=5)
    b_ok = b_err = b_missing = b_no = 0
    for bsn, profile in sample_b:
        result = eval_profile(engine_b, bsn, profile, "participatiewet/bijstand", today)
        print_result(bsn, profile, result)
        if result.get("error"):        b_err += 1
        elif result.get("missing_required"): b_missing += 1
        elif result.get("requirements_met"): b_ok += 1
        else:                          b_no += 1

    print(f"\n  Bijstand (5 profielen): OK={b_ok}  NO={b_no}  missing={b_missing}  errors={b_err}")

    # ── ALCOHOLWET ────────────────────────────────────────────────
    alcoholwet_file = find_latest("profiles_200_alcoholwet_*.yaml")
    print(f"\n[ALCOHOLWET] {alcoholwet_file.name}")
    print(f"  Initialiseer engine met {alcoholwet_file.name}...")
    engine_a, all_a = make_engine_for(alcoholwet_file)
    print(f"  Engine klaar. {len(all_a)} profielen geladen.")
    print("-" * 65)

    sample_a = load_sample(alcoholwet_file, n=5)
    a_ok = a_err = a_missing = a_no = 0
    for bsn, profile in sample_a:
        result = eval_profile(engine_a, bsn, profile, "alcoholwet/vergunning", today)
        print_result(bsn, profile, result)
        if result.get("error"):        a_err += 1
        elif result.get("missing_required"): a_missing += 1
        elif result.get("requirements_met"): a_ok += 1
        else:                          a_no += 1

    print(f"\n  Alcoholwet (5 profielen): OK={a_ok}  NO={a_no}  missing={a_missing}  errors={a_err}")

    # ── Samenvatting ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    total_ok = b_ok + a_ok
    total_no = b_no + a_no
    total_err = b_err + a_err
    total_missing = b_missing + a_missing
    print(f"TOTAAL (10 profielen): OK={total_ok}  NO={total_no}  missing={total_missing}  errors={total_err}")

    if total_err > 0:
        print("\nWAARSCHUWING: Engine fouten — controleer ontbrekende velden.")
    elif total_missing > 0:
        print("\nWAARSCHUWING: Verplichte data ontbreekt in sommige profielen.")
    elif total_no > 0:
        print(f"\nWAARSCHUWING: {total_no}/10 profielen voldoen NIET aan de voorwaarden.")
        print("  -> Kijk naar de output hierboven voor welke checks falen.")
    else:
        print("\nSUCCES: Alle profielen bevatten voldoende data en voldoen aan de voorwaarden!")


if __name__ == "__main__":
    main()
