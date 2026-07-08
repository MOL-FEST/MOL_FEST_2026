"""
03_fetch_prepare_chembl_target_dataset.py

Purpose
-------
Download bioactivity data from ChEMBL for one target, clean it, convert it into
one molecule-level binary classification dataset, and save a transparent cleaning
report.

Default example
---------------
TARGET_CHEMBL_ID = "CHEMBL206"  # human estrogen receptor alpha / ER-alpha

How to run
----------
1) Install dependencies:
   pip install pandas numpy requests rdkit

2) Run with the default target:
   python 03_fetch_prepare_chembl_target_dataset.py

3) Or pass another target from the command line:
   python 03_fetch_prepare_chembl_target_dataset.py --target CHEMBL233

Main outputs
------------
- <TARGET>_raw_chembl_activities.csv
- <TARGET>_cleaned_activity_level.csv
- <TARGET>_prepared_molecule_classification.csv
- <TARGET>_cleaning_report.csv

Final ML-ready columns include:
- QSAR-Ready SMILES
- molecule_chembl_id
- p_activity_median
- binary_label_active1_inactive0

Cleaning logic
--------------
The script prints and saves every major filtering step:
1) removes rows without molecule ID or SMILES
2) keeps selected standard activity types, e.g. IC50/Ki/Kd/EC50
3) keeps high-confidence target mappings
4) removes censored values such as >, <, >=, <= if EXACT_RELATION_ONLY=True
5) keeps rows with pChEMBL or positive nM standard_value
6) calculates p_activity = pChEMBL if available, otherwise 9 - log10(nM)
7) removes implausible p_activity values outside a chosen range
8) optionally standardizes SMILES with RDKit: largest fragment, uncharge, canonicalize
9) aggregates repeated measurements per molecule/SMILES by median activity
10) converts median p_activity to a binary active/inactive label

Important note
--------------
This script prepares target activity/inactivity data. It does NOT create
agonist-vs-antagonist labels unless those labels are explicitly present in the
source data. For agonist/antagonist modelling, use curated functional assay labels
or add an assay-description-based curation layer manually.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests


# ============================================================
# USER SETTINGS
# ============================================================

TARGET_CHEMBL_ID = "CHEMBL206"  # default: human estrogen receptor alpha / ER-alpha

# Common activity endpoints for ligand-based target classification.
# You can make this stricter, e.g. ["IC50"] only, if you want one endpoint type.
STANDARD_TYPES = ["IC50", "Ki", "Kd", "EC50"]

# ChEMBL confidence_score measures how confidently an assay maps to the target.
# 9 is the strictest for direct single-protein assignment; >=8 is a practical default.
MIN_CONFIDENCE_SCORE = 8

# If True, keep only exact measurements with relation "=".
# This removes censored values such as ">", "<", ">=", "<=".
EXACT_RELATION_ONLY = True

# p_activity is pChEMBL-like activity: higher means more potent.
# p_activity = 6 corresponds approximately to 1 uM.
ACTIVE_THRESHOLD_PACTIVITY = 6.0

# Two possible binary-labelling strategies:
# False: label p >= 6 active, p < 6 inactive. More data, less strict.
# True:  label p >= 6 active, p <= 5 inactive, drop 5 < p < 6 grey-zone compounds.
USE_GREY_ZONE = False
INACTIVE_THRESHOLD_PACTIVITY = 5.0

# Basic plausibility range; avoids impossible/garbled values.
MIN_PACTIVITY_ALLOWED = 3.0
MAX_PACTIVITY_ALLOWED = 12.0

# Limit records for testing. Use None for all available records.
MAX_RECORDS: int | None = None

# ChEMBL API settings.
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data/"
REQUEST_TIMEOUT_SECONDS = 60
PAGE_LIMIT = 1000
SLEEP_BETWEEN_REQUESTS = 0.05

# Output directory. Default means the folder where this script is run.
OUTPUT_DIR = "."


# ============================================================
# SMALL UTILITIES
# ============================================================


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def safe_join_unique(values: Iterable[Any], max_items: int = 12) -> str:
    items = sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
    if len(items) > max_items:
        return ";".join(items[:max_items]) + f";...(+{len(items) - max_items} more)"
    return ";".join(items)


def clean_filename_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


class CleaningLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, step: str, before: int, after: int, note: str = "") -> None:
        self.rows.append(
            {
                "step": step,
                "before_rows": before,
                "after_rows": after,
                "removed_rows": before - after,
                "remaining_percent_of_previous": round(100 * after / before, 2) if before else np.nan,
                "note": note,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def print(self) -> None:
        if not self.rows:
            print("No cleaning steps were logged.")
            return
        print("\nCleaning report")
        print("=" * 80)
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(self.to_frame().to_string(index=False))


# ============================================================
# ChEMBL API FUNCTIONS
# ============================================================


def chembl_get_json(url_or_endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET JSON from ChEMBL. Accepts either a full URL or an endpoint name."""
    if url_or_endpoint.startswith("http"):
        url = url_or_endpoint
    else:
        url = urljoin(CHEMBL_API_BASE, url_or_endpoint.lstrip("/"))

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_target_metadata(target_chembl_id: str) -> dict[str, Any]:
    try:
        return chembl_get_json(f"target/{target_chembl_id}.json")
    except Exception as exc:  # noqa: BLE001 - workshop-friendly message
        print(f"Warning: target metadata could not be fetched: {exc}")
        return {}


def fetch_chembl_activities(
    target_chembl_id: str,
    standard_types: list[str],
    max_records: int | None = None,
) -> pd.DataFrame:
    """Download paginated ChEMBL activity records for one target."""
    params: dict[str, Any] = {
        "target_chembl_id": target_chembl_id,
        "limit": PAGE_LIMIT,
        "offset": 0,
    }
    if standard_types:
        params["standard_type__in"] = ",".join(standard_types)

    all_records: list[dict[str, Any]] = []
    next_url: str | None = "activity.json"
    page = 0

    print("\nDownloading ChEMBL activities")
    print("=" * 80)
    print(f"Target: {target_chembl_id}")
    print(f"Activity types: {standard_types if standard_types else 'all'}")

    while next_url:
        page += 1
        data = chembl_get_json(next_url, params=params if page == 1 else None)
        records = data.get("activities", [])
        all_records.extend(records)

        page_meta = data.get("page_meta", {}) or {}
        total_count = page_meta.get("total_count", "unknown")
        next_url = page_meta.get("next")

        print(f"Page {page:>3}: downloaded {len(records):>5} records | total so far {len(all_records):>6} / {total_count}")

        if max_records is not None and len(all_records) >= max_records:
            all_records = all_records[:max_records]
            print(f"Stopped early because MAX_RECORDS={max_records}")
            break

        if next_url and next_url.startswith("/"):
            next_url = "https://www.ebi.ac.uk" + next_url

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    df = pd.DataFrame(all_records)
    print(f"\nDownloaded rows: {len(df)}")
    return df


def fetch_smiles_for_missing_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    ChEMBL activity records usually include canonical_smiles.
    This fallback fills missing SMILES from the molecule endpoint.
    """
    if "canonical_smiles" not in df.columns or "molecule_chembl_id" not in df.columns:
        return df

    missing_mask = df["canonical_smiles"].isna() | (df["canonical_smiles"].astype(str).str.strip() == "")
    missing_ids = sorted(df.loc[missing_mask, "molecule_chembl_id"].dropna().astype(str).unique())
    if not missing_ids:
        return df

    print(f"\nFetching missing SMILES for {len(missing_ids)} molecule IDs...")
    smiles_lookup: dict[str, str | None] = {}
    for i, mol_id in enumerate(missing_ids, start=1):
        try:
            data = chembl_get_json(f"molecule/{mol_id}.json")
            structures = data.get("molecule_structures") or {}
            smiles_lookup[mol_id] = structures.get("canonical_smiles")
        except Exception:  # noqa: BLE001
            smiles_lookup[mol_id] = None
        if i % 100 == 0:
            print(f"  fetched {i}/{len(missing_ids)}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    fill_values = df.loc[missing_mask, "molecule_chembl_id"].astype(str).map(smiles_lookup)
    df.loc[missing_mask, "canonical_smiles"] = fill_values
    return df


# ============================================================
# CHEMISTRY / CLEANING FUNCTIONS
# ============================================================


def standardize_smiles_series(smiles: pd.Series) -> tuple[pd.Series, str]:
    """
    RDKit-based standardization if RDKit is available.
    Falls back to raw ChEMBL canonical_smiles if RDKit is not installed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception as exc:  # noqa: BLE001
        note = f"RDKit unavailable; using ChEMBL canonical_smiles without extra standardization. Reason: {exc}"
        return smiles.astype(str), note

    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    def one(smi: Any) -> str | None:
        if pd.isna(smi):
            return None
        smi = str(smi).strip()
        if not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            mol = chooser.choose(mol)
            mol = uncharger.uncharge(mol)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:  # noqa: BLE001
            return None

    standardized = smiles.map(one)
    note = "RDKit standardization applied: parse SMILES -> largest fragment -> uncharge -> canonical isomeric SMILES."
    return standardized, note


def prepare_activity_level_dataframe(raw_df: pd.DataFrame, logger: CleaningLogger) -> pd.DataFrame:
    df = raw_df.copy()

    before = len(df)
    df = fetch_smiles_for_missing_molecules(df)
    after = len(df)
    logger.add("Optional SMILES completion from molecule endpoint", before, after, "No rows removed here.")

    required_soft_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_units",
        "standard_value",
        "pchembl_value",
        "confidence_score",
        "assay_chembl_id",
        "assay_type",
        "activity_id",
    ]
    for col in required_soft_cols:
        if col not in df.columns:
            df[col] = np.nan

    before = len(df)
    df = df[df["molecule_chembl_id"].notna()]
    df = df[df["canonical_smiles"].notna()]
    df = df[df["canonical_smiles"].astype(str).str.strip() != ""]
    logger.add(
        "Keep rows with molecule_chembl_id and canonical_smiles",
        before,
        len(df),
        "Molecules without structures cannot be used for SMILES-based ML.",
    )

    if STANDARD_TYPES:
        before = len(df)
        df = df[df["standard_type"].isin(STANDARD_TYPES)]
        logger.add(
            "Keep selected standard_type values",
            before,
            len(df),
            f"Allowed: {STANDARD_TYPES}",
        )

    before = len(df)
    df["confidence_score_num"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df = df[df["confidence_score_num"] >= MIN_CONFIDENCE_SCORE]
    logger.add(
        "Keep high-confidence target mappings",
        before,
        len(df),
        f"confidence_score >= {MIN_CONFIDENCE_SCORE}",
    )

    if EXACT_RELATION_ONLY:
        before = len(df)
        relation = df["standard_relation"].astype(str).str.strip()
        df = df[relation.eq("=")]
        logger.add(
            "Keep exact activity relations only",
            before,
            len(df),
            "Removed censored measurements such as >, <, >=, <=.",
        )

    before = len(df)
    df["standard_value_num"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df["pchembl_value_num"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    units = df["standard_units"].astype(str).str.lower().str.strip()
    has_pchembl = df["pchembl_value_num"].notna()
    has_positive_nm = units.eq("nm") & df["standard_value_num"].gt(0)
    df = df[has_pchembl | has_positive_nm]
    logger.add(
        "Keep rows with pChEMBL or positive nM standard_value",
        before,
        len(df),
        "If pChEMBL is missing, p_activity is calculated as 9 - log10(standard_value in nM).",
    )

    # p_activity calculation.
    df["p_activity"] = df["pchembl_value_num"]
    missing_pchembl = df["p_activity"].isna()
    df.loc[missing_pchembl, "p_activity"] = 9.0 - np.log10(df.loc[missing_pchembl, "standard_value_num"])

    before = len(df)
    df = df[df["p_activity"].between(MIN_PACTIVITY_ALLOWED, MAX_PACTIVITY_ALLOWED, inclusive="both")]
    logger.add(
        "Remove implausible p_activity values",
        before,
        len(df),
        f"Kept {MIN_PACTIVITY_ALLOWED} <= p_activity <= {MAX_PACTIVITY_ALLOWED}.",
    )

    before = len(df)
    df["QSAR-Ready SMILES"], rdkit_note = standardize_smiles_series(df["canonical_smiles"])
    df = df[df["QSAR-Ready SMILES"].notna()]
    df = df[df["QSAR-Ready SMILES"].astype(str).str.strip() != ""]
    logger.add(
        "Standardize and validate SMILES",
        before,
        len(df),
        rdkit_note,
    )

    # Useful lightweight flags.
    df["activity_source"] = np.where(df["pchembl_value_num"].notna(), "pchembl_value", "computed_from_standard_value_nM")

    return df.reset_index(drop=True)


def aggregate_to_molecule_level(clean_df: pd.DataFrame, logger: CleaningLogger) -> pd.DataFrame:
    before = len(clean_df)
    if clean_df.empty:
        logger.add("Aggregate repeated activity records per molecule", before, 0, "No rows available.")
        return pd.DataFrame()

    grouped = (
        clean_df.groupby("QSAR-Ready SMILES", dropna=False)
        .agg(
            molecule_chembl_id=("molecule_chembl_id", safe_join_unique),
            p_activity_median=("p_activity", "median"),
            p_activity_mean=("p_activity", "mean"),
            p_activity_std=("p_activity", "std"),
            p_activity_min=("p_activity", "min"),
            p_activity_max=("p_activity", "max"),
            n_measurements=("p_activity", "size"),
            n_assays=("assay_chembl_id", pd.Series.nunique),
            standard_types=("standard_type", safe_join_unique),
            assay_types=("assay_type", safe_join_unique),
            activity_sources=("activity_source", safe_join_unique),
        )
        .reset_index()
    )

    logger.add(
        "Aggregate repeated activity records per unique QSAR-Ready SMILES",
        before,
        len(grouped),
        "Median p_activity is used as the molecule-level label value.",
    )

    before = len(grouped)
    if USE_GREY_ZONE:
        conditions = [
            grouped["p_activity_median"] >= ACTIVE_THRESHOLD_PACTIVITY,
            grouped["p_activity_median"] <= INACTIVE_THRESHOLD_PACTIVITY,
        ]
        choices = [1, 0]
        grouped["binary_label_active1_inactive0"] = np.select(conditions, choices, default=np.nan)
        grouped = grouped[grouped["binary_label_active1_inactive0"].notna()].copy()
        grouped["binary_label_active1_inactive0"] = grouped["binary_label_active1_inactive0"].astype(int)
        note = (
            f"Strict grey-zone labelling: active if p >= {ACTIVE_THRESHOLD_PACTIVITY}, "
            f"inactive if p <= {INACTIVE_THRESHOLD_PACTIVITY}; middle values dropped."
        )
    else:
        grouped["binary_label_active1_inactive0"] = (grouped["p_activity_median"] >= ACTIVE_THRESHOLD_PACTIVITY).astype(int)
        note = f"Simple threshold labelling: active if p >= {ACTIVE_THRESHOLD_PACTIVITY}, otherwise inactive."

    logger.add("Create binary active/inactive label", before, len(grouped), note)

    grouped["target_chembl_id"] = TARGET_CHEMBL_ID
    grouped["active_threshold_pactivity"] = ACTIVE_THRESHOLD_PACTIVITY
    grouped["use_grey_zone"] = USE_GREY_ZONE

    # Put the most important columns first.
    first_cols = [
        "target_chembl_id",
        "molecule_chembl_id",
        "QSAR-Ready SMILES",
        "binary_label_active1_inactive0",
        "p_activity_median",
        "n_measurements",
        "n_assays",
    ]
    other_cols = [c for c in grouped.columns if c not in first_cols]
    return grouped[first_cols + other_cols].reset_index(drop=True)


# ============================================================
# REPORTING
# ============================================================


def print_target_summary(meta: dict[str, Any], target_chembl_id: str) -> None:
    print("\nTarget metadata")
    print("=" * 80)
    print(f"target_chembl_id : {target_chembl_id}")
    if not meta:
        print("metadata         : not available")
        return
    print(f"pref_name        : {meta.get('pref_name')}")
    print(f"target_type      : {meta.get('target_type')}")
    print(f"organism         : {meta.get('organism')}")


def print_final_summary(mol_df: pd.DataFrame) -> None:
    print("\nFinal molecule-level dataset")
    print("=" * 80)
    print(f"Rows / molecules : {len(mol_df)}")
    if mol_df.empty:
        print("No data left after cleaning. Try relaxing filters.")
        return

    counts = mol_df["binary_label_active1_inactive0"].value_counts().sort_index()
    inactive = int(counts.get(0, 0))
    active = int(counts.get(1, 0))
    print(f"Inactive class 0 : {inactive}")
    print(f"Active class 1   : {active}")
    if len(mol_df):
        print(f"Active fraction  : {active / len(mol_df):.3f}")

    print("\np_activity_median summary")
    print(mol_df["p_activity_median"].describe().to_string())

    print("\nTop 5 rows")
    cols = ["molecule_chembl_id", "QSAR-Ready SMILES", "binary_label_active1_inactive0", "p_activity_median", "n_measurements"]
    print(mol_df[cols].head(5).to_string(index=False))


def save_outputs(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    log_df: pd.DataFrame,
    output_dir: Path,
    target_chembl_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = clean_filename_token(target_chembl_id)

    raw_path = output_dir / f"{prefix}_raw_chembl_activities.csv"
    clean_path = output_dir / f"{prefix}_cleaned_activity_level.csv"
    mol_path = output_dir / f"{prefix}_prepared_molecule_classification.csv"
    log_path = output_dir / f"{prefix}_cleaning_report.csv"

    raw_df.to_csv(raw_path, index=False)
    clean_df.to_csv(clean_path, index=False)
    mol_df.to_csv(mol_path, index=False)
    log_df.to_csv(log_path, index=False)

    print("\nSaved files")
    print("=" * 80)
    print(f"Raw activities             : {raw_path}")
    print(f"Cleaned activity-level data: {clean_path}")
    print(f"ML-ready molecule dataset  : {mol_path}")
    print(f"Cleaning report            : {log_path}")


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ChEMBL target activities and prepare a binary classification dataset."
    )
    parser.add_argument(
        "--target",
        default=TARGET_CHEMBL_ID,
        help="Target ChEMBL ID, e.g. CHEMBL206 for ER-alpha or CHEMBL233 for mu-opioid receptor.",
    )
    parser.add_argument(
        "--out-dir",
        default=OUTPUT_DIR,
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--standard-types",
        default=",".join(STANDARD_TYPES),
        help="Comma-separated standard_type values, e.g. IC50,Ki,Kd,EC50 or IC50 only.",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=MIN_CONFIDENCE_SCORE,
        help="Minimum ChEMBL confidence_score to keep.",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=ACTIVE_THRESHOLD_PACTIVITY,
        help="p_activity threshold for active class. Default 6.0 = about 1 uM.",
    )
    parser.add_argument(
        "--use-grey-zone",
        action="store_true",
        help="Use strict labels: active >= active_threshold, inactive <= inactive_threshold, drop middle.",
    )
    parser.add_argument(
        "--inactive-threshold",
        type=float,
        default=INACTIVE_THRESHOLD_PACTIVITY,
        help="Inactive p_activity threshold when --use-grey-zone is enabled.",
    )
    parser.add_argument(
        "--keep-censored",
        action="store_true",
        help="Keep censored relations such as > and <. Default removes them.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS,
        help="Optional maximum number of raw activity records to download for testing.",
    )
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> list[str]:
    """Update module-level settings from command-line arguments."""
    global TARGET_CHEMBL_ID
    global STANDARD_TYPES
    global MIN_CONFIDENCE_SCORE
    global ACTIVE_THRESHOLD_PACTIVITY
    global USE_GREY_ZONE
    global INACTIVE_THRESHOLD_PACTIVITY
    global EXACT_RELATION_ONLY
    global MAX_RECORDS

    TARGET_CHEMBL_ID = args.target.strip()
    STANDARD_TYPES = [x.strip() for x in str(args.standard_types).split(",") if x.strip()]
    MIN_CONFIDENCE_SCORE = int(args.min_confidence)
    ACTIVE_THRESHOLD_PACTIVITY = float(args.active_threshold)
    USE_GREY_ZONE = bool(args.use_grey_zone)
    INACTIVE_THRESHOLD_PACTIVITY = float(args.inactive_threshold)
    EXACT_RELATION_ONLY = not bool(args.keep_censored)
    MAX_RECORDS = args.max_records
    return STANDARD_TYPES


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()
    standard_types = apply_cli_overrides(args)
    output_dir = Path(args.out_dir)

    print("Settings")
    print("=" * 80)
    print(f"TARGET_CHEMBL_ID              : {TARGET_CHEMBL_ID}")
    print(f"STANDARD_TYPES                : {standard_types}")
    print(f"MIN_CONFIDENCE_SCORE          : {MIN_CONFIDENCE_SCORE}")
    print(f"EXACT_RELATION_ONLY           : {yes_no(EXACT_RELATION_ONLY)}")
    print(f"ACTIVE_THRESHOLD_PACTIVITY    : {ACTIVE_THRESHOLD_PACTIVITY}")
    print(f"USE_GREY_ZONE                 : {yes_no(USE_GREY_ZONE)}")
    print(f"INACTIVE_THRESHOLD_PACTIVITY  : {INACTIVE_THRESHOLD_PACTIVITY}")
    print(f"MAX_RECORDS                   : {MAX_RECORDS}")

    meta = fetch_target_metadata(TARGET_CHEMBL_ID)
    print_target_summary(meta, TARGET_CHEMBL_ID)

    logger = CleaningLogger()
    raw_df = fetch_chembl_activities(TARGET_CHEMBL_ID, standard_types, MAX_RECORDS)
    if raw_df.empty:
        print("No records downloaded. Check the target ChEMBL ID or filters.")
        sys.exit(1)

    clean_df = prepare_activity_level_dataframe(raw_df, logger)
    mol_df = aggregate_to_molecule_level(clean_df, logger)
    log_df = logger.to_frame()

    logger.print()
    print_final_summary(mol_df)
    save_outputs(raw_df, clean_df, mol_df, log_df, output_dir, TARGET_CHEMBL_ID)

    print("\nNext step")
    print("=" * 80)
    print(
        "Use the ML-ready file as INPUT_FILE in your benchmark scripts and set:\n"
        "TARGET_COLUMN = 'binary_label_active1_inactive0'\n"
        "SMILES_COLUMN = 'QSAR-Ready SMILES'"
    )


if __name__ == "__main__":
    main()
