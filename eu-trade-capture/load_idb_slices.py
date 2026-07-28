#!/usr/bin/env python3
"""
load_idb_slices.py  —  parse downloaded IDB slice files (TP ICAP, BGC, Tradition)
into the store. Broker-agnostic: routes each row to a broker by its MIC, parses
with the shared RTS-2 schema, classifies swaptions / IR options, enriches
(currency + option style from description, expiry/effective dates), tags deferral
state, dedupes, and writes to CSV or Supabase.

Point it at one folder; mix zips/CSVs from any broker — it sorts them out.

    python load_idb_slices.py --folder "C:\\Users\\willp\\Downloads\\idb_slices"
    python load_idb_slices.py --folder ".\\slices" --sink supabase
    python load_idb_slices.py --folder ".\\slices" --all-asset-classes
    python load_idb_slices.py --folder ".\\slices" --reprocess        # ignore manifest

Re-runs are safe (manifest tracks processed files).
Needs eu_transparency_puller.py alongside it. Supabase sink: pip install supabase
and env SUPABASE_URL / SUPABASE_KEY.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
from pathlib import Path

import eu_transparency_puller as core

LOG = logging.getLogger("idb_slices")
MANIFEST = core.OUT_DIR / "idb_processed.json"


def _processed() -> set[str]:
    if MANIFEST.exists():
        try:
            return set(json.loads(MANIFEST.read_text()))
        except Exception:
            return set()
    return set()


def _mark(names: set[str]) -> None:
    MANIFEST.write_text(json.dumps(sorted(_processed() | names), indent=0))


_BGC_GROUP = {"BGCI", "BGCO", "GFIC", "GFSO", "GFSM", "AURB", "AURO"}


# Venue MICs that appear in slice FILENAMES (IOTF-TRD-..., INTRADAY_TRADES_BGCO_...).
# Used to stamp venue_mic when the CSV itself omits the MIC column.
_FNAME_MIC_RE = re.compile(
    r"(?<![A-Z0-9])(IOTF|BGCO|BGCI|GFSO|GFSM|GFIC|AURO|AURB|"
    r"TPEU|TPEL|TPEO|TPEM|TRDX|TRXE|ICAP|ISWA)(?![A-Z0-9])", re.I)


def _mic_from_filename(fname: str) -> str:
    """Derive the venue MIC from the slice filename when the CSV omits it.
        IOTF-TRD-20260612-00123.zip        -> IOTF
        INTRADAY_TRADES_BGCO_2026_06_12_1  -> BGCO
    Returns '' if no known venue token is found."""
    base = fname.split("::")[0].upper()          # drop any "zip::member" suffix
    m = _FNAME_MIC_RE.search(base)
    return m.group(1).upper() if m else ""


def mic_to_broker(mic: str) -> str:
    """Route a venue MIC to its IDB. Extend as new MICs appear in the data."""
    m = (mic or "").upper()
    if m in ("GFIC", "GFSO", "GFSM"):
        return "gfi"                      # GFI Securities OTF/MTF (BGC group)
    if m in ("BGCI", "BGCO"):
        return "bgc"                      # BGC Brokers LP OTF
    if m in ("AURB", "AURO"):
        return "aurel"                    # Aurel BGC OTF (France, BGC group)
    if m.startswith("TP"):
        return "tp"                       # Tullett Prebon (TPEU/TPEL/TPEO...)
    if m.startswith(("IC", "ISW", "IG", "IO")):
        return "icap"                     # ICAP / iSwap family (IOTF = swaptions)
    if m.startswith(("TRD", "TRX")):
        return "tradition"                # Trad-X / Tradition
    return mic or "unknown"


def iter_csv_bytes(folder: Path):
    for p in sorted(folder.glob("*.csv")):
        yield p.name, p.read_bytes()
    for z in sorted(folder.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".csv"):
                        yield f"{z.name}::{member}", zf.read(member)
        except zipfile.BadZipFile:
            LOG.warning("bad zip skipped: %s", z.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Load IDB slice files (TP/ICAP/BGC/Tradition)")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--sink", default="csv", choices=["csv", "supabase", "none"])
    ap.add_argument("--all-asset-classes", action="store_true")
    ap.add_argument("--reprocess", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"folder not found: {folder}")

    done = set() if args.reprocess else _processed()
    rates_only = not args.all_asset_classes
    all_prints: list[core.Print] = []
    seen: set[str] = set()

    for fname, raw in iter_csv_bytes(folder):
        if fname in done:
            continue
        prints = core.normalize_csv(raw, "idb")
        _fmic = _mic_from_filename(fname)        # MIC from slice origin (IOTF/BGC...)
        for p in prints:
            if not p.venue_mic and _fmic:
                p.venue_mic = _fmic              # CSV omitted the MIC — take it from the filename
            p.source = mic_to_broker(p.venue_mic)
            p.finalize()                         # row_hash now includes the resolved MIC
        kept = [p for p in prints if core.is_rate_option(p)] if rates_only else prints
        LOG.info("%-46s parsed=%d kept=%d", fname[-46:], len(prints), len(kept))
        all_prints.extend(kept)
        seen.add(fname)

    LOG.info("total kept: %d from %d new files", len(all_prints), len(seen))
    if all_prints:
        import pandas as pd
        df = pd.DataFrame([p.__dict__ for p in all_prints])
        for col in ("source", "asset_class", "publication_mode", "notional_ccy"):
            LOG.info("by %s:\n%s", col, df[col].value_counts().to_string())

    if args.sink == "csv":
        LOG.info("appended %d new rows to %s", core.append_csv(all_prints), core.MASTER_CSV)
    elif args.sink == "supabase":
        LOG.info("upserted %d rows to Supabase", core.upsert_supabase(all_prints))

    if seen and not args.reprocess:
        _mark(seen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
