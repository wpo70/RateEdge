#!/usr/bin/env python3
"""
firds_resolver.py  â€”  resolve ISIN -> CFI via ESMA FIRDS, for CFI-less feeds.

Some feeds (MarketAxess APA, any ToTV EZ/QZ-ISIN feed) publish only an ISIN â€” no
CFI code, no description â€” so swaptions can't be classified directly. This module
looks each ISIN up in ESMA's reference data (via the official `esma_data_py`
package) to recover its CFI, caches the result, and lets the loader reclassify.

Mechanism (verified against esma_data_py API):
  EsmaDataLoader().load_latest_files(cfi=<cat>, eqt=False, isin=[...]) returns the
  reference rows for non-equity instruments in CFI category <cat> matching the
  ISINs. We probe categories options-first (H = swaptions/IR options live here),
  capture the full CFI when present, and map ISIN -> CFI / asset_class.

Install:  pip install git+https://github.com/European-Securities-Markets-Authority/esma_data_py.git

CLI:
  python firds_resolver.py --isins EZ7M6ZH0HTB6,XS2770920937
  python firds_resolver.py --from-csv APAPubDataExport.csv --col "Instrument identification code"
  python firds_resolver.py --show-cache

Cache: eu_transparency_data/firds_cache.json  (ISIN -> {cfi, category, asset_class})

NETWORK: the lookup downloads ESMA reference files at run time (large, cached by
the package). First run is slow; re-runs hit the cache. If esma_data_py isn't
installed or ESMA is unreachable, resolve() returns {} and the caller falls back
to whatever classification it already had.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("firds")

OUT_DIR = Path("./eu_transparency_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "firds_cache.json"

# Non-equity CFI categories, options first so swaptions resolve before we waste
# time on bond/swap files. H = options (incl. swaptions), S = swaps, J = forwards,
# D = debt, C = CIVs, I = spot, F = futures, O = listed options, R = entitlements.
PROBE_ORDER = ["H", "S", "J", "F", "O", "D", "R", "I", "C"]

# Column-name candidates in the returned reference frame.
_ISIN_COLS = ["Id", "ISIN", "Isin", "Instrument identification code", "FinInstrmId"]
_CFI_COLS = ["ClssfctnTp", "CFI", "CFICode", "Clssfctn", "FinInstrmClssfctn", "classification"]


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(c: dict) -> None:
    CACHE.write_text(json.dumps(c, indent=0, sort_keys=True))


def _pick_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.strip().lower() in lower:
            return lower[cand.strip().lower()]
    return None


def _classify_from_cfi(cfi: str, category: str):
    """Map CFI (or bare category) to our asset_class. Reuses core CFI logic when a
    full CFI is available; otherwise returns a category-level verdict."""
    try:
        import eu_transparency_puller as core
        if cfi and len(cfi) >= 2:
            return core.classify_asset_class("", cfi, "")
    except Exception:
        pass
    # category-only fallback (no full CFI in the reference frame)
    if category == "H":
        return "OPTION_UNCONFIRMED"   # an option, but rates(HR) vs FX(HF) unknown
    return "OTHER"


def resolve(isins, loader=None, probe=PROBE_ORDER, use_cache=True) -> dict:
    """Return {isin: {'cfi':..,'category':..,'asset_class':..}} for the ISINs.
    Unresolved ISINs are simply absent from the result."""
    isins = sorted({i.strip().upper() for i in isins if i and str(i).strip()})
    cache = _load_cache() if use_cache else {}
    todo = [i for i in isins if i not in cache]
    if not todo:
        return {i: cache[i] for i in isins if i in cache}

    if loader is None:
        try:
            from esma_data_py import EsmaDataLoader
            loader = EsmaDataLoader()
        except ImportError:
            LOG.error("esma_data_py not installed â€” "
                      "pip install git+https://github.com/European-Securities-Markets-Authority/esma_data_py.git")
            return {i: cache[i] for i in isins if i in cache}

    remaining = set(todo)
    for cat in probe:
        if not remaining:
            break
        try:
            LOG.info("FIRDS probe category %s for %d ISINs", cat, len(remaining))
            df = loader.load_latest_files(cfi=cat, eqt=False, isin=list(remaining))
        except Exception as e:
            LOG.warning("category %s lookup failed: %s", cat, e)
            continue
        if df is None or len(df) == 0:
            continue
        isin_col = _pick_col(df, _ISIN_COLS)
        cfi_col = _pick_col(df, _CFI_COLS)
        if not isin_col:
            LOG.warning("no ISIN column found in category %s frame; cols=%s", cat, list(df.columns)[:12])
            continue
        for _, row in df.iterrows():
            iid = str(row[isin_col]).strip().upper()
            if iid not in remaining:
                continue
            cfi = str(row[cfi_col]).strip().upper() if cfi_col and row.get(cfi_col) is not None else ""
            rec = {"cfi": cfi, "category": cat, "asset_class": _classify_from_cfi(cfi, cat)}
            cache[iid] = rec
            remaining.discard(iid)

    if use_cache:
        _save_cache(cache)
    if remaining:
        LOG.info("%d ISINs unresolved (not found in probed categories)", len(remaining))
    return {i: cache[i] for i in isins if i in cache}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Resolve ISIN -> CFI via ESMA FIRDS")
    ap.add_argument("--isins", help="comma-separated ISINs")
    ap.add_argument("--from-csv", help="CSV file to pull ISINs from")
    ap.add_argument("--col", default="Instrument identification code", help="ISIN column name in --from-csv")
    ap.add_argument("--show-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.show_cache:
        print(CACHE.read_text() if CACHE.exists() else "{} (empty)")
        return 0

    isins = set()
    if args.isins:
        isins |= {x.strip() for x in args.isins.split(",") if x.strip()}
    if args.from_csv:
        import csv
        with open(args.from_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = next((k for k in row if (k or "").strip().lower() == args.col.strip().lower()), None)
                val = (row.get(key) or "").strip() if key else ""
                if val:
                    isins.add(val)
    if not isins:
        sys.exit("provide --isins or --from-csv")

    LOG.info("resolving %d unique ISINs", len(isins))
    res = resolve(isins)
    from collections import Counter
    by_class = Counter(v["asset_class"] for v in res.values())
    print(f"\nresolved {len(res)}/{len(isins)} ISINs")
    print("by asset_class:", dict(by_class))
    swaptions = [i for i, v in res.items() if v["asset_class"] == "SWAPTION"]
    if swaptions:
        print(f"\nSWAPTIONS found ({len(swaptions)}):")
        for i in swaptions[:25]:
            print(f"  {i}  cfi={res[i]['cfi']}")
    else:
        print("\nNo swaptions among resolved ISINs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

