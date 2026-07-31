#!/usr/bin/env python3
"""
eu_transparency_puller.py  â€”  RateEdge EU MiFIR post-trade transparency puller

Pulls free (15-min delayed) RTS 2 post-trade transparency data from EU/UK
trading-venue + APA portals, filters to interest-rate options / swaptions,
classifies the deferral state of every print, dedupes, and stores.

WHY A FRAMEWORK, NOT HARDCODED SCRAPES
--------------------------------------
The IDB portals (TP ICAP, Fenics/BGC, Trad-X) are JavaScript single-page apps;
their underlying file endpoints are not publicly documented. Rather than guess
URLs, each venue is an ADAPTER with one of two fetch modes:

  mode="http"     -> direct GET of a real CSV/file URL (use when you have one)
  mode="browser"  -> headless Chromium loads the portal, applies the asset-class
                     filter, clicks download, captures the file. No XHR reversing.

Adapters with a verified static URL are wired. The JS-portal adapters are stubs
with the entry URL filled in and a TODO for the selector recipe â€” finalize each
by running with --capture <venue> (saves a HAR + screenshot) or paste me the
network request and I'll complete the recipe.

DELIBERATE CAVEAT BAKED IN
--------------------------
Most EUR rate-vol notional is LIS-deferred (often ~4 weeks) and volume-masked.
Every row is tagged publication_mode in {REAL_TIME, DEFERRED, VOL_MASKED,
AGGREGATED, UNKNOWN} from RTS 2 flags so you never mistake a deferred print for
a live one. Do not treat this as a live tape.

USAGE
-----
  pip install requests pandas python-dateutil    # core
  pip install playwright && playwright install chromium   # for browser mode
  pip install supabase                            # only if --sink supabase

  python eu_transparency_puller.py --once
  python eu_transparency_puller.py --loop --interval 1200      # every 20 min
  python eu_transparency_puller.py --once --sources nasdaq_apa,tpicap
  python eu_transparency_puller.py --capture tpicap            # recipe-building aid

Windows Task Scheduler (every 20 min):
  schtasks /Create /SC MINUTE /MO 20 /TN RateEdge_EU_Pull ^
    /TR "C:\\Path\\python.exe C:\\Path\\eu_transparency_puller.py --once" /F
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests
import pandas as pd

LOG = logging.getLogger("eu_pull")

# --------------------------------------------------------------------------- #
#  Output location
# --------------------------------------------------------------------------- #
OUT_DIR = Path(os.environ.get("EU_PULL_OUT", "./eu_transparency_data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)
MASTER_CSV = OUT_DIR / "eu_iro_prints.csv"

UA = "RateEdge-EU-Transparency-Puller/1.0 (post-trade transparency consumption)"

# --------------------------------------------------------------------------- #
#  Canonical record (one normalized print)
# --------------------------------------------------------------------------- #
CANON_FIELDS = [
    "source",            # venue key
    "venue_mic",         # MIC of execution venue
    "isin",              # instrument ISIN (venue-generated for OTC-on-venue)
    "instrument_desc",   # free text / FISN if present
    "asset_class",       # our classification
    "exec_utc",          # execution timestamp (UTC ISO)
    "pub_utc",           # publication timestamp (UTC ISO)
    "price",             # price as reported
    "price_ccy",
    "notional",          # notional amount (may be masked)
    "notional_ccy",
    "quantity",
    "publication_mode",  # REAL_TIME / DEFERRED / VOL_MASKED / AGGREGATED / UNKNOWN
    "deferral_flags",    # raw flag string for audit
    "trade_id",          # venue TVTIC / trade id if present
    "row_hash",          # dedupe key
    "ingested_utc",
]


@dataclass
class Print:
    source: str
    venue_mic: str = ""
    isin: str = ""
    instrument_desc: str = ""
    asset_class: str = ""
    exec_utc: str = ""
    pub_utc: str = ""
    price: Optional[float] = None
    price_ccy: str = ""
    notional: Optional[float] = None
    notional_ccy: str = ""
    quantity: Optional[float] = None
    publication_mode: str = "UNKNOWN"
    deferral_flags: str = ""
    trade_id: str = ""
    row_hash: str = ""
    ingested_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def finalize(self) -> "Print":
        if not self.row_hash:
            basis = "|".join([
                self.source, self.venue_mic, self.isin, self.exec_utc,
                str(self.price), str(self.notional), self.trade_id,
            ])
            self.row_hash = hashlib.sha1(basis.encode("utf-8")).hexdigest()
        return self


# --------------------------------------------------------------------------- #
#  Swaption / interest-rate-option classification
# --------------------------------------------------------------------------- #
# Reliable classification = map ISIN -> CFI/sub-asset-class via ESMA FIRDS.
# Heuristic fallback below catches the obvious cases from description text and
# CFI when present. CFI (ISO 10962): options start with 'O'. Swaption is an
# option whose underlying is a swap; RTS 2 sub-asset class is "Swaptions" (a
# distinct sub-class) or "Interest rate options".
IRO_TEXT_HINTS = (
    "swaption", "swpt", "ir option", "interest rate option", "irÐ¾",
    "payer", "receiver", "straddle", "cap", "floor", "collar",
)


def classify_asset_class(desc: str, cfi: str = "", sub_asset: str = "") -> str:
    s = (sub_asset or "").lower()
    if "swaption" in s:
        return "SWAPTION"
    if "interest rate option" in s or "ir option" in s:
        return "IR_OPTION"
    cfi = (cfi or "").upper()
    if cfi.startswith("HR"):          # ISO 10962 H=option R=rates -> swaption (BGC HRCAVC/HRCDVC, IOTF HRA/C/D/H)
        return "SWAPTION"
    d = (desc or "").lower()
    is_option = cfi.startswith("O") or any(h in d for h in IRO_TEXT_HINTS)
    if is_option and ("swap" in d or "swaption" in d or "swpt" in d):
        return "SWAPTION"
    if is_option and any(h in d for h in ("cap", "floor", "collar")):
        return "IR_OPTION"
    if is_option and any(h in d for h in ("interest", "rate", "ir ")):
        return "IR_OPTION"
    return "OTHER"


def is_rate_option(p: Print) -> bool:
    return p.asset_class in ("SWAPTION", "IR_OPTION")


# --------------------------------------------------------------------------- #
#  Deferral classification from RTS 2 flags
# --------------------------------------------------------------------------- #
# RTS 1/2 post-trade flag codes commonly seen in venue/APA feeds:
#   'LRGS'/'LIST' large-in-scale | 'SIZE'/'ILQD' SSTI/illiquid deferral
#   'DEFR' deferred | 'VOLO'/'VOLW' volume omission/masked | 'FWAF'/'4FOUR' 4-week
#   'ACTX' agency cross | 'AGGR' aggregated | 'CANC' cancellation | 'AMND' amend
DEFERRAL_TOKENS = {
    "DEFR", "LRGS", "LIST", "SIZE", "ILQD", "FWAF", "4WKS", "SDIV", "TPAC",
}
VOLMASK_TOKENS = {"VOLO", "VOLW", "FULV", "FULA"}
AGG_TOKENS = {"AGGR", "ACTX"}


def classify_publication(flags: str, exec_utc: str = "", pub_utc: str = "") -> str:
    raw = (flags or "").upper().replace(";", ",").replace("|", ",")
    toks = {t.strip() for t in raw.split(",") if t.strip()}
    if toks & VOLMASK_TOKENS:
        return "VOL_MASKED"
    if toks & AGG_TOKENS:
        return "AGGREGATED"
    if toks & DEFERRAL_TOKENS:
        return "DEFERRED"
    # time-gap fallback: if published >> 15 min after exec, treat as deferred
    try:
        if exec_utc and pub_utc:
            e = pd.to_datetime(exec_utc, utc=True)
            pb = pd.to_datetime(pub_utc, utc=True)
            if (pb - e).total_seconds() > 30 * 60:
                return "DEFERRED"
            return "REAL_TIME"
    except Exception:
        pass
    return "UNKNOWN"


# --------------------------------------------------------------------------- #
#  Fetch strategies
# --------------------------------------------------------------------------- #
def fetch_http_file(url: str, params: dict | None = None, timeout: int = 60) -> bytes:
    LOG.info("HTTP GET %s", url)
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return r.content


def fetch_via_browser(recipe: dict, capture: bool = False) -> bytes:
    """
    Headless-Chromium fetch for JS portals. `recipe` keys:
      url            : entry page
      steps          : list of {action, selector, value} (click/select/wait)
      download_trigger_selector : element whose click starts the file download
    Returns the downloaded file bytes. Requires:  pip install playwright
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("browser mode needs: pip install playwright && playwright install chromium")

    data = {"bytes": b""}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not capture)
        ctx = browser.new_context(user_agent=UA, accept_downloads=True,
                                   record_har_path=str(RAW_DIR / f"capture_{int(time.time())}.har") if capture else None)
        page = ctx.new_page()
        page.goto(recipe["url"], wait_until="networkidle", timeout=60000)
        for step in recipe.get("steps", []):
            act = step["action"]
            if act == "wait":
                page.wait_for_timeout(int(step.get("value", 1000)))
            elif act == "click":
                page.click(step["selector"], timeout=20000)
            elif act == "select":
                page.select_option(step["selector"], step["value"], timeout=20000)
            elif act == "wait_for":
                page.wait_for_selector(step["selector"], timeout=30000)
        trig = recipe.get("download_trigger_selector")
        if trig:
            with page.expect_download(timeout=60000) as dl_info:
                page.click(trig)
            dl = dl_info.value
            tmp = RAW_DIR / dl.suggested_filename
            dl.save_as(str(tmp))
            data["bytes"] = tmp.read_bytes()
        if capture:
            page.screenshot(path=str(RAW_DIR / "capture_screenshot.png"), full_page=True)
            LOG.warning("Capture saved to %s â€” inspect HAR/screenshot to build the recipe.", RAW_DIR)
        ctx.close()
        browser.close()
    return data["bytes"]


# --------------------------------------------------------------------------- #
#  Generic RTS 2 CSV normalizer
# --------------------------------------------------------------------------- #
# Column-name candidates seen across venue feeds. Each adapter may override.
COLMAP_DEFAULT = {
    "isin": ["ISIN", "Isin", "InstrumentId", "Instrument Identification Code"],
    "instrument_desc": ["Instrument Description", "FISN", "InstrumentName", "Instrument", "Description",
                         "Instrument Full Name"],
    "venue_mic": ["MIC", "Venue", "Venue of Execution", "TradingVenue",
                  "Operating MIC", "Segment MIC", "MIC Code", "Trading Venue MIC",
                  "Execution Venue", "Venue MIC", "Market Identifier Code"],
    "exec_utc": ["Trade Time Stamp", "TradingDateTime", "ExecutionTimestamp", "Trade Date Time",
                 "Trading Date and Time", "TransactionTime"],
    "pub_utc": ["Published Time Stamp", "PublicationDateTime", "Publication Date Time", "PublicationTimestamp", "Publication Date & Time"],
    "price": ["Price"],
    "price_ccy": ["PriceCurrency", "Price Currency", "PriceNotation"],
    "notional": ["Notional", "NotionalAmount", "Notional Amount", "Quantity in Measurement Unit"],
    "notional_ccy": ["NotionalCurrency", "Notional Currency"],
    "quantity": ["Quantity", "Volume"],
    "deferral_flags": ["Flags", "Flag", "PublicationFlags", "Trade Flags", "FlagType"],
    "trade_id": ["TVTIC", "TradeId", "Transaction Identification Code", "TradeID"],
    "cfi": ["CFI", "CFICode", "CFI Code"],
    "sub_asset": ["SubAssetClass", "Sub Asset Class", "AssetClass"],
    "effective_date": ["Effective Date", "EffectiveDate", "Effective"],
    "maturity_date": ["Maturity Date", "MaturityDate", "Maturity"],
}


def _pick(row: dict, names: list[str]) -> str:
    for n in names:
        if n in row and str(row[n]).strip() not in ("", "nan", "None"):
            return str(row[n]).strip()
    norm = {k.strip().lower(): v for k, v in row.items() if isinstance(k, str)}
    for n in names:
        v = norm.get(n.strip().lower())
        if v is not None and str(v).strip() not in ("", "nan", "None"):
            return str(v).strip()
    return ""


def normalize_csv(content: bytes, source: str, colmap: dict | None = None,
                  delimiter: str | None = None) -> list[Print]:
    cm = {**COLMAP_DEFAULT, **(colmap or {})}
    text = content.decode("utf-8-sig", errors="replace")
    # sniff delimiter if not given
    if delimiter is None:
        sample = text[:4096]
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    out: list[Print] = []
    for row in reader:
        desc = _pick(row, cm["instrument_desc"])
        cfi = _pick(row, cm["cfi"])
        sub = _pick(row, cm["sub_asset"])
        flags = _pick(row, cm["deferral_flags"])
        exec_t = _pick(row, cm["exec_utc"])
        pub_t = _pick(row, cm["pub_utc"])
        _ac = classify_asset_class(desc, cfi, sub)
        if _ac == "SWAPTION" and not desc:
            # BGC/Fenics carry no text description; build one from the
            # structured dates so expiry + swap tenor are recoverable.
            _eff = re.sub(r"[^0-9]", "", _pick(row, cm["effective_date"]))[:8]
            _mat = re.sub(r"[^0-9]", "", _pick(row, cm["maturity_date"]))[:8]
            if _eff:
                desc = f"{source.upper()} O Opt Epn Fxd Flt EUR {_eff}"
                if _mat:
                    desc += f" {_mat}"
        p = Print(
            source=source,
            venue_mic=_pick(row, cm["venue_mic"]),
            isin=_pick(row, cm["isin"]),
            instrument_desc=desc,
            asset_class=_ac,
            exec_utc=exec_t,
            pub_utc=pub_t,
            price=_to_float(_pick(row, cm["price"])),
            price_ccy=_pick(row, cm["price_ccy"]),
            notional=_to_float(_pick(row, cm["notional"])),
            notional_ccy=_pick(row, cm["notional_ccy"]),
            quantity=_to_float(_pick(row, cm["quantity"])),
            deferral_flags=flags,
            publication_mode=classify_publication(flags, exec_t, pub_t),
            trade_id=_pick(row, cm["trade_id"]),
        )
        out.append(p.finalize())
    return out


def _to_float(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def normalize_json(content: bytes, source: str, colmap: dict | None = None,
                   records_path: str | None = None) -> list[Print]:
    """Normalize a JSON feed (common for portal XHR endpoints). `records_path`
    is an optional dotted path to the row array, e.g. 'data.rows'."""
    cm = {**COLMAP_DEFAULT, **(colmap or {})}
    obj = json.loads(content.decode("utf-8-sig", errors="replace"))
    rows = obj
    if records_path:
        for part in records_path.split("."):
            rows = rows.get(part, []) if isinstance(rows, dict) else []
    else:
        # auto-find the first list-of-dicts in the payload
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    rows = v
                    break
            else:
                rows = obj.get("data") or obj.get("rows") or obj.get("results") or []
    out: list[Print] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        desc = _pick(row, cm["instrument_desc"])
        cfi = _pick(row, cm["cfi"])
        sub = _pick(row, cm["sub_asset"])
        flags = _pick(row, cm["deferral_flags"])
        exec_t = _pick(row, cm["exec_utc"])
        pub_t = _pick(row, cm["pub_utc"])
        out.append(Print(
            source=source,
            venue_mic=_pick(row, cm["venue_mic"]),
            isin=_pick(row, cm["isin"]),
            instrument_desc=desc,
            asset_class=classify_asset_class(desc, cfi, sub),
            exec_utc=exec_t, pub_utc=pub_t,
            price=_to_float(_pick(row, cm["price"])),
            price_ccy=_pick(row, cm["price_ccy"]),
            notional=_to_float(_pick(row, cm["notional"])),
            notional_ccy=_pick(row, cm["notional_ccy"]),
            quantity=_to_float(_pick(row, cm["quantity"])),
            deferral_flags=flags,
            publication_mode=classify_publication(flags, exec_t, pub_t),
            trade_id=_pick(row, cm["trade_id"]),
        ).finalize())
    return out


def normalize_auto(content: bytes, source: str, colmap: dict | None = None) -> list[Print]:
    """Sniff CSV vs JSON and dispatch."""
    head = content[:64].lstrip()
    if head[:1] in (b"{", b"["):
        return normalize_json(content, source, colmap)
    return normalize_csv(content, source, colmap)


# --------------------------------------------------------------------------- #
#  endpoints.json (written by idb_capture.py) + per-IDB MIC groups
# --------------------------------------------------------------------------- #
ENDPOINTS_JSON = OUT_DIR / "endpoints.json"

# MIC groups split the shared portals into the 4 IDB feeds.
# CONFIRM the *_CONFIRM lists from the MICs that show up in the first pull
# (the puller prints observed MICs per portal to help you map them).
MIC_GROUPS = {
    "bgc":        {"BGCO", "GFSO"},                 # BGC OTF + GFI Securities OTF (confident)
    "tradition":  {"TRDX", "TRXE"},                 # Trad-X MTF UK/EU; add Tradition OTF MICs (CONFIRM)
    "tp":         set(),                            # Tullett Prebon EU OTF/MTF MICs (CONFIRM from pull)
    "icap":       set(),                            # ICAP/iSwap EU venue MICs (CONFIRM from pull)
}
# Which portal endpoint each IDB feed reads from.
IDB_PORTAL = {"tp": "tpicap", "icap": "tpicap", "bgc": "fenics", "tradition": "tradition"}


def _load_endpoints() -> dict:
    if ENDPOINTS_JSON.exists():
        try:
            return json.loads(ENDPOINTS_JSON.read_text())
        except Exception:
            LOG.error("endpoints.json is not valid JSON")
    return {}


# --------------------------------------------------------------------------- #
#  Source registry  (adapters)
# --------------------------------------------------------------------------- #
# Each adapter returns list[Print]. Use the helpers above.
# STATUS legend:  WIRED = runnable now | STUB = needs endpoint/recipe from you.

@dataclass
class Source:
    key: str
    name: str
    status: str                    # WIRED | STUB
    fetch: Callable[[], list[Print]]
    note: str = ""


def _adapter_nasdaq_apa() -> list[Print]:
    """
    WIRED skeleton. Nasdaq publishes free 15-min delayed RTS 2 CSV per asset
    class at tradereports.nasdaq.com (per-minute files, kept ~48h). The exact
    per-asset CSV URL must be set in EU_PULL_NASDAQ_URL once confirmed from the
    portal's 'Download CSV' link (it is a real CSV endpoint, not JS-gated).
    """
    url = os.environ.get("EU_PULL_NASDAQ_URL", "").strip()
    if not url:
        LOG.warning("[nasdaq_apa] set EU_PULL_NASDAQ_URL to the rates/derivatives Download-CSV link")
        return []
    content = fetch_http_file(url)
    (RAW_DIR / f"nasdaq_apa_{_ts()}.csv").write_bytes(content)
    return normalize_csv(content, "nasdaq_apa")


_PORTAL_CACHE: dict[str, list[Print]] = {}


def _fetch_portal(portal_key: str) -> list[Print]:
    """Fetch+normalize a shared portal once per run (TP+ICAP share 'tpicap').
    Reads the endpoint discovered by idb_capture.py from endpoints.json."""
    if portal_key in _PORTAL_CACHE:
        return _PORTAL_CACHE[portal_key]
    eps = _load_endpoints()
    ep = eps.get(portal_key)
    if not ep or not ep.get("url"):
        LOG.warning("[%s] no endpoint yet \u2014 run: python idb_capture.py capture %s ; then analyze",
                    portal_key, portal_key)
        _PORTAL_CACHE[portal_key] = []
        return []
    method = (ep.get("method") or "GET").upper()
    url = ep["url"]
    LOG.info("[%s] %s %s", portal_key, method, url)
    if method == "POST":
        body = ep.get("post_data") or ""
        r = requests.post(url, data=body, headers={"User-Agent": UA,
                          "Content-Type": ep.get("content_type", "application/json")}, timeout=90)
        r.raise_for_status()
        content = r.content
    else:
        content = fetch_http_file(url)
    (RAW_DIR / f"{portal_key}_{_ts()}.raw").write_bytes(content)
    fmt = ep.get("format", "auto")
    if fmt == "csv":
        prints = normalize_csv(content, portal_key)
    elif fmt == "json":
        prints = normalize_json(content, portal_key, records_path=ep.get("records_path"))
    else:
        prints = normalize_auto(content, portal_key)
    mics = sorted({p.venue_mic for p in prints if p.venue_mic})
    if mics:
        LOG.info("[%s] observed MICs: %s", portal_key, ", ".join(mics))
    _PORTAL_CACHE[portal_key] = prints
    return prints


def _idb_adapter(feed_key: str) -> Callable[[], list[Print]]:
    """One IDB feed: fetch its portal, filter to that IDB's MIC group."""
    portal = IDB_PORTAL[feed_key]

    def _fn() -> list[Print]:
        rows = _fetch_portal(portal)
        mics = MIC_GROUPS.get(feed_key, set())
        if mics:
            rows = [p for p in rows if p.venue_mic in mics]
        elif portal == "tpicap":
            LOG.warning("[%s] MIC_GROUPS['%s'] empty \u2014 map it from the observed MICs "
                        "to split TP vs ICAP (skipping to avoid double count)", feed_key, feed_key)
            return []
        for p in rows:
            p.source = feed_key
            p.finalize()
        return rows
    return _fn


def _adapter_generic_http(key: str, env_var: str) -> Callable[[], list[Print]]:
    """Factory for any venue that exposes a real CSV/file URL via an env var.
    Use for Deutsche BÃ¶rse, Euronext, Cboe, Tradeweb APA, Bloomberg APA, etc.
    Set the URL in the env var and the adapter is live."""
    def _fn() -> list[Print]:
        url = os.environ.get(env_var, "").strip()
        if not url:
            LOG.warning("[%s] set %s to the venue's Download-CSV/file link", key, env_var)
            return []
        content = fetch_http_file(url)
        (RAW_DIR / f"{key}_{_ts()}.csv").write_bytes(content)
        return normalize_csv(content, key)
    return _fn


def build_registry() -> dict[str, Source]:
    reg: dict[str, Source] = {}

    def add(s: Source):
        reg[s.key] = s

    add(Source("nasdaq_apa", "Nasdaq APA / Nordic", "WIRED", _adapter_nasdaq_apa,
               "Set EU_PULL_NASDAQ_URL to rates/derivatives CSV link."))

    # The 4 IDB feeds. tp+icap share the TP ICAP portal (split by MIC_GROUPS).
    add(Source("tp", "Tullett Prebon (TP ICAP portal)", "ENDPOINT", _idb_adapter("tp"),
               "endpoints.json['tpicap']; then map MIC_GROUPS['tp']."))
    add(Source("icap", "ICAP / iSwap (TP ICAP portal)", "ENDPOINT", _idb_adapter("icap"),
               "endpoints.json['tpicap']; then map MIC_GROUPS['icap']."))
    add(Source("bgc", "BGC / GFI (Fenics portal)", "ENDPOINT", _idb_adapter("bgc"),
               "endpoints.json['fenics']; MICs BGCO/GFSO preset."))
    add(Source("tradition", "Tradition OTF / Trad-X", "ENDPOINT", _idb_adapter("tradition"),
               "endpoints.json['tradition']; confirm OTF MICs (TRDX/TRXE preset)."))

    # Generic HTTP venues â€” flip to WIRED by setting the env var to a real URL.
    for key, env in [
        ("deutsche_boerse", "EU_PULL_DBAG_URL"),
        ("euronext",        "EU_PULL_EURONEXT_URL"),
        ("cboe_apa",        "EU_PULL_CBOE_URL"),
        ("tradeweb_apa",    "EU_PULL_TRADEWEB_URL"),
        ("bloomberg_apa",   "EU_PULL_BBG_APA_URL"),
        ("marketaxess_apa", "EU_PULL_MAX_URL"),
        ("lseg_tradecho",   "EU_PULL_TRADECHO_URL"),
    ]:
        add(Source(key, key.replace("_", " ").title(), "WIRED(env)",
                   _adapter_generic_http(key, env),
                   f"Set {env} to the venue Download-CSV/file URL."))
    return reg


# --------------------------------------------------------------------------- #
#  Persistence
# --------------------------------------------------------------------------- #
def load_existing_hashes() -> set[str]:
    if not MASTER_CSV.exists():
        return set()
    try:
        df = pd.read_csv(MASTER_CSV, usecols=["row_hash"])
        return set(df["row_hash"].astype(str))
    except Exception:
        return set()


def append_csv(prints: list[Print]) -> int:
    if not prints:
        return 0
    seen = load_existing_hashes()
    fresh = [p for p in prints if p.row_hash not in seen]
    if not fresh:
        return 0
    write_header = not MASTER_CSV.exists()
    with MASTER_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CANON_FIELDS)
        if write_header:
            w.writeheader()
        for p in fresh:
            w.writerow({k: getattr(p, k) for k in CANON_FIELDS})
    return len(fresh)


def upsert_supabase(prints: list[Print]) -> int:
    """Optional sink. Table DDL (run once in Supabase):

        CREATE TABLE IF NOT EXISTS eu_iro_prints (
            row_hash         text PRIMARY KEY,
            source           text,
            venue_mic        text,
            isin             text,
            instrument_desc  text,
            asset_class      text,
            exec_utc         timestamptz,
            pub_utc          timestamptz,
            price            double precision,
            price_ccy        text,
            notional         double precision,
            notional_ccy     text,
            quantity         double precision,
            publication_mode text,
            deferral_flags   text,
            trade_id         text,
            ingested_utc     timestamptz
        );
    Needs: pip install supabase ; env SUPABASE_URL, SUPABASE_KEY.
    """
    if not prints:
        return 0
    try:
        from supabase import create_client
    except ImportError:
        raise RuntimeError("supabase sink needs: pip install supabase")
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        raise RuntimeError("set SUPABASE_URL and SUPABASE_KEY")
    client = create_client(url, key)
    rows = {}
    for p in prints:
        d = {k: getattr(p, k) for k in CANON_FIELDS}
        for t in ("exec_utc", "pub_utc"):
            d[t] = d[t] or None
        rows[d["row_hash"]] = d
    # upsert on PK row_hash => idempotent re-runs; dedupe in-batch first
    client.table("eu_iro_prints").upsert(list(rows.values()), on_conflict="row_hash").execute()
    return len(rows)


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_once(registry: dict[str, Source], keys: list[str], sink: str,
             rates_only: bool = True) -> None:
    all_prints: list[Print] = []
    for k in keys:
        src = registry.get(k)
        if not src:
            LOG.error("unknown source: %s", k)
            continue
        try:
            prints = src.fetch()
            if rates_only:
                prints = [p for p in prints if is_rate_option(p)]
            LOG.info("[%s] %d rate-option prints", k, len(prints))
            all_prints.extend(prints)
        except Exception as e:
            LOG.exception("[%s] failed: %s", k, e)

    if sink == "csv":
        n = append_csv(all_prints)
        LOG.info("appended %d new rows to %s", n, MASTER_CSV)
    elif sink == "supabase":
        n = upsert_supabase(all_prints)
        LOG.info("upserted %d rows to Supabase", n)

    # quick deferral breakdown so you see live vs deferred at a glance
    if all_prints:
        df = pd.DataFrame([asdict(p) for p in all_prints])
        LOG.info("publication_mode breakdown:\n%s",
                 df["publication_mode"].value_counts().to_string())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RateEdge EU MiFIR transparency puller")
    ap.add_argument("--once", action="store_true", help="single pull then exit")
    ap.add_argument("--loop", action="store_true", help="pull repeatedly")
    ap.add_argument("--interval", type=int, default=1200, help="loop seconds (default 1200=20min)")
    ap.add_argument("--sources", default="all", help="comma list or 'all'")
    ap.add_argument("--sink", default="csv", choices=["csv", "supabase", "none"])
    ap.add_argument("--all-asset-classes", action="store_true",
                    help="keep everything, not just swaptions/IR options")
    ap.add_argument("--capture", default="", help="venue key: open headed browser + save HAR to build a recipe")
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    registry = build_registry()

    if args.list:
        for s in registry.values():
            print(f"{s.key:16} [{s.status:10}] {s.name}  â€” {s.note}")
        return 0

    if args.capture:
        print("Capture + endpoint discovery now lives in idb_capture.py:\n"
              f"    python idb_capture.py capture {args.capture}\n"
              f"    python idb_capture.py analyze eu_transparency_data/raw/<file>.har\n"
              "Then re-run this puller; it reads eu_transparency_data/endpoints.json.")
        return 0

    keys = list(registry.keys()) if args.sources == "all" else \
        [k.strip() for k in args.sources.split(",") if k.strip()]

    rates_only = not args.all_asset_classes
    sink = args.sink

    if args.loop:
        LOG.info("loop mode: every %ds, sources=%s, sink=%s", args.interval, keys, sink)
        while True:
            run_once(registry, keys, sink, rates_only)
            time.sleep(args.interval)
    else:
        run_once(registry, keys, sink, rates_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())

