#!/usr/bin/env python3
"""
dtcc_sdr_fetcher_v2.py  —  RateEdge DTCC SDR Live Poller
=========================================================
Polls https://pddata.dtcc.com/ppd/api/search/webdisplay directly —
no S3, no manual downloads. Runs every N minutes during market hours
and upserts new IRO trades into Supabase [dtcc_sdr].

Usage:
    python dtcc_sdr_fetcher_v2.py                  # poll once (last 2 hours)
    python dtcc_sdr_fetcher_v2.py --loop 15        # poll every 15 minutes
    python dtcc_sdr_fetcher_v2.py --backfill 5     # load last 5 days
    python dtcc_sdr_fetcher_v2.py --init-only      # create table and exit

Render cron (every 30 min, 9pm-11pm UTC = 7am-9am AEST):
    Add as a Render Cron Job service pointing at this script.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import zipfile
import csv
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dtcc_sdr")

# ── DTCC API ──────────────────────────────────────────────────────────────────

DTCC_API = "https://pddata.dtcc.com/ppd/api/search/webdisplay"

HEADERS = {
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type":    "application/json",
    "Origin":          "https://pddata.dtcc.com",
    "Referer":         "https://pddata.dtcc.com/ppd/cftcdashboard",
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
}

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_db_url():
    return os.environ.get("RATEEDGE_DB_URL", "")

def get_db_connection():
    try:
        import psycopg2
    except ImportError:
        log.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    url = get_db_url()
    if not url:
        log.error("RATEEDGE_DB_URL not set")
        sys.exit(1)
    r = urlparse(url)
    qs = parse_qs(r.query)
    try:
        return psycopg2.connect(
            host=r.hostname, port=r.port or 5432,
            dbname=r.path.lstrip("/"), user=r.username,
            password=r.password,
            sslmode=qs.get("sslmode", ["require"])[0],
            connect_timeout=15,
            options="-c statement_timeout=30000",
        )
    except Exception as e:
        log.error(f"DB connection failed: {e}")
        sys.exit(1)

# ── Table ─────────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dtcc_sdr (
    dissemination_id        TEXT        PRIMARY KEY,
    orig_dissemination_id   TEXT,
    action_type             TEXT,
    event_type              TEXT,
    event_timestamp         TIMESTAMPTZ,
    execution_timestamp     TIMESTAMPTZ,
    asset_class             TEXT DEFAULT 'IR',
    embedded_option_type    TEXT,
    upi_fisn                TEXT,
    upi_underlier_name      TEXT,
    option_type_decoded     TEXT,
    strike_raw              TEXT,
    strike_pct              NUMERIC,
    premium_amount          NUMERIC,
    premium_ccy             TEXT,
    notional_leg1           NUMERIC,
    notional_ccy            TEXT,
    effective_date          DATE,
    expiration_date         DATE,
    maturity_underlier      DATE,
    opt_tenor               TEXT,
    swp_tenor               TEXT,
    first_exercise_date     DATE,
    fixed_rate_leg1         NUMERIC,
    cleared                 TEXT,
    platform_identifier     TEXT,
    trade_date              DATE,
    loaded_at               TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS dtcc_sdr_event_ts   ON dtcc_sdr (event_timestamp DESC);
CREATE INDEX IF NOT EXISTS dtcc_sdr_trade_date  ON dtcc_sdr (trade_date DESC);
CREATE INDEX IF NOT EXISTS dtcc_sdr_opt_type    ON dtcc_sdr (option_type_decoded);
CREATE INDEX IF NOT EXISTS dtcc_sdr_ccy         ON dtcc_sdr (notional_ccy);
"""

def init_table(conn):
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    cur.close()
    log.info("dtcc_sdr table ready.")

# ── Data helpers ──────────────────────────────────────────────────────────────

def decode_type(fisn: str) -> str:
    f = (fisn or "").lower()
    if "/o call" in f or " call " in f: return "CALL"
    if "/o p " in f or " p epn" in f:  return "PUT"
    if "straddle" in f or " opt " in f: return "STR"
    return "OTH"

def months_diff(a, b):
    if not a or not b: return None
    return (b.year - a.year) * 12 + (b.month - a.month)

def fmt_tenor(months) -> str:
    if months is None or months < 0: return ""
    if months < 1:  return "<1M"
    if months < 12: return f"{months}M"
    y, m = divmod(months, 12)
    return f"{y}Y{m}M" if m else f"{y}Y"

def parse_date_str(s):
    if not s: return None
    try: return date.fromisoformat(str(s)[:10])
    except: return None

def parse_ts(s):
    if not s: return None
    try: return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except: return None

def parse_num(s):
    if s is None: return None
    try: return float(str(s).replace(",", ""))
    except: return None

def fmt_strike_pct(val, notation):
    v = parse_num(val)
    if v is None: return None, None
    raw = str(v)
    if str(notation) == "3":   pct = round(v * 100, 6)
    elif str(notation) == "1": pct = round(v, 6)
    else: pct = round(v * 100, 6) if v < 1 else round(v, 6)
    return raw, pct

# ── DTCC API fetch ────────────────────────────────────────────────────────────

def fetch_dtcc(dt_low: datetime, dt_high: datetime, max_notional=5_000_000_000) -> list:
    """
    Call the DTCC webdisplay API for IR trades in a time window.
    Returns list of raw row dicts.
    """
    payload = {
        "jurisdiction":             "CFTC",
        "assetClass":               "RATES",
        "currency":                 "",
        "displayType":              "W",
        "disseminationDateTimeHigh": dt_high.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "disseminationDateTimeLow":  dt_low.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "maxNotionalAmount":        str(max_notional),
        "minNotionalAmount":        "0",
        "name":                     None,
        "productId":                None,
        "searchIndicator":          "post",
        "underlyingAsset":          None,
        "upi":                      None,
        "upiShortName":             None,
    }

    log.info(f"Querying DTCC API: {dt_low.strftime('%Y-%m-%d %H:%M')} → {dt_high.strftime('%Y-%m-%d %H:%M')} UTC")

    try:
        r = requests.post(DTCC_API, json=payload, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Response is typically {"data": [...], "total": N} or just a list
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", data.get("trades", data.get("results", [])))
        return []
    except requests.RequestException as e:
        log.error(f"DTCC API error: {e}")
        return []
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return []

# ── Filter + map rows ─────────────────────────────────────────────────────────

OPTION_CODES = {"OPET", "MDET", "OTHR", "CANC", "BARC", "AMER", "EURO", "BERM"}

def map_row(row: dict, trade_date: date) -> dict | None:
    """Map a DTCC API response row to our DB schema. Returns None if not an option."""

    # Field names from API may be camelCase — handle both
    def g(*keys):
        for k in keys:
            v = row.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return None

    # Detect options
    et   = g("embeddedOptionType", "Embedded Option type", "embedded_option_type") or ""
    fisn = g("upiFisn", "UPI FISN", "upi_fisn") or ""

    if et.upper() not in OPTION_CODES and "/O " not in fisn:
        return None

    eff_date = parse_date_str(g("effectiveDate", "Effective Date"))
    exp_date = parse_date_str(g("expirationDate", "Expiration Date"))
    mat_date = parse_date_str(g("maturityDateOfTheUnderlier", "Maturity date of the underlier"))

    strike_raw, strike_pct = fmt_strike_pct(
        g("strikePrice", "Strike Price"),
        g("strikePriceNotation", "Strike price notation")
    )

    diss_id = g("disseminationIdentifier", "Dissemination Identifier", "id")
    if not diss_id:
        return None

    return {
        "dissemination_id":      diss_id,
        "orig_dissemination_id": g("originalDisseminationIdentifier", "Original Dissemination Identifier"),
        "action_type":           g("actionType", "Action type"),
        "event_type":            g("eventType", "Event type"),
        "event_timestamp":       parse_ts(g("eventTimestamp", "Event timestamp")),
        "execution_timestamp":   parse_ts(g("executionTimestamp", "Execution Timestamp")),
        "asset_class":           g("assetClass", "Asset Class") or "IR",
        "embedded_option_type":  et or None,
        "upi_fisn":              fisn or None,
        "upi_underlier_name":    g("upiUnderlierName", "UPI Underlier Name"),
        "option_type_decoded":   decode_type(fisn),
        "strike_raw":            strike_raw,
        "strike_pct":            strike_pct,
        "premium_amount":        parse_num(g("optionPremiumAmount", "Option Premium Amount")),
        "premium_ccy":           g("optionPremiumCurrency", "Option Premium Currency"),
        "notional_leg1":         parse_num(g("notionalAmountLeg1", "Notional amount-Leg 1")),
        "notional_ccy":          g("notionalCurrencyLeg1", "Notional currency-Leg 1"),
        "effective_date":        eff_date,
        "expiration_date":       exp_date,
        "maturity_underlier":    mat_date,
        "opt_tenor":             fmt_tenor(months_diff(eff_date, exp_date)),
        "swp_tenor":             fmt_tenor(months_diff(exp_date, mat_date)),
        "first_exercise_date":   parse_date_str(g("firstExerciseDate", "First exercise date")),
        "fixed_rate_leg1":       parse_num(g("fixedRateLeg1", "Fixed rate-Leg 1")),
        "cleared":               g("cleared", "Cleared"),
        "platform_identifier":   g("platformIdentifier", "Platform identifier"),
        "trade_date":            trade_date,
    }

# ── Upsert ────────────────────────────────────────────────────────────────────

UPSERT_SQL = """
    INSERT INTO dtcc_sdr (
        dissemination_id, orig_dissemination_id, action_type, event_type,
        event_timestamp, execution_timestamp, asset_class,
        embedded_option_type, upi_fisn, upi_underlier_name, option_type_decoded,
        strike_raw, strike_pct, premium_amount, premium_ccy,
        notional_leg1, notional_ccy, effective_date, expiration_date,
        maturity_underlier, opt_tenor, swp_tenor, first_exercise_date,
        fixed_rate_leg1, cleared, platform_identifier, trade_date, loaded_at
    ) VALUES (
        %(dissemination_id)s, %(orig_dissemination_id)s, %(action_type)s, %(event_type)s,
        %(event_timestamp)s, %(execution_timestamp)s, %(asset_class)s,
        %(embedded_option_type)s, %(upi_fisn)s, %(upi_underlier_name)s, %(option_type_decoded)s,
        %(strike_raw)s, %(strike_pct)s, %(premium_amount)s, %(premium_ccy)s,
        %(notional_leg1)s, %(notional_ccy)s, %(effective_date)s, %(expiration_date)s,
        %(maturity_underlier)s, %(opt_tenor)s, %(swp_tenor)s, %(first_exercise_date)s,
        %(fixed_rate_leg1)s, %(cleared)s, %(platform_identifier)s, %(trade_date)s, NOW()
    )
    ON CONFLICT (dissemination_id) DO UPDATE SET
        action_type          = EXCLUDED.action_type,
        event_timestamp      = EXCLUDED.event_timestamp,
        option_type_decoded  = EXCLUDED.option_type_decoded,
        strike_pct           = EXCLUDED.strike_pct,
        premium_amount       = EXCLUDED.premium_amount,
        notional_leg1        = EXCLUDED.notional_leg1,
        opt_tenor            = EXCLUDED.opt_tenor,
        swp_tenor            = EXCLUDED.swp_tenor,
        cleared              = EXCLUDED.cleared,
        platform_identifier  = EXCLUDED.platform_identifier,
        loaded_at            = NOW()
"""

def upsert_rows(conn, rows: list) -> tuple:
    if not rows: return 0, 0
    cur = conn.cursor()
    ins = upd = 0
    for r in rows:
        cur.execute(UPSERT_SQL, r)
        if cur.rowcount == 1: ins += 1
        else: upd += 1
    conn.commit()
    cur.close()
    return ins, upd

# ── Poll logic ────────────────────────────────────────────────────────────────

def poll_once(conn, hours_back: float = 2.0):
    """Fetch last N hours of trades and upsert options."""
    now = datetime.now(timezone.utc)
    dt_low  = now - timedelta(hours=hours_back)
    dt_high = now
    trade_date = now.date()

    raw_rows = fetch_dtcc(dt_low, dt_high)
    if not raw_rows:
        log.info("No rows returned from API.")
        return 0, 0

    log.info(f"API returned {len(raw_rows)} rows — filtering options...")
    mapped = [map_row(r, trade_date) for r in raw_rows]
    option_rows = [r for r in mapped if r is not None]
    log.info(f"  {len(option_rows)} option rows")

    if not option_rows:
        # Log a sample to help debug field names
        if raw_rows:
            log.info(f"Sample row keys: {list(raw_rows[0].keys())[:15]}")
        return 0, 0

    return upsert_rows(conn, option_rows)

def backfill_day(conn, d: date):
    """Fetch a full day (midnight to midnight UTC)."""
    log.info(f"─── Backfilling {d} ───")
    dt_low  = datetime(d.year, d.month, d.day, 0,  0,  0, tzinfo=timezone.utc)
    dt_high = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    raw_rows = fetch_dtcc(dt_low, dt_high, max_notional=50_000_000_000)
    if not raw_rows:
        log.info("  No rows returned.")
        return
    mapped = [map_row(r, d) for r in raw_rows]
    option_rows = [r for r in mapped if r is not None]
    log.info(f"  {len(raw_rows)} total → {len(option_rows)} options")
    if option_rows:
        ins, upd = upsert_rows(conn, option_rows)
        log.info(f"  Done: {ins} inserted, {upd} updated.")

def prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5: d -= timedelta(days=1)
    return d

def last_n_weekdays(n: int) -> list:
    days, d = [], date.today()
    while len(days) < n:
        d = prev_weekday(d)
        days.append(d)
    return days

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DTCC SDR live poller for RateEdge")
    parser.add_argument("--loop",      type=int,   help="Poll every N minutes continuously")
    parser.add_argument("--hours",     type=float, default=2.0, help="Hours back per poll (default 2)")
    parser.add_argument("--backfill",  type=int,   help="Backfill last N trading days")
    parser.add_argument("--date",      type=str,   help="Backfill specific date YYYY-MM-DD")
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    conn = get_db_connection()
    log.info("DB connected.")
    init_table(conn)

    if args.init_only:
        log.info("Done. Exiting.")
        conn.close()
        return

    if args.backfill:
        for d in reversed(last_n_weekdays(args.backfill)):
            backfill_day(conn, d)
            time.sleep(2)  # be polite to DTCC

    elif args.date:
        backfill_day(conn, date.fromisoformat(args.date))

    elif args.loop:
        log.info(f"Starting poll loop every {args.loop} minutes. Ctrl+C to stop.")
        while True:
            try:
                ins, upd = poll_once(conn, hours_back=args.hours)
                log.info(f"Poll complete: {ins} new, {upd} updated.")
            except Exception as e:
                log.error(f"Poll error: {e}")
            log.info(f"Sleeping {args.loop} minutes...")
            time.sleep(args.loop * 60)

    else:
        ins, upd = poll_once(conn, hours_back=args.hours)
        log.info(f"Done: {ins} new, {upd} updated.")

    conn.close()

if __name__ == "__main__":
    main()
