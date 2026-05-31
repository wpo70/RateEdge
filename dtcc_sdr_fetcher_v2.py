#!/usr/bin/env python3
"""
dtcc_sdr_fetcher_v2.py  —  RateEdge DTCC SDR Live Poller
=========================================================
Polls https://pddata.dtcc.com/ppd/api/search/webdisplay directly.
AUD: every 5 minutes during Sydney/Tokyo hours.
USD: EOD pull at NY close.

Usage:
    python dtcc_sdr_fetcher_v2.py                  # poll once (last 10 min)
    python dtcc_sdr_fetcher_v2.py --loop 5         # poll every 5 minutes
    python dtcc_sdr_fetcher_v2.py --date 2026-04-10
    python dtcc_sdr_fetcher_v2.py --backfill 5
    python dtcc_sdr_fetcher_v2.py --init-only
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs, unquote

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

OPTION_CODES = {"OPET", "MDET", "OTHR", "CANC", "BARC", "AMER", "EURO", "BERM"}

# ── DB ────────────────────────────────────────────────────────────────────────

def get_db_url():
    return os.environ.get("RATEEDGE_DB_URL", "")

def get_db_connection():
    try:
        import psycopg2
    except ImportError:
        log.error("pip install psycopg2-binary")
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

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dtcc_sdr (
    dissemination_id        TEXT        PRIMARY KEY,
    orig_dissemination_id   TEXT,
    action_type             TEXT,
    event_type              TEXT,
    event_timestamp         TIMESTAMPTZ,
    execution_timestamp     TIMESTAMPTZ,
    dissemination_timestamp TIMESTAMPTZ,
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
    loaded_at               TIMESTAMPTZ DEFAULT NOW(),
    trade_type              TEXT DEFAULT 'OPT'  -- OPT = option, CFS = forward-starting IRS
);
CREATE INDEX IF NOT EXISTS dtcc_sdr_event_ts   ON dtcc_sdr (event_timestamp DESC);
CREATE INDEX IF NOT EXISTS dtcc_sdr_diss_ts    ON dtcc_sdr (dissemination_timestamp DESC);
CREATE INDEX IF NOT EXISTS dtcc_sdr_trade_date ON dtcc_sdr (trade_date DESC);
CREATE INDEX IF NOT EXISTS dtcc_sdr_opt_type   ON dtcc_sdr (option_type_decoded);
CREATE INDEX IF NOT EXISTS dtcc_sdr_ccy        ON dtcc_sdr (notional_ccy);
"""

def init_table(conn):
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    cur.close()
    log.info("dtcc_sdr table ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_type(fisn: str, embedded_option_type: str = "") -> str:
    """
    Map DTCC FISN + embeddedOptionType to RateEdge type code.

    DTCC → BBG equivalents:
      /O Call Epn  = C  (European Call)
      /O P Epn     = P  (European Put)
      /O Opt Epn   = D  (Straddle / European Chooser = BBG EC)
      /O Call Brm  = BC (Bermudan Call)
      /O Nstd      = NC (Non-standard)
      OPET+Swap    = EC (European swaption / cancelable)
      MDET+Swap    = MDET (Mandatory early termination)
      CANC+Swap    = NC (Cancellable swap)
      OPET+FltFlt  = XCS (XCCY swaption)
      MDET+FltFlt  = XCS-MDET
      OTHR         = OTH
    """
    f = (fisn or "").lower()
    et = (embedded_option_type or "").upper()

    # ── Vanilla options (FISN starts with /O) ────────────────────────────────
    if "/o call brm" in f or " call brm" in f: return "BCALL"   # Bermudan Call
    if "/o call" in f or " call epn" in f:     return "CALL"    # European Call
    if "/o p " in f or " p epn" in f:          return "PUT"     # European Put
    if "/o opt" in f or " opt epn" in f:       return "STR"     # Straddle (BBG EC)
    if "/o nstd" in f or " nstd " in f:        return "NSTD"    # Non-standard

    # ── XCCY swaptions (Flt Flt FISN) ────────────────────────────────────────
    if "flt flt" in f:
        if et == "MDET": return "XCS-M"   # XCCY with mandatory termination
        return "XCS"                       # XCCY swaption

    # ── Plain swap FISN with embedded option type ─────────────────────────────
    if et == "OPET":  return "EC"     # European swaption (BBG: EC = Cancelable)
    if et == "MDET":  return "MDET"   # Mandatory early termination
    if et == "CANC":  return "CANC"   # Cancellable swap
    if et == "OTHR":  return "OTH"    # Other/exotic

    return "OTH"

def months_diff(a, b):
    """Month difference ignoring day component — avoids T+2 stub inflation."""
    if not a or not b: return None
    # Replace day with 1 to avoid T+2 settlement date pushing into next month
    try:
        from datetime import date as _date
        a1 = _date(a.year, a.month, 1)
        b1 = _date(b.year, b.month, 1)
        m = (b1.year - a1.year) * 12 + (b1.month - a1.month)
        return m
    except Exception:
        return (b.year - a.year) * 12 + (b.month - a.month)

def fmt_tenor(months) -> str:
    if months is None or months < 0: return ""
    if months < 1:  return "<1M"
    if months < 12: return f"{months}M"
    y, m = divmod(months, 12)
    # Business day stubs:
    # m <= 2  → round down (e.g. 10Y1M → 10Y)
    # m >= 10 → round up   (e.g. 1Y11M → 2Y)
    # else    → keep genuine broken tenor (e.g. 1Y6M = 18M)
    if m <= 2:   return f"{y}Y"
    if m >= 10:  return f"{y+1}Y"
    return f"{y}Y{m}M"

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
    n = str(notation) if notation else ""
    if n == "3":   pct = round(v * 100, 6)
    elif n == "1": pct = round(v, 6)
    else:          pct = round(v * 100, 6) if v < 1 else round(v, 6)
    return raw, pct

# ── Map row — exact camelCase field names from API ────────────────────────────

def is_cfs(row: dict) -> bool:
    """Detect forward-starting IRS (CFS) — effective date >30 days after execution."""
    et = (row.get("embeddedOptionType") or "").strip()
    if et:  # has an embedded option — not a plain IRS
        return False
    fisn = unquote((row.get("uniqueProductIdentifierShortName") or "").lower())
    # Must be a plain IRS product
    if not any(x in fisn for x in ["fxd flt", "ois", "flt flt"]):
        return False
    eff  = parse_date_str(row.get("effectiveDate"))
    exec_ts = parse_ts(row.get("executionTimestamp"))
    if not eff or not exec_ts:
        return False
    exec_date = exec_ts.date() if hasattr(exec_ts, 'date') else exec_ts
    days_fwd = (eff - exec_date).days
    return days_fwd > 30  # forward start > 1 month = CFS

def map_row(row: dict, trade_date: date):
    et   = (row.get("embeddedOptionType") or "").strip()
    fisn = unquote((row.get("uniqueProductIdentifierShortName") or "").strip())

    is_option = et.upper() in OPTION_CODES or "/O " in fisn
    is_cfs_trade = not is_option and is_cfs(row)

    if not is_option and not is_cfs_trade:
        return None
    
    trade_type = "CFS" if is_cfs_trade else "OPT"

    diss_id = str(row.get("disseminationIdentifier") or "").strip()
    if not diss_id:
        return None

    eff_date = parse_date_str(row.get("effectiveDate"))
    exp_date = parse_date_str(row.get("expirationDate"))
    mat_date = parse_date_str(row.get("maturityDateOfTheUnderlier"))
    strike_raw, strike_pct = fmt_strike_pct(
        row.get("strikePrice"),
        row.get("strikePriceNotation")
    )

    return {
        "dissemination_id":      diss_id,
        "orig_dissemination_id": str(row.get("originalDisseminationIdentifier") or "") or None,
        "action_type":           row.get("actionType"),
        "event_type":            row.get("eventType"),
        "event_timestamp":       parse_ts(row.get("eventTimestamp")),
        "execution_timestamp":   parse_ts(row.get("executionTimestamp")),
        "dissemination_timestamp": parse_ts(row.get("disseminationTimestamp")),
        "asset_class":           row.get("assetClass") or "IR",
        "embedded_option_type":  et or None,
        "upi_fisn":              fisn or None,
        "upi_underlier_name":    row.get("uniqueProductIdentifierUnderlierName"),
        "option_type_decoded":   "CFS" if is_cfs_trade else decode_type(fisn, et),
        "strike_raw":            strike_raw,
        "strike_pct":            strike_pct,
        "premium_amount":        parse_num(row.get("optionPremiumAmount")),
        "premium_ccy":           row.get("optionPremiumCurrency"),
        "notional_leg1":         parse_num(row.get("notionalAmountLeg1")),
        "notional_ccy":          row.get("notionalCurrencyLeg1"),
        "effective_date":        eff_date,
        "expiration_date":       exp_date,
        "maturity_underlier":    mat_date,
        "opt_tenor":             fmt_tenor(months_diff(eff_date, exp_date)),
        "swp_tenor":             fmt_tenor(months_diff(exp_date, mat_date)),
        "first_exercise_date":   parse_date_str(row.get("firstExerciseDate")),
        "fixed_rate_leg1":       parse_num(row.get("fixedRateLeg1")),
        "cleared":               row.get("cleared"),
        "platform_identifier":   row.get("platformIdentifier"),
        "trade_date":            trade_date,
        "trade_type":            trade_type,
    }

# ── DTCC fetch ────────────────────────────────────────────────────────────────

def fetch_dtcc(dt_low: datetime, dt_high: datetime,
               min_notional: int = 0,
               max_notional: int = 50_000_000_000) -> list:

    payload = {
        "jurisdiction":              "CFTC",
        "assetClass":                "RATES",
        "currency":                  "",
        "displayType":               "W",
        "disseminationDateTimeHigh": dt_high.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "disseminationDateTimeLow":  dt_low.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "maxNotionalAmount":         str(max_notional),
        "minNotionalAmount":         str(min_notional),
        "name":                      None,
        "productId":                 None,
        "searchIndicator":           "post",
        "underlyingAsset":           None,
        "upi":                       None,
        "upiShortName":              None,
    }

    log.info(f"DTCC query: {dt_low.strftime('%Y-%m-%d %H:%M')} → {dt_high.strftime('%Y-%m-%d %H:%M')} UTC")

    try:
        r = requests.post(DTCC_API, json=payload, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        trades = data.get("tradeList") or []
        log.info(f"  API returned {len(trades):,} total rows")
        return trades
    except Exception as e:
        log.error(f"DTCC API error: {e}")
        return []

# ── Upsert ────────────────────────────────────────────────────────────────────

def ensure_heartbeat_table(conn):
    """Create sdr_heartbeat table if it doesn't exist (self-bootstraps)."""
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sdr_heartbeat (
                id      INTEGER PRIMARY KEY DEFAULT 1,
                last_run TIMESTAMPTZ NOT NULL,
                status  TEXT NOT NULL DEFAULT 'ok'
            )
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        log.warning(f"Heartbeat table bootstrap failed: {e}")


def upsert_heartbeat(conn, status: str = 'ok'):
    """Write a heartbeat timestamp on every poll — regardless of trade activity."""
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sdr_heartbeat (id, last_run, status)
            VALUES (1, NOW() AT TIME ZONE 'UTC', %s)
            ON CONFLICT (id) DO UPDATE SET last_run = NOW() AT TIME ZONE 'UTC', status = EXCLUDED.status
        """, (status,))
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        log.warning(f"Heartbeat upsert failed: {e}")


def upsert_rows(conn, rows: list) -> tuple:
    """Batch upsert using execute_values for 10-20x speed improvement."""
    if not rows: return 0, 0
    from psycopg2.extras import execute_values

    COLS = [
        "dissemination_id", "orig_dissemination_id", "action_type", "event_type",
        "event_timestamp", "execution_timestamp", "dissemination_timestamp", "asset_class",
        "embedded_option_type", "upi_fisn", "upi_underlier_name", "option_type_decoded",
        "strike_raw", "strike_pct", "premium_amount", "premium_ccy",
        "notional_leg1", "notional_ccy", "effective_date", "expiration_date",
        "maturity_underlier", "opt_tenor", "swp_tenor", "first_exercise_date",
        "fixed_rate_leg1", "cleared", "platform_identifier", "trade_date",
    ]

    SQL = """
        INSERT INTO dtcc_sdr ({cols}, loaded_at)
        VALUES %s
        ON CONFLICT (dissemination_id) DO UPDATE SET
            action_type             = EXCLUDED.action_type,
            event_timestamp         = EXCLUDED.event_timestamp,
            dissemination_timestamp = EXCLUDED.dissemination_timestamp,
            option_type_decoded     = EXCLUDED.option_type_decoded,
            strike_pct              = EXCLUDED.strike_pct,
            premium_amount          = EXCLUDED.premium_amount,
            notional_leg1           = EXCLUDED.notional_leg1,
            opt_tenor               = EXCLUDED.opt_tenor,
            swp_tenor               = EXCLUDED.swp_tenor,
            cleared                 = EXCLUDED.cleared,
            platform_identifier     = EXCLUDED.platform_identifier,
            loaded_at               = NOW()
    """.format(cols=", ".join(COLS))

    template = "(" + ", ".join(["%s"] * len(COLS)) + ", NOW())"
    values = [tuple(r[c] for c in COLS) for r in rows]

    cur = conn.cursor()
    execute_values(cur, SQL, values, template=template, page_size=500)
    count = cur.rowcount
    conn.commit()
    cur.close()
    # rowcount from execute_values = rows affected (inserts + updates)
    return count, 0

# ── Poll / backfill logic ─────────────────────────────────────────────────────

def poll_once(conn, minutes_back: int = 10):
    """Fetch last N minutes and upsert option rows."""
    now = datetime.now(timezone.utc)
    dt_low  = now - timedelta(minutes=minutes_back)
    dt_high = now
    trade_date = now.date()

    raw = fetch_dtcc(dt_low, dt_high)
    if not raw:
        return 0, 0

    mapped = [map_row(r, trade_date) for r in raw]
    opts = [r for r in mapped if r is not None]
    cfs_count = sum(1 for r in opts if r.get("trade_type") == "CFS")
    opt_count = len(opts) - cfs_count
    log.info(f"  {len(opts)} rows ({opt_count} options, {cfs_count} CFS | AUD: {sum(1 for r in opts if r['notional_ccy']=='AUD')})")

    if not opts:
        return 0, 0
    return upsert_rows(conn, opts)

def backfill_day(conn, d: date):
    log.info(f"─── Backfilling {d} ───")
    dt_low  = datetime(d.year, d.month, d.day,  0,  0,  0, tzinfo=timezone.utc)
    dt_high = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    raw = fetch_dtcc(dt_low, dt_high)
    if not raw:
        log.info("  No rows returned.")
        return
    mapped = [map_row(r, d) for r in raw]
    opts = [r for r in mapped if r is not None]
    aud = [r for r in opts if r.get("notional_ccy") == "AUD"]
    log.info(f"  {len(opts)} options total  |  {len(aud)} AUD")
    if opts:
        ins, upd = upsert_rows(conn, opts)
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
    parser = argparse.ArgumentParser(description="DTCC SDR live poller — RateEdge")
    parser.add_argument("--loop",      type=int,   help="Poll every N minutes continuously")
    parser.add_argument("--minutes",   type=int,   default=10, help="Look-back window per poll (default 10)")
    parser.add_argument("--backfill",  type=int,   help="Backfill last N trading days")
    parser.add_argument("--date",      type=str,   help="Backfill specific date YYYY-MM-DD")
    parser.add_argument("--date-from", type=str,   dest="date_from", help="Range start YYYY-MM-DD")
    parser.add_argument("--date-to",   type=str,   dest="date_to",   help="Range end YYYY-MM-DD")
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    conn = get_db_connection()
    log.info("DB connected.")
    init_table(conn)

    if args.init_only:
        log.info("Done."); conn.close(); return

    if args.date_from and args.date_to:
        d_from = date.fromisoformat(args.date_from)
        d_to   = date.fromisoformat(args.date_to)
        days   = []
        d = d_from
        while d <= d_to:
            if d.weekday() < 5:
                days.append(d)
            d += __import__('datetime').timedelta(days=1)
        log.info(f"Date range: {d_from} → {d_to} ({len(days)} trading days)")
        for i, d in enumerate(days, 1):
            log.info(f"[{i}/{len(days)}] {d}")
            backfill_day(conn, d)
            time.sleep(2)

    elif args.backfill:
        for d in reversed(last_n_weekdays(args.backfill)):
            backfill_day(conn, d)
            time.sleep(3)

    elif args.date:
        backfill_day(conn, date.fromisoformat(args.date))

    elif args.loop:
        # Time-aware polling — active CCY windows (UTC)
        CCY_WINDOWS = {
            "AUD": (21, 8),   # 21:00-08:00 UTC
            "USD": (10, 21),  # 10:00-21:00 UTC
            "JPY": (22, 7),   # 22:00-07:00 UTC
            "EUR": (7,  16),  # 07:00-16:00 UTC
            "GBP": (7,  16),  # 07:00-16:00 UTC
        }

        # Pre-open hours: fetch 1hr of catch-up data before each market opens
        PRE_OPEN_HOURS = {
            "AUD": 20,   # 1hr before 21:00 UTC open
            "USD":  9,   # 1hr before 10:00 UTC open
            "JPY":  21,  # 1hr before 22:00 UTC open
            "EUR":   6,  # 1hr before 07:00 UTC open
            "GBP":   6,
        }
        _pre_open_done = set()  # track which CCYs had pre-open run today

        def active_ccys(utc_hour: int) -> list:
            active = []
            for ccy, (start, end) in CCY_WINDOWS.items():
                if start > end:
                    if utc_hour >= start or utc_hour < end:
                        active.append(ccy)
                else:
                    if start <= utc_hour < end:
                        active.append(ccy)
            return active

        def pre_open_ccys(utc_hour: int, utc_date) -> list:
            """Return CCYs that need a pre-open catch-up run this hour."""
            due = []
            for ccy, pre_hr in PRE_OPEN_HOURS.items():
                key = f"{ccy}_{utc_date}"
                if utc_hour == pre_hr and key not in _pre_open_done:
                    due.append(ccy)
                    _pre_open_done.add(key)
            # Clean up old keys (keep only today)
            stale = {k for k in _pre_open_done if str(utc_date) not in k}
            _pre_open_done.difference_update(stale)
            return due

        ensure_heartbeat_table(conn)
        log.info(f"Starting time-aware poll loop every {args.loop} min. Ctrl+C to stop.")
        log.info("Active windows (UTC): AUD 21-08 | USD 10-21 | JPY 22-07 | EUR/GBP 07-16")
        log.info("Pre-open catch-up:    AUD @20 | USD @09 | JPY @21 | EUR/GBP @06")

        while True:
            now_utc  = datetime.now(timezone.utc)
            utc_hour = now_utc.hour
            utc_date = now_utc.date()

            ccys    = active_ccys(utc_hour)
            pre_ccys = pre_open_ccys(utc_hour, utc_date)

            if pre_ccys:
                log.info(f"UTC {utc_hour:02d}:xx — PRE-OPEN catch-up for: {', '.join(pre_ccys)}")
                try:
                    # Fetch last 90 minutes to catch any late-lodged trades
                    ins, upd = poll_once(conn, minutes_back=90)
                    log.info(f"  Pre-open: {ins} new, {upd} updated.")
                except Exception as e:
                    log.error(f"Pre-open poll error: {e}")

            if ccys:
                log.info(f"UTC {utc_hour:02d}:xx — active: {', '.join(ccys)}")
                try:
                    ins, upd = poll_once(conn, minutes_back=args.minutes)
                    log.info(f"Poll: {ins} new, {upd} updated.")
                except Exception as e:
                    log.error(f"Poll error: {e}")
            elif not pre_ccys:
                log.info(f"UTC {utc_hour:02d}:xx — no active markets, sleeping...")

            upsert_heartbeat(conn)  # heartbeat every poll regardless of activity
            time.sleep(args.loop * 60)

    else:
        ins, upd = poll_once(conn, minutes_back=args.minutes)
        log.info(f"Done: {ins} new, {upd} updated.")

    conn.close()

if __name__ == "__main__":
    main()
