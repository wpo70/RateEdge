#!/usr/bin/env python3
"""
pull_fenics.py  —  fetch BGC-group (BGC/GFI/Sunrise/Aurel) MiFID post-trade slice
files from the Fenics Market Data portal (regdata.fenicsmd.com).

Flow:
  1. GET /reports  (cookie + X-Requested-With) -> HTML table listing every
     INTRADAY_TRADES_<venue>_<date>_<seq> report and its download id.
  2. GET /reports/download/intraday/<id>  -> the .zip (saved to fenics_files/).

Auth: a session cookie JSESSIONID. The Fenics session is short-lived, so when it
expires you refresh it: log into https://regdata.fenicsmd.com, then DevTools ->
Application -> Cookies -> copy JSESSIONID, and paste it into fenics_cookie.txt
(next to this script) or env FENICS_COOKIE. Format: just the JSESSIONID value,
or the full "cookieconsent_status=dismiss; JSESSIONID=...".

The Fenics TLS cert is expired, so verification is disabled for this host.

Usage:
  python pull_fenics.py                      # pull TRADES reports listed now
  python pull_fenics.py --date 2026-06-09    # only that date
  python pull_fenics.py --orders             # also pull pre-trade ORDERS reports
Downloads land in ./fenics_files (then load_fenics.py ingests them).
"""
from __future__ import annotations
import argparse, os, re, ssl, sys, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "fenics_files"
COOKIE_FILE = HERE / "fenics_cookie.txt"
BASE = "https://regdata.fenicsmd.com"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

# href="reports/download/intraday/<id>"  immediately followed (same row) by the
# report name in a later cell. We capture id + name by scanning the table.
ID_RE = re.compile(r"intraday/([0-9a-fA-F]{16,})")
ROW_RE = re.compile(
    r"intraday/([0-9a-fA-F]{16,})\"[^>]*>\s*(INTRADAY_(?:TRADES|ORDERS)_[A-Z]+_\d{4}_\d{2}_\d{2}_\d+)",
    re.I)


def _cookie() -> str:
    c = os.environ.get("FENICS_COOKIE", "").strip()
    if not c and COOKIE_FILE.exists():
        c = COOKIE_FILE.read_text().strip()
    if not c:
        sys.exit(f"No cookie. Put JSESSIONID in {COOKIE_FILE} or env FENICS_COOKIE.")
    if "JSESSIONID" not in c:                       # bare value given
        c = f"cookieconsent_status=dismiss; JSESSIONID={c}"
    return c


_AUTH_FAILED = False  # set True on a 401/403 — signals an expired/invalid cookie


def _get(url: str, cookie: str) -> bytes | None:
    global _AUTH_FAILED
    req = urllib.request.Request(url, headers={
        "Cookie": cookie, "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0", "Referer": f"{BASE}/dashboard"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 302, 303, 307):   # redirect to login = dead session
            _AUTH_FAILED = True
        print(f"  HTTP {e.code} on {url}")
    except Exception as e:
        print(f"  error on {url}: {e}")
    return None


def _cookie_expired_banner() -> None:
    """Loud, unmissable warning + non-zero exit so the scheduler surfaces it."""
    msg = (
        "\n" + "!" * 68 + "\n"
        "  FENICS COOKIE EXPIRED — NO BGC DATA PULLED\n"
        "  Refresh JSESSIONID: log in at https://regdata.fenicsmd.com,\n"
        "  DevTools -> Application -> Cookies -> copy JSESSIONID, paste into\n"
        f"  {COOKIE_FILE}\n"
        "  then re-run this puller and load the slices.\n"
        + "!" * 68
    )
    print(msg, file=sys.stderr)
    sys.exit(2)


def _list(cookie: str):
    """Return list of (id, name) for every report row on /reports."""
    b = _get(f"{BASE}/reports", cookie)
    if _AUTH_FAILED:                       # definitive: 401/403 from _get
        _cookie_expired_banner()
    if not b:                              # no response — network or dead session
        print("  no response from Fenics /reports (network or session issue)",
              file=sys.stderr)
        sys.exit(1)
    html = b.decode("utf-8", "ignore")
    low = html.lower()
    # Expired session = a login screen, NOT a valid-but-empty report list.
    # Only trip on login-FORM markers so "No data available" while logged in passes.
    looks_like_login = "intraday/" not in html and (
        'type="password"' in low or "password" in low
        or "sign in" in low or "signin" in low or "/login" in low)
    if looks_like_login:
        _cookie_expired_banner()
    pairs = ROW_RE.findall(html)
    # de-dupe, preserve order
    seen, out = set(), []
    for _id, name in pairs:
        if _id not in seen:
            seen.add(_id); out.append((_id, name))
    # any ids without a matched name (regex miss) -> pull with id as filename
    for _id in ID_RE.findall(html):
        if _id not in seen:
            seen.add(_id); out.append((_id, f"FENICS_{_id}"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch BGC/Fenics MiFID slices")
    ap.add_argument("--date", help="filter to YYYY-MM-DD")
    ap.add_argument("--orders", action="store_true", help="also pull pre-trade ORDERS")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cookie = _cookie()
    items = _list(cookie)
    if not items:
        print("No reports listed."); return 1

    ymd = args.date.replace("-", "_") if args.date else None
    want = []
    for _id, name in items:
        if not args.orders and "_ORDERS_" in name:
            continue
        if ymd and ymd not in name:
            continue
        want.append((_id, name))

    print(f"listed {len(items)} reports; {len(want)} to fetch")
    got = 0
    for _id, name in want:
        dest = OUT_DIR / (name if name.endswith(".zip") else f"{name}.zip")
        if dest.exists():
            continue
        data = _get(f"{BASE}/reports/download/intraday/{_id}", cookie)
        if not data:
            continue
        if data[:2] != b"PK":           # not a zip (maybe raw csv) — save as-is
            dest = OUT_DIR / (name + (".csv" if data[:1] in (b'C', b'"', b'I') else ".bin"))
        dest.write_bytes(data)
        got += 1
        print(f"  downloaded {dest.name}")
    print(f"done: {got} new file(s) in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
