#!/usr/bin/env python3
"""
idb_capture.py  —  capture + HAR analysis for the 4 IDB transparency portals

Purpose: the TP ICAP / Fenics / Trad-X portals are JS apps. Rather than hand-write
click recipes, this tool opens the portal in a real browser, records ALL network
traffic to a HAR, then analyses the HAR to surface the actual data request(s)
the page makes (the XHR/fetch that returns the trade rows). That request URL is
what eu_transparency_puller.py then calls directly — no DOM scraping.

Run order per portal:
    python idb_capture.py capture tpicap        # opens browser, you click to
                                                 # Post-Trade > Rates, save HAR
    python idb_capture.py analyze <the.har>      # ranks candidate data endpoints,
                                                 # writes/updates endpoints.json
    # repeat for fenics, tradition
    python eu_transparency_puller.py --once --sources tp,icap,bgc,tradition

Or in one go:
    python idb_capture.py capture-all
    python idb_capture.py analyze-all

Needs:  pip install playwright && playwright install chromium

NOTE: endpoints.json is the contract between this tool and the puller. You can
also hand-edit it. If you'd rather, send me the HAR and I'll fill endpoints.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

OUT_DIR = Path("./eu_transparency_data")
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
ENDPOINTS_JSON = OUT_DIR / "endpoints.json"

# The 3 portals; TP + ICAP both publish via the TP ICAP portal, split later by MIC.
PORTALS = {
    "tpicap":    {"url": "https://www.tpicapmifidiidata.com/home",
                  "feeds": ["tp", "icap"],
                  "hint": "Open Post-Trade transparency, choose Rates/IRD, trigger the table or CSV."},
    "fenics":    {"url": "https://regdata.fenicsmd.com/home",
                  "feeds": ["bgc"],
                  "hint": "Open Post-Trade transparency for BGC OTF (BGCO) / GFI OTF (GFSO)."},
    "tradition": {"url": "https://www.trad-x.com/compliance-regulations",
                  "feeds": ["tradition"],
                  "hint": "Open the OTF post-trade link (not just the Trad-X MTF CLOB)."},
}

# Keywords that suggest a request is the data feed (not assets/telemetry).
URL_HINTS = ("trade", "transpar", "post", "rts", "publication", "deferr",
             "data", "csv", "export", "download", "report", "feed", "api",
             "search", "query", "result")
MIME_DATA = ("text/csv", "application/json", "application/octet-stream",
             "text/plain", "application/vnd.ms-excel")
NOISE = ("google", "analytics", "gtag", "doubleclick", "hotjar", "sentry",
         "cookie", "fonts", ".png", ".jpg", ".svg", ".css", ".woff", ".ico",
         "telemetry", "beacon")


def _need_playwright():
    try:
        from playwright.sync_api import sync_playwright  # noqa
        return sync_playwright
    except ImportError:
        sys.exit("Needs: pip install playwright && playwright install chromium")


def capture(target: str, dwell: int = 45) -> Path:
    sync_playwright = _need_playwright()
    portal = PORTALS.get(target)
    if not portal:
        sys.exit(f"unknown portal '{target}'. choices: {list(PORTALS)}")
    har_path = RAW_DIR / f"{target}_{int(time.time())}.har"
    print(f"\n=== CAPTURE {target} ===")
    print(portal["hint"])
    print(f"A real browser will open. You have ~{dwell}s: navigate to the rates "
          f"post-trade data and trigger the table/download. HAR -> {har_path}\n")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context(accept_downloads=True, record_har_path=str(har_path),
                                  record_har_content="embed")
        page = ctx.new_page()
        page.goto(portal["url"], wait_until="domcontentloaded", timeout=60000)
        # give the user time to click through; also catch any auto-download
        end = time.time() + dwell
        try:
            while time.time() < end:
                page.wait_for_timeout(1000)
        except Exception:
            pass
        page.screenshot(path=str(RAW_DIR / f"{target}_screenshot.png"), full_page=True)
        ctx.close()  # flushes HAR
        browser.close()
    print(f"saved {har_path}")
    return har_path


def _score_entry(entry: dict) -> tuple[int, dict] | None:
    req = entry.get("request", {})
    resp = entry.get("response", {})
    url = req.get("url", "")
    low = url.lower()
    if any(n in low for n in NOISE):
        return None
    content = resp.get("content", {}) or {}
    mime = (content.get("mimeType") or "").lower()
    size = content.get("size", 0) or resp.get("bodySize", 0) or 0
    score = 0
    if any(m in mime for m in MIME_DATA):
        score += 5
    if any(h in low for h in URL_HINTS):
        score += 3
    if req.get("method", "GET").upper() in ("GET", "POST"):
        score += 1
    if size and size > 2000:
        score += 2
    if size and size > 50000:
        score += 2
    # XHR/fetch resource types score higher when present
    rtype = (entry.get("_resourceType") or entry.get("resourceType") or "").lower()
    if rtype in ("xhr", "fetch"):
        score += 3
    if score <= 0:
        return None
    return score, {
        "url": url,
        "method": req.get("method", "GET"),
        "mime": mime,
        "size": size,
        "resource_type": rtype,
        "post_data": (req.get("postData") or {}).get("text", "")[:500],
        "query": urlparse(url).query[:300],
    }


def analyze(har_path: str, write: bool = True) -> list[dict]:
    p = Path(har_path)
    if not p.exists():
        sys.exit(f"no such HAR: {har_path}")
    har = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    entries = har.get("log", {}).get("entries", [])
    scored = []
    for e in entries:
        r = _score_entry(e)
        if r:
            scored.append(r)
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"\n=== HAR ANALYSIS: {p.name}  ({len(entries)} requests) ===")
    print(f"{'score':>5}  {'method':6} {'mime':22} {'size':>8}  url")
    cands = []
    for score, info in scored[:15]:
        print(f"{score:>5}  {info['method']:6} {info['mime'][:22]:22} {info['size']:>8}  {info['url'][:110]}")
        cands.append(info)
    if not cands:
        print("No obvious data endpoint found. The feed may be a websocket or "
              "an embedded download. Send me the HAR and I'll dig in.")
        return []

    # infer which portal this HAR is for, to slot into endpoints.json
    target = None
    for key, portal in PORTALS.items():
        host = urlparse(portal["url"]).netloc
        if any(host.split(".")[-2] in c["url"] for c in cands) or key in p.name:
            target = key
            break
    target = target or p.name.split("_")[0]

    top = cands[0]
    if write:
        store = {}
        if ENDPOINTS_JSON.exists():
            store = json.loads(ENDPOINTS_JSON.read_text())
        store[target] = {
            "url": top["url"],
            "method": top["method"],
            "format": "csv" if "csv" in top["mime"] or top["url"].lower().endswith(".csv") else
                      ("json" if "json" in top["mime"] else "auto"),
            "post_data": top["post_data"],
            "discovered_from": p.name,
            "_candidates": [c["url"] for c in cands[:5]],
        }
        ENDPOINTS_JSON.write_text(json.dumps(store, indent=2))
        print(f"\nwrote top candidate for '{target}' -> {ENDPOINTS_JSON}")
        print("If the top pick is wrong, edit endpoints.json and use one of _candidates.")
    return cands


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IDB transparency capture + HAR analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture"); c.add_argument("target"); c.add_argument("--dwell", type=int, default=45)
    sub.add_parser("capture-all").add_argument("--dwell", type=int, default=45, nargs="?")
    a = sub.add_parser("analyze"); a.add_argument("har")
    sub.add_parser("analyze-all")
    sub.add_parser("show")  # print endpoints.json

    args = ap.parse_args(argv)
    if args.cmd == "capture":
        capture(args.target, args.dwell)
    elif args.cmd == "capture-all":
        for t in PORTALS:
            capture(t, getattr(args, "dwell", 45) or 45)
    elif args.cmd == "analyze":
        analyze(args.har)
    elif args.cmd == "analyze-all":
        for har in sorted(RAW_DIR.glob("*.har")):
            analyze(str(har))
    elif args.cmd == "show":
        print(ENDPOINTS_JSON.read_text() if ENDPOINTS_JSON.exists() else "{} (none yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
