#!/usr/bin/env python3
"""
fenics_auto.py  —  fully automated BGC/Fenics daily pull.

Logs into regdata.fenicsmd.com headlessly (plain username/password form POST to
/login), captures a fresh JSESSIONID, writes it to fenics_cookie.txt, then runs
pull_fenics.py and load_idb_slices.py. Designed to be driven by a scheduled task
so BGC/GFI/Aurel swaptions land in eu_iro_prints with zero manual steps.

Credentials come from env vars (NEVER hardcode):
    FENICS_USER   your Fenics username
    FENICS_PASS   your Fenics password

Usage:
    python fenics_auto.py                 # login -> cookie -> pull -> load
    python fenics_auto.py --date 2026-06-12
    python fenics_auto.py --no-load       # refresh cookie + pull only
    python fenics_auto.py --cookie-only   # just refresh fenics_cookie.txt

Needs:  pip install playwright && playwright install chromium
Exit codes:  0 ok | 2 login failed (cookie not refreshed) | 3 pull/load failed
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COOKIE_FILE = HERE / "fenics_cookie.txt"
LOGIN_URL = "https://regdata.fenicsmd.com/login"
REPORTS_URL = "https://regdata.fenicsmd.com/reports"


def _log(msg: str) -> None:
    print(f"[fenics_auto] {msg}", flush=True)


def _heartbeat(login_ok: bool, reports_listed=None, detail: str = "") -> None:
    """Write the auto-pull status to fenics_heartbeat so the app can show whether
    the scheduled-task cookie is still good. Connects with DISCRETE fields so the
    password is passed RAW (handles spaces, !, @, etc. with no URL parsing).
    Only FENICS_DB_PASS is required; host/port/db/user default to the Supabase
    pooler but can be overridden by env vars. No-op if no password / psycopg."""
    pw = os.environ.get("FENICS_DB_PASS", "")
    if not pw.strip():
        # optional fallback: a full URL, if someone set one with no awkward chars
        _url = (os.environ.get("FENICS_DB_URL") or os.environ.get("DATABASE_URL") or "").strip()
        if not _url:
            _log("heartbeat: set FENICS_DB_PASS (raw password) — skipping")
            return
        _kwargs = {"dsn": _url}
    else:
        _kwargs = dict(
            host=os.environ.get("FENICS_DB_HOST", "aws-1-ap-southeast-1.pooler.supabase.com"),
            port=int(os.environ.get("FENICS_DB_PORT", "6543")),
            dbname=os.environ.get("FENICS_DB_NAME", "postgres"),
            user=os.environ.get("FENICS_DB_USER", "postgres.oxwbyotzdqccaajyaqhn"),
            password=pw,          # raw — spaces / ! / @ all fine here
        )
    try:
        import psycopg2
        conn = psycopg2.connect(**_kwargs)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO fenics_heartbeat (id, last_run, login_ok, reports_listed, detail) "
            "VALUES (1, now(), %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET last_run=now(), login_ok=EXCLUDED.login_ok, "
            "reports_listed=EXCLUDED.reports_listed, detail=EXCLUDED.detail",
            (bool(login_ok), reports_listed, (detail or "")[:500]),
        )
        conn.commit(); cur.close(); conn.close()
        _log("heartbeat written")
    except Exception as e:
        _log(f"heartbeat write failed: {e}")


def refresh_cookie() -> str | None:
    """Headless login -> return a fresh JSESSIONID (and write fenics_cookie.txt)."""
    user = os.environ.get("FENICS_USER", "").strip()
    pw = os.environ.get("FENICS_PASS", "").strip()
    if not (user and pw):
        _log("ERROR: set FENICS_USER and FENICS_PASS env vars.")
        return None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _log("ERROR: needs  pip install playwright && playwright install chromium")
        return None

    with sync_playwright() as pw_ctx:
        browser = pw_ctx.chromium.launch(headless=True)
        # Fenics TLS cert is expired -> ignore https errors
        ctx = browser.new_context(ignore_https_errors=True)
        page = ctx.new_page()
        try:
            page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=60000)
            # Form fields confirmed from the login page DOM:
            #   <input name="username" id="username">  + a password input + LOG IN submit
            page.fill("input[name='username']", user, timeout=20000)
            # password field has no stable id in the markup; target by type within the form
            page.fill("form[action='/login'] input[type='password']", pw, timeout=20000)
            # submit: the LOG IN button (button or input submit inside the login form)
            try:
                page.click("form[action='/login'] button[type='submit']", timeout=8000)
            except Exception:
                try:
                    page.click("form[action='/login'] input[type='submit']", timeout=8000)
                except Exception:
                    # fall back to the visible "LOG IN" text / Enter key
                    try:
                        page.get_by_text("LOG IN").click(timeout=8000)
                    except Exception:
                        page.keyboard.press("Enter")
            # wait for the post-login navigation away from /login
            try:
                page.wait_for_url(lambda u: "/login" not in u, timeout=30000)
            except Exception:
                page.wait_for_timeout(4000)
            # confirm we're authenticated by loading /reports
            page.goto(REPORTS_URL, wait_until="domcontentloaded", timeout=60000)
            html = (page.content() or "").lower()
            if "intraday/" not in html and ("password" in html or "log in" in html):
                _log("ERROR: still on a login page after submit — bad credentials?")
                ctx.close(); browser.close()
                return None
            # pull JSESSIONID from the context cookies
            jsid = None
            for c in ctx.cookies():
                if c.get("name") == "JSESSIONID":
                    jsid = c.get("value")
                    break
            ctx.close(); browser.close()
            if not jsid:
                _log("ERROR: logged in but no JSESSIONID cookie found.")
                return None
            COOKIE_FILE.write_text(jsid)
            _log(f"OK: fresh JSESSIONID written to {COOKIE_FILE.name}")
            return jsid
        except Exception as e:
            _log(f"ERROR during login: {e}")
            try:
                ctx.close(); browser.close()
            except Exception:
                pass
            return None


def _run(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(HERE / script)] + extra
    _log("run: " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Automated Fenics login + pull + load")
    ap.add_argument("--date", help="filter to YYYY-MM-DD")
    ap.add_argument("--no-load", action="store_true", help="refresh cookie + pull only")
    ap.add_argument("--cookie-only", action="store_true", help="only refresh the cookie")
    ap.add_argument("--folder", default=str(HERE / "fenics_files"),
                    help="folder load_idb_slices.py reads (default ./fenics_files)")
    args = ap.parse_args(argv)

    jsid = refresh_cookie()
    if not jsid:
        _log("login failed — cookie NOT refreshed. Aborting.")
        _heartbeat(False, None, "login failed — cookie not refreshed")
        return 2
    _heartbeat(True, None, "login ok")
    if args.cookie_only:
        return 0

    pull_args = ["--date", args.date] if args.date else []
    rc = _run("pull_fenics.py", pull_args)
    if rc != 0:
        _log(f"pull_fenics.py exited {rc}")
        return 3
    if args.no_load:
        return 0

    rc = _run("load_idb_slices.py", ["--folder", args.folder, "--sink", "supabase"])
    if rc != 0:
        _log(f"load_idb_slices.py exited {rc}")
        return 3
    _log("done: cookie refreshed, pulled, loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
