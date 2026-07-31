#!/usr/bin/env bash
set -euxo pipefail
export PLAYWRIGHT_BROWSERS_PATH=/opt/render/project/src/.playwright-browsers
pip install --upgrade pip
pip install playwright psycopg2-binary requests pandas openpyxl lxml
playwright install chromium
ls -la /opt/render/project/src/.playwright-browsers/
find /opt/render/project/src/.playwright-browsers -name 'chrome-headless-shell' -o -name 'chrome' | head