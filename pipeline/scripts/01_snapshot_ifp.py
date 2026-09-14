"""Fetch a one-time snapshot of the IFP AI Action Plan Database.

The snapshot pins the data to a date so later steps are reproducible and the
saved file serves as the provenance record. Run once.

    python3 scripts/01_snapshot_ifp.py

The endpoint builds ~5,000 records per request and is slow (roughly 8s warm),
so cold requests sometimes hit a gateway timeout. Retries handle that. If it
still fails, see the browser fallback in README.md.

Output: data/raw/ifp_airtable_YYYY-MM-DD.json
"""

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

API_URL = "https://www.aiactionplan.org/api/airtable"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ATTEMPTS = 5
TIMEOUT = 300


def fetch(url: str) -> object:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "rfi-discourse-research/0.1 (academic use; one-time snapshot)",
            "Accept": "application/json",
            "Referer": "https://www.aiactionplan.org/",
        },
    )
    for attempt in range(1, ATTEMPTS + 1):
        try:
            print(f"  attempt {attempt}/{ATTEMPTS} ...", end="", flush=True)
            started = time.time()
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
            print(f" ok ({time.time() - started:.1f}s)")
            return json.loads(body)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print(f" failed: {exc}")
            if attempt == ATTEMPTS:
                raise
            wait = 10 * attempt
            print(f"  waiting {wait}s before retry")
            time.sleep(wait)


def find_records(payload: object) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    raise ValueError("No record array found in response; inspect the payload.")


def main() -> int:
    print(f"GET {API_URL}")
    try:
        payload = fetch(API_URL)
    except Exception as exc:
        print(f"\nCould not reach the endpoint: {exc}", file=sys.stderr)
        print("Use the browser fallback described in README.md.", file=sys.stderr)
        return 1

    records = find_records(payload)
    print(f"Received {len(records):,} records")
    if records:
        print("Fields:", ", ".join(sorted(records[0].keys())))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ifp_airtable_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Add data/raw/ to .gitignore; do not commit this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
