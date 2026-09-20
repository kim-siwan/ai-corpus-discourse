"""Fetch the NITRD respondent table.

The docket page renders a DataTable whose source is a single JSON file. It
carries the one piece of metadata the corpus most needs and IFP does not
supply: the respondent type each submitter declared, as the agency recorded
it. `org_type_ifp` is an LLM's guess at the same thing.

    python3 scripts/01b_snapshot_nitrd.py

Output: data/raw/nitrd_responses_YYYY-MM-DD.json
"""

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

URL = "https://www.nitrd.gov/data-tables/rfi-90fr9088/responses.txt"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"nitrd_responses_{date.today().isoformat()}.json"

    print(f"fetching {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "ai-corpus-discourse/0.1"})
    with urllib.request.urlopen(req, timeout=120) as res:
        # The file is served with a UTF-8 byte-order mark, which json.loads
        # rejects. utf-8-sig strips it when present and is a no-op otherwise.
        raw = res.read().decode("utf-8-sig")

    parsed = json.loads(raw)
    rows = parsed["data"] if isinstance(parsed, dict) and "data" in parsed else parsed
    if not isinstance(rows, list) or not rows:
        sys.exit("unexpected shape; expected a list of rows")

    target.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("RespondentType") or "(empty)"] = counts.get(r.get("RespondentType") or "(empty)", 0) + 1

    print(f"\n{len(rows):,} rows -> {target.relative_to(OUT.parent.parent)}")
    print(f"fields: {', '.join(rows[0].keys())}\n")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6,}  {k}")
    print(f"\n  {sum(v for k, v in counts.items() if k != 'Individual'):6,}  non-Individual")

    # How far the official listing and the corpus we built from IFP agree.
    docs_path = OUT.parent / "documents.json"
    if docs_path.exists():
        ours = {d["url"].rsplit("/", 1)[-1] for d in json.loads(docs_path.read_text())}
        official = {r["FileName"] for r in rows if r.get("RespondentType") != "Individual"}
        print(f"\n  corpus documents        {len(ours):6,}")
        print(f"  official non-Individual {len(official):6,}")
        print(f"  in both                 {len(ours & official):6,}")
        print(f"  official only           {len(official - ours):6,}")
        print(f"  corpus only             {len(ours - official):6,}")
        for name in sorted(official - ours)[:10]:
            print(f"    official only: {name}")
        for name in sorted(ours - official)[:10]:
            rt = next((r.get("RespondentType") for r in rows if r.get("FileName") == name), "not listed")
            print(f"    corpus only:   {name}  [{rt}]")


if __name__ == "__main__":
    main()
