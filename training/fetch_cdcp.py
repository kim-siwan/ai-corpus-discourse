"""Download the CDCP corpus into training/data/cdcp/.

CDCP (Park & Cardie, LREC 2018) annotates 731 citizen comments on the CFPB's
debt-collection rulemaking docket with proposition types: fact, testimony,
value, policy, reference. The original release is the Cornell eRulemaking
Corpus (joonsuk.org). We fetch a JSON conversion with train/dev/test splits
from the public GitHub repo HITSZ-HLT/GMAM, pinned to one commit so the data
cannot change under us.

That repo has no license file, so the data is NOT committed here
(training/data/ is gitignored). Re-run this script to get it.

    python training/fetch_cdcp.py
"""

import hashlib
import json
import pathlib
import urllib.request

COMMIT = "9f83af9e37171586cc20449778ccaa917b217adb"
BASE = f"https://raw.githubusercontent.com/HITSZ-HLT/GMAM/{COMMIT}/data/cdcp"
SPLITS = ["train", "dev", "test"]

OUT = pathlib.Path(__file__).resolve().parent / "data" / "cdcp"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": f"github.com/HITSZ-HLT/GMAM@{COMMIT}", "files": {}}
    for split in SPLITS:
        url = f"{BASE}/{split}.json"
        dest = OUT / f"{split}.json"
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        json.loads(data)  # fail loudly if it is not JSON
        dest.write_bytes(data)
        manifest["files"][f"{split}.json"] = hashlib.sha256(data).hexdigest()
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(SPLITS)} splits to {OUT}")


if __name__ == "__main__":
    main()
