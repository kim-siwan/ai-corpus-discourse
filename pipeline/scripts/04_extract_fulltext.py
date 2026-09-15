"""Extract the full submission text for every document in the corpus.

The segments carry an excerpt produced by IFP, not the submission itself. This
step pulls the underlying PDF text out of the NITRD archive so the viewer can
show a recommendation in the context it came from.

    python3 scripts/04_extract_fulltext.py

Reads  data/raw/NITRD-responses.zip  (not committed)
Writes data/fulltext/<doc_id>.txt    (one file per document, fetched on demand)
       data/fulltext/index.json      (sizes and extraction status)

Normalization is nfkc-v1, imported from the adapter rather than reimplemented:
the two must agree or offsets computed later will not line up.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

adapter = import_module("02_adapter_ifp")
normalize = adapter.normalize

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "raw" / "NITRD-responses.zip"
OUT = ROOT / "data" / "fulltext"
PREFIX = "90-fr-9088-combined-responses/"


def pdf_to_text(data: bytes) -> str:
    """pdftotext reads a PDF from stdin with '-' as both input and output."""
    proc = subprocess.run(
        ["pdftotext", "-layout", "-", "-"],
        input=data,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:200])
    return proc.stdout.decode("utf-8", "replace")


def main() -> None:
    if not ARCHIVE.exists():
        sys.exit(f"missing {ARCHIVE} — download it from nitrd.gov first")

    documents = json.loads((ROOT / "data" / "documents.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    index, failed = {}, []
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = set(zf.namelist())
        for i, doc in enumerate(documents, 1):
            filename = doc["url"].rsplit("/", 1)[-1]
            member = PREFIX + filename
            if member not in members:
                failed.append((doc["doc_id"], "not in archive"))
                continue
            try:
                text = normalize(pdf_to_text(zf.read(member)))
            except Exception as exc:
                failed.append((doc["doc_id"], str(exc)[:80]))
                continue
            if not text:
                failed.append((doc["doc_id"], "no extractable text (scanned?)"))
                continue
            (OUT / f"{doc['doc_id']}.txt").write_text(text, encoding="utf-8")
            index[doc["doc_id"]] = {"chars": len(text), "source": filename}
            if i % 50 == 0:
                print(f"  {i}/{len(documents)}", flush=True)

    (OUT / "index.json").write_text(
        json.dumps(
            {"normalization": adapter.NORMALIZATION, "documents": index},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    total = sum(v["chars"] for v in index.values())
    print(f"\nextracted {len(index)}/{len(documents)} documents, {total:,} chars")
    if failed:
        print(f"failed {len(failed)}:")
        for doc_id, why in failed[:10]:
            print(f"  {doc_id}: {why}")


if __name__ == "__main__":
    main()
