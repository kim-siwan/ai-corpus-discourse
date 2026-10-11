"""Annotation store for the RFI viewer.

The viewer can run without this: annotations fall back to the browser's
localStorage. What the backend adds is durability across machines and browsers,
and a place for a second coder's work to land next to the first's — which the
`coder` field on every annotation exists for.

    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Local only. There is no authentication, and none is wanted: the database holds
one researcher's coding of public documents, and adding accounts now would be
scaffolding for a problem this project does not have.
"""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import predict

DB = Path(__file__).parent / "annotations.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    ann_id      TEXT PRIMARY KEY,
    segment_id  TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    start       INTEGER NOT NULL,
    end         INTEGER NOT NULL,
    quote       TEXT NOT NULL DEFAULT '',
    tag         TEXT NOT NULL,
    scheme      TEXT NOT NULL,
    coder       TEXT NOT NULL,
    created     TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_ann_segment ON annotations(segment_id);
CREATE INDEX IF NOT EXISTS ix_ann_tag     ON annotations(tag);

-- "Read and judged to contain nothing" is different from "not looked at yet",
-- and only this table can tell them apart.
CREATE TABLE IF NOT EXISTS reviewed (
    segment_id  TEXT PRIMARY KEY,
    coder       TEXT NOT NULL,
    at          TEXT NOT NULL
);
"""


class Annotation(BaseModel):
    ann_id: str
    segment_id: str
    doc_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str = ""
    # Free tags, not a fixed codebook: the scheme is meant to emerge from the
    # data in October rather than be imposed now.
    tag: str = Field(min_length=1, max_length=120)
    scheme: str = "freeform-v0"
    coder: str = "kim"
    created: str
    note: str = ""


class Reviewed(BaseModel):
    segment_id: str
    coder: str = "kim"
    at: str


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


app = FastAPI(title="RFI annotation store", version="0.1.0")

# The viewer is served by Vite on another port during development, so the
# browser treats it as a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict.router)  # HW2: POST /predict, GET /predict/segment/{id}


@app.on_event("startup")
def create_schema() -> None:
    with closing(connect()) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@app.get("/health")
def health() -> dict:
    with closing(connect()) as conn:
        n = conn.execute("SELECT count(*) FROM annotations").fetchone()[0]
    return {"ok": True, "annotations": n, "db": DB.name}


@app.get("/annotations")
def list_annotations(segment_id: str | None = None) -> list[dict]:
    sql = "SELECT * FROM annotations"
    args: tuple = ()
    if segment_id:
        sql += " WHERE segment_id = ?"
        args = (segment_id,)
    sql += " ORDER BY created"
    with closing(connect()) as conn:
        return [dict(r) for r in conn.execute(sql, args)]


@app.put("/annotations/{ann_id}")
def put_annotation(ann_id: str, body: Annotation) -> dict:
    """Upsert, keyed on the id the client generated.

    The client mints ann_id so an annotation made while the backend is down
    keeps its identity when it is synced, and so replaying the same write twice
    cannot duplicate it.
    """
    if ann_id != body.ann_id:
        raise HTTPException(400, "ann_id in the path and the body must match")
    if body.end < body.start:
        raise HTTPException(400, "end must not precede start")

    with closing(connect()) as conn:
        conn.execute(
            """INSERT INTO annotations
               (ann_id, segment_id, doc_id, start, end, quote, tag, scheme, coder, created, note)
               VALUES (:ann_id, :segment_id, :doc_id, :start, :end, :quote, :tag, :scheme,
                       :coder, :created, :note)
               ON CONFLICT(ann_id) DO UPDATE SET
                   tag = excluded.tag, quote = excluded.quote, note = excluded.note""",
            body.model_dump(),
        )
        conn.commit()
    return body.model_dump()


@app.delete("/annotations/{ann_id}")
def delete_annotation(ann_id: str) -> dict:
    with closing(connect()) as conn:
        cur = conn.execute("DELETE FROM annotations WHERE ann_id = ?", (ann_id,))
        conn.commit()
    return {"deleted": cur.rowcount}


FULLTEXT = Path(__file__).parent.parent / "pipeline" / "data" / "fulltext"


@app.get("/search")
def search(q: str, limit: int = 1000) -> dict:
    """Full-text search across the submissions.

    The viewer holds paragraph offsets, not paragraph text — 14 MB of it stays
    on disk — so it cannot search bodies on its own. Scanning the extracted
    files here takes a few tens of milliseconds and keeps the browser from
    downloading the corpus to do it. Without the backend the viewer still
    filters by organization and topic, and says that body search is off.
    """
    needle = q.strip().lower()
    if len(needle) < 2:
        raise HTTPException(400, "query must be at least two characters")
    if not FULLTEXT.exists():
        raise HTTPException(503, "no extracted text on this machine")

    hits = []
    for path in sorted(FULLTEXT.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        at = text.lower().find(needle)
        if at < 0:
            continue
        start = max(0, at - 60)
        hits.append({
            "doc_id": path.stem,
            "count": text.lower().count(needle),
            "first_offset": at,
            "excerpt": ("…" if start else "") + text[start:at + len(needle) + 90].strip(),
        })
    # Rank every match before cutting, not after: stopping at the limit while
    # scanning in file-name order returned the alphabetically first 200
    # matches, so a common word hid every submission after the letter H.
    # The default limit is above the corpus size; the viewer wants them all.
    hits.sort(key=lambda h: -h["count"])
    return {"query": q, "documents": len(hits), "hits": hits[:limit]}


@app.get("/tags")
def tags() -> list[dict]:
    """The tag registry, most used first. Keeps `literacy` and `AI literacy`
    from accumulating as separate tags for the same idea."""
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT tag, count(*) AS n FROM annotations GROUP BY tag ORDER BY n DESC, tag"
        )
        return [dict(r) for r in rows]


@app.get("/progress")
def progress() -> dict:
    """What a labelling session needs to know: how far it has got, and whether
    any class is still too small to train on."""
    with closing(connect()) as conn:
        tagged = conn.execute("SELECT count(DISTINCT segment_id) FROM annotations").fetchone()[0]
        reviewed = conn.execute("SELECT count(*) FROM reviewed").fetchone()[0]
        thin = conn.execute(
            "SELECT tag, count(*) AS n FROM annotations GROUP BY tag HAVING n < 50 ORDER BY n"
        ).fetchall()
    return {
        "segments_tagged": tagged,
        "segments_reviewed": reviewed,
        "tags_under_50": [dict(r) for r in thin],
    }


@app.get("/reviewed")
def list_reviewed() -> list[str]:
    with closing(connect()) as conn:
        return [r["segment_id"] for r in conn.execute("SELECT segment_id FROM reviewed")]


@app.put("/reviewed/{segment_id}")
def set_reviewed(segment_id: str, body: Reviewed) -> dict:
    with closing(connect()) as conn:
        conn.execute(
            "INSERT INTO reviewed (segment_id, coder, at) VALUES (?, ?, ?) "
            "ON CONFLICT(segment_id) DO UPDATE SET at = excluded.at",
            (segment_id, body.coder, body.at),
        )
        conn.commit()
    return {"segment_id": segment_id, "reviewed": True}


@app.delete("/reviewed/{segment_id}")
def unset_reviewed(segment_id: str) -> dict:
    with closing(connect()) as conn:
        conn.execute("DELETE FROM reviewed WHERE segment_id = ?", (segment_id,))
        conn.commit()
    return {"segment_id": segment_id, "reviewed": False}
