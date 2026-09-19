# Annotation store

FastAPI over SQLite. Holds the annotations the viewer produces.

## Running it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then start the viewer as usual. It looks for the backend on
`http://localhost:8000`, and the header says which store is in use:
`annotations → SQLite` or `annotations → this browser`.

Interactive API docs are at <http://localhost:8000/docs>.

With Docker instead:

```bash
docker build -t rfi-annotations backend
docker run -p 8000:8000 -v "$PWD/backend:/app" rfi-annotations
```

## What it stores

| Table | Holds |
|---|---|
| `annotations` | One row per tag, in the stand-off shape: `segment_id`, `start`, `end`, `quote`, `tag`, `scheme`, `coder`, `created` |
| `reviewed` | Segments read and judged to contain nothing. Without it, "read and empty" is indistinguishable from "not read yet" |

Offsets are code-point offsets into `segment.text`, counted the way Python's
`len()` counts. The server stores them and does not reinterpret them.

## Endpoints

| | |
|---|---|
| `GET /health` | Liveness, and how many annotations are stored |
| `GET /annotations` | All, or one segment's with `?segment_id=` |
| `PUT /annotations/{ann_id}` | Upsert |
| `DELETE /annotations/{ann_id}` | Remove |
| `GET /tags` | The tag registry with counts |
| `GET /progress` | Segments tagged, segments reviewed, and tags with fewer than 50 examples |
| `GET /reviewed`, `PUT`/`DELETE /reviewed/{segment_id}` | The reviewed flag |

## Two decisions worth knowing

**The client mints `ann_id`.** An annotation made while the backend is down
keeps its identity when it is synced, and replaying a write cannot duplicate
it. The viewer writes to localStorage as well, always, so a backend that stops
mid-session does not take the session's work with it.

**There is no authentication.** The database holds one researcher's coding of
public documents on one machine. Accounts would be scaffolding for a problem
this project does not have. `coder` is on every row so a second coder's work
can land beside the first's when reliability needs measuring.

## Not done yet

`GET /progress` reports which tags have fewer than 50 examples, because a class
that thin cannot be trained on — but nothing consumes it yet. The viewer should
show it during a labelling pass rather than after.
