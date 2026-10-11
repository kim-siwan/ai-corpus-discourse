"""Sentence-type predictions (HW2): fact / policy / testimony / value.

Loads the trained network from training/artifacts/ once, when the server
starts, and labels each sentence of a text:

    POST /predict                      {"text": "..."}  -> any pasted text
    GET  /predict/segment/{segment_id}                  -> one RFI paragraph

Offsets are code points into the text that was sent (or into the paragraph,
for a segment) — the same convention as the annotations table, so a
prediction can sit beside a hand-coded tag.

The model needs torch and sentence-transformers, which live in
training/.venv, so run the server from there to get predictions:

    training/.venv/bin/uvicorn app:app --app-dir backend --reload --port 8000

Started from backend/.venv instead, everything from HW1 works as before and
these two endpoints answer 503.

Predictions on RFI text are unvalidated: the network was trained on CDCP,
citizen comments on a CFPB rule, not on organizational submissions.
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "training" / "artifacts"
SEGMENTS = ROOT / "pipeline" / "data" / "segments.json"
FULLTEXT = ROOT / "pipeline" / "data" / "fulltext"

MODEL_INFO = {
    "name": "minilm-mlp",
    "scheme": "cdcp-pred-v1",
    "coder": "model:minilm-mlp",
    "trained_on": "CDCP (citizen comments, CFPB debt-collection rule)",
    "validated_on_rfi": False,
}


class Predictor:
    """The encoder and the trained MLP, loaded together so they cannot drift apart."""

    def __init__(self, artifacts: Path):
        import torch
        from sentence_transformers import SentenceTransformer

        # The network class comes from training/, not from a copy, so the
        # shape served here is the shape that was trained.
        sys.path.insert(0, str(artifacts.parent))
        from model import SentenceTypeMLP

        label_map = json.loads((artifacts / "label_map.json").read_text())
        config = json.loads((artifacts / "config.json").read_text())
        self.labels = label_map["labels"]
        self.model = SentenceTypeMLP(in_dim=label_map["input_dim"], hidden=config["hidden"],
                                     n_classes=len(self.labels), activation=config["activation"],
                                     dropout=config["dropout"])
        self.model.load_state_dict(torch.load(artifacts / "model.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.encoder = SentenceTransformer(label_map["encoder_model"], device="cpu")
        self.torch = torch

    def predict(self, text: str) -> dict:
        spans = split_sentences(text)
        sentences = []
        if spans:
            inputs = [model_text(text[start:end]) for start, end in spans]
            # Same encoder and normalization as in training (embed.py).
            X = self.encoder.encode(inputs, batch_size=64, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
            probs = self.model.predict_proba(self.torch.from_numpy(X)).tolist()
            for (start, end), p in zip(spans, probs):
                best = max(range(len(self.labels)), key=lambda i: p[i])
                sentences.append({
                    "start": start, "end": end, "text": text[start:end],
                    "label": self.labels[best],
                    "probs": {label: round(v, 4) for label, v in zip(self.labels, p)},
                })
        found = {s["label"] for s in sentences}
        return {
            "model": MODEL_INFO,
            "sentences": sentences,
            "labels": [label for label in self.labels if label in found],  # fixed order
            "contains_demand": "policy" in found,
        }


# ---------------------------------------------------------------- sentences

_segmenter = None


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) code-point offsets of each sentence in `text`.

    pdftotext leaves the PDF's line breaks inside sentences, and pysbd treats
    a newline as a boundary. So pysbd reads a copy in which every whitespace
    character is replaced by one space. The copy has the same length as the
    original, so an offset found in it is valid in the original too.

    Offsets come from finding each sentence in the copy, left to right, rather
    than from pysbd's own char_span, which can misplace repeated sentences.
    """
    global _segmenter
    if _segmenter is None:
        import pysbd
        _segmenter = pysbd.Segmenter(language="en", clean=False)

    flat = re.sub(r"\s", " ", text)
    spans, cursor = [], 0
    for piece in _segmenter.segment(flat):
        piece = piece.strip()
        if not piece:
            continue
        at = flat.find(piece, cursor)
        if at < 0:  # pysbd changed the text; skip it rather than guess an offset
            continue
        spans.append((at, at + len(piece)))
        cursor = at + len(piece)
    return spans


def model_text(sentence: str) -> str:
    """What the encoder sees: a word broken across lines joined again
    ("regu-\\nlation" -> "regulation", as the viewer hides that hyphen), and
    every run of whitespace collapsed to one space. Offsets are not affected:
    they always point into the original."""
    sentence = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", sentence)
    return " ".join(sentence.split())


# ---------------------------------------------------------------- loading


def _load() -> tuple[Predictor | None, str | None]:
    try:
        return Predictor(ARTIFACTS), None
    except ImportError as e:
        return None, f"prediction needs training/.venv ({e.name} is not installed here)"
    except FileNotFoundError as e:
        return None, f"no trained model: {e.filename} is missing"


# Once, when the server imports this module at startup.
predictor, unavailable = _load()


def _require() -> Predictor:
    if predictor is None:
        raise HTTPException(503, unavailable)
    return predictor


@lru_cache(maxsize=1)
def _segments() -> dict[str, dict]:
    """segment_id -> paragraph record. 31,384 entries, read on first use."""
    return {s["segment_id"]: s for s in json.loads(SEGMENTS.read_text())}


# ---------------------------------------------------------------- endpoints

router = APIRouter()


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


@router.post("/predict")
def predict_text(body: PredictRequest) -> dict:
    return _require().predict(body.text)


@router.get("/predict/segment/{segment_id}")
def predict_segment(segment_id: str) -> dict:
    p = _require()
    if not SEGMENTS.exists() or not FULLTEXT.exists():
        raise HTTPException(503, "no corpus data on this machine")
    seg = _segments().get(segment_id)
    if seg is None:
        raise HTTPException(404, f"unknown segment_id {segment_id!r}")
    doc = (FULLTEXT / f"{seg['doc_id']}.txt").read_text(encoding="utf-8")
    paragraph = doc[seg["char_offset_in_doc"]:seg["char_end_in_doc"]]
    return {"segment_id": segment_id, "doc_id": seg["doc_id"], **p.predict(paragraph)}
