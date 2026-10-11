"""Load CDCP as (sentence-level) propositions with their type.

Each CDCP comment is split into elementary units ("propositions"), each with
one type. We return one example per proposition.

`reference` is dropped: 40 training examples and a single test example, which
makes any per-class metric for it meaningless. It is reported in the README.
"""

import json
import pathlib
from dataclasses import dataclass

DATA = pathlib.Path(__file__).resolve().parent / "data" / "cdcp"

LABELS = ["fact", "policy", "testimony", "value"]  # fixed order = output index
DROPPED = {"reference"}


@dataclass
class Example:
    text: str
    label: str
    comment_id: str  # "<split>-<index>": lets us keep a comment inside one split


def load(split: str) -> list[Example]:
    path = DATA / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: python training/fetch_cdcp.py")
    out = []
    for i, comment in enumerate(json.loads(path.read_text())):
        words = comment["words"]
        for comp in comment["component_src_r"]:
            if comp["component"] in DROPPED:
                continue
            text = " ".join(words[comp["from"]:comp["to"]])
            out.append(Example(text=text, label=comp["component"], comment_id=f"{split}-{i}"))
    return out


def label_index(labels: list[str]) -> list[int]:
    return [LABELS.index(l) for l in labels]


if __name__ == "__main__":
    from collections import Counter

    for split in ["train", "dev", "test"]:
        ex = load(split)
        print(split, len(ex), dict(Counter(e.label for e in ex)))
