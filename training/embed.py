"""Turn texts into fixed-length vectors, cached on disk by encoder name.

The encoder is frozen — it is a pre-trained part we do not train. Only the
MLP on top learns. Caching means the MLP can be retrained in seconds, which is
what makes interactive training in the web app possible later.

Encoders:
  minilm   sentence-transformers/all-MiniLM-L6-v2   384-d  (default)
  mpnet    sentence-transformers/all-mpnet-base-v2  768-d  (fallback if minilm is not enough)
  lsa      TF-IDF -> TruncatedSVD(300), fitted on CDCP train. No download;
           used to test the pipeline offline. Lexical, so do not expect it to
           help the MLP.

The cache key includes the encoder name, so swapping encoders never mixes
vectors. This is the seed of HW4's encoder-selection knob.
"""

import hashlib
import pathlib

import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent / "cache"

ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

_models = {}


def _key(encoder: str, texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(encoder.encode())
    for t in texts:
        h.update(b"\0" + t.encode())
    return h.hexdigest()[:16]


def embed(texts: list[str], encoder: str = "minilm", fit_texts: list[str] | None = None) -> np.ndarray:
    """Return an (n, d) float32 array. `fit_texts` is only used by `lsa`."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{encoder}-{_key(encoder, texts)}.npy"
    if path.exists():
        return np.load(path)

    if encoder in ENCODERS:
        from sentence_transformers import SentenceTransformer

        if encoder not in _models:
            _models[encoder] = SentenceTransformer(ENCODERS[encoder])
        vecs = _models[encoder].encode(texts, batch_size=64, show_progress_bar=True,
                                       normalize_embeddings=True)
    elif encoder == "lsa":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if fit_texts is None:
            raise ValueError("lsa needs fit_texts (the training texts)")
        if "lsa" not in _models:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            svd = TruncatedSVD(300, random_state=0)
            svd.fit(vec.fit_transform(fit_texts))
            _models["lsa"] = (vec, svd)
        vec, svd = _models["lsa"]
        vecs = normalize(svd.transform(vec.transform(texts)))
    else:
        raise ValueError(f"unknown encoder {encoder!r}; choose from {list(ENCODERS) + ['lsa']}")

    vecs = np.asarray(vecs, dtype=np.float32)
    np.save(path, vecs)
    return vecs
