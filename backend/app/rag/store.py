from __future__ import annotations

import hashlib
import re

import numpy as np

from app.core.config import settings

# Minimal, self-contained vector store. Avoids a heavy vector-DB dependency while
# exposing the same add/query interface — swappable for Chroma later. Uses Gemini
# embeddings when a key is present, else a deterministic hashing embedding so RAG
# works fully offline.

_DIM = 256


def _hash_embed(text: str) -> np.ndarray:
    """Deterministic bag-of-hashed-tokens embedding (offline fallback)."""
    vec = np.zeros(_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _gemini_embed(texts: list[str]) -> list[np.ndarray] | None:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        emb = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.google_api_key,
        )
        vecs = emb.embed_documents(texts)
        return [np.array(v, dtype=np.float32) for v in vecs]
    except Exception:
        return None


class VectorStore:
    def __init__(self) -> None:
        self._docs: list[str] = []
        self._meta: list[dict] = []
        self._vecs: list[np.ndarray] = []
        self._use_gemini = settings.has_llm

    @property
    def backend(self) -> str:
        return "gemini-embeddings" if self._use_gemini else "hash-embeddings"

    def add(self, docs: list[str], metas: list[dict] | None = None) -> None:
        metas = metas or [{} for _ in docs]
        vecs: list[np.ndarray] | None = None
        if self._use_gemini:
            vecs = _gemini_embed(docs)
            if vecs is None:  # key invalid / offline → degrade
                self._use_gemini = False
        if vecs is None:
            vecs = [_hash_embed(d) for d in docs]
        self._docs.extend(docs)
        self._meta.extend(metas)
        self._vecs.extend(vecs)

    def query(self, text: str, k: int = 3) -> list[dict]:
        if not self._vecs:
            return []
        qv: np.ndarray | None = None
        if self._use_gemini:
            got = _gemini_embed([text])
            qv = got[0] if got else None
        if qv is None:
            qv = _hash_embed(text)

        mat = np.vstack(self._vecs)
        sims = mat @ qv / (
            np.linalg.norm(mat, axis=1) * np.linalg.norm(qv) + 1e-9
        )
        top = np.argsort(-sims)[:k]
        return [
            {"text": self._docs[i], "meta": self._meta[i], "score": float(sims[i])}
            for i in top
        ]
