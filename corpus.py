from __future__ import annotations
"""Corpus wrapper utilities.

This module provides a light-weight wrapper around a DataFrame-based corpus
with two required columns: an identifier column (default: "id") and a text
column (default: "text"). It computes & stores sentence-transformer embeddings
and exposes convenience retrieval helpers.

Typical usage:

    from sentence_transformers import SentenceTransformer
    import pandas as pd
    from corpus import CorpusWrapper

    df = pd.read_csv("corpus.csv")  # must contain columns id,text
    embedder = SentenceTransformer("julian-schelb/SPhilBerta-latin-intertextuality")
    corpus = CorpusWrapper(df, embedder)
    results = corpus.search("arma virumque", top_k=5)
    print(results)

The search method returns a list of dictionaries sorted by descending cosine
similarity. Embeddings are computed once at initialization.
"""

from dataclasses import dataclass
from typing import List, Sequence, Optional, Iterable, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedItem:
    """Container for a retrieved corpus segment."""
    id: Any
    text: str
    score: float  # cosine similarity (0..1 or -1..1 depending on model normalization)

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "score": self.score}


class CorpusWrapper:
    """In-memory corpus index built on SentenceTransformer embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the id and text columns.
    embedder : SentenceTransformer
        Loaded sentence-transformer model used for encoding.
    id_col : str, default 'id'
        Column containing unique identifiers (does not have to be strictly unique but recommended).
    text_col : str, default 'text'
        Column containing the textual content to embed.
    normalize : bool, default True
        L2-normalize embeddings to enable fast cosine similarity via dot product.
    batch_size : int, default 32
        Batch size passed to SentenceTransformer.encode.
    show_progress_bar : bool, default False
        Forwarded to SentenceTransformer.encode.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embedder: SentenceTransformer,
        *,
        id_col: str = "id",
        text_col: str = "text",
        normalize: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        if id_col not in df.columns or text_col not in df.columns:
            missing = {id_col, text_col} - set(df.columns)
            raise ValueError(f"DataFrame missing required column(s): {', '.join(missing)}")
        self._df = df.reset_index(drop=True)
        self.id_col = id_col
        self.text_col = text_col
        self.embedder = embedder
        self.normalize = normalize

        texts: List[str] = self._df[text_col].astype(str).tolist()
        # Compute embeddings once.
        embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        self.embeddings = embeddings  # shape (N, D)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._df)

    def ids(self) -> Sequence[Any]:
        return self._df[self.id_col].tolist()

    def texts(self) -> Sequence[str]:
        return self._df[self.text_col].astype(str).tolist()

    # ------------------------------------------------------------------
    def get_text(self, id_value: Any) -> Optional[str]:
        """Return text for a given id (first match) or None."""
        matches = self._df[self._df[self.id_col] == id_value]
        if matches.empty:
            return None
        return matches.iloc[0][self.text_col]

    # ------------------------------------------------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode(query, convert_to_numpy=True)
        if self.normalize:
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        return vec

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """Return top_k most similar corpus rows to the query.

        Uses cosine similarity. If embeddings are normalized, dot products are
        equivalent to cosine similarity and faster.
        """
        if len(self) == 0:
            return []
        q = self._embed_query(query)
        if self.normalize:
            # dot product == cosine
            scores = self.embeddings @ q
        else:
            # fall back to cosine similarity
            # (v dot q) / (||v|| * ||q||) ; precompute norms
            v_norms = np.linalg.norm(self.embeddings, axis=1)
            q_norm = np.linalg.norm(q)
            denom = v_norms * (q_norm if q_norm != 0 else 1.0)
            raw = self.embeddings @ q
            denom[denom == 0] = 1.0
            scores = raw / denom
        # Argpartition for efficiency
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        # Sort indices by score descending
        idx = idx[np.argsort(-scores[idx])]
        items: List[RetrievedItem] = []
        for i in idx:
            row = self._df.iloc[i]
            items.append(
                RetrievedItem(
                    id=row[self.id_col],
                    text=row[self.text_col],
                    score=float(scores[i]),
                )
            )
        return items

    # ------------------------------------------------------------------
    def calc_rank(self, query: str, candidate_text: str) -> tuple[int, int, float]:
        """Compute 1-based rank of candidate_text among corpus documents for query.

        The candidate text is NOT added permanently; it's encoded on the fly.

        Returns (rank, total_docs_including_candidate, candidate_score)
        If corpus empty, returns (1, 1, 0.0).
        """
        if len(self) == 0:
            # Only candidate exists
            return 1, 1, 0.0
        q_vec = self._embed_query(query)
        c_vec = self.embedder.encode(candidate_text, convert_to_numpy=True)
        if self.normalize:
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                c_vec = c_vec / c_norm
            # dot products == cosine
            scores = self.embeddings @ q_vec  # shape (N,)
            candidate_score = float(c_vec @ q_vec)
        else:
            # cosine for corpus
            v_norms = np.linalg.norm(self.embeddings, axis=1)
            q_norm = np.linalg.norm(q_vec)
            denom = v_norms * (q_norm if q_norm != 0 else 1.0)
            raw = self.embeddings @ q_vec
            denom[denom == 0] = 1.0
            scores = raw / denom
            # candidate cosine
            c_norm = np.linalg.norm(c_vec)
            denom_c = (q_norm if q_norm != 0 else 1.0) * (c_norm if c_norm != 0 else 1.0)
            candidate_score = float((c_vec @ q_vec) / denom_c) if denom_c != 0 else 0.0
        better = int(np.sum(scores > candidate_score))
        rank = better + 1
        total = len(scores) + 1
        return rank, total, candidate_score

    # ------------------------------------------------------------------
    def as_dataframe(self) -> pd.DataFrame:
        """Return underlying DataFrame (copy)."""
        return self._df.copy()

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "size": len(self),
            "id_col": self.id_col,
            "text_col": self.text_col,
            "normalized": self.normalize,
        }


__all__ = ["CorpusWrapper", "RetrievedItem"]
