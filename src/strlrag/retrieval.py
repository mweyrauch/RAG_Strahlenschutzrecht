"""Retriever mit einheitlicher Schnittstelle.

Alle Retriever implementieren:
    fit(chunks)          -> self
    suche(frage, k)      -> list[Treffer] (absteigend nach Score)

Enthalten:
- DenseRetriever       Satz-Embeddings ueber sentence-transformers (Haupt-Retriever)
- BM25Retriever        Okapi BM25 auf Wort-Tokens mit CISTEM-Stemming (Basislinie zum Vergleich)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .chunking import Chunk
from .text import Analysator

# Zwischenspeicher fuer Embeddings (nicht versioniert, siehe .gitignore)
_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"


@dataclass(frozen=True)
class Treffer:
    index: int
    score: float
    chunk: Chunk


class Retriever:
    name = "basis"

    def fit(self, chunks: list[Chunk]) -> "Retriever":
        self.chunks = list(chunks)
        self._fit([c.index_text for c in self.chunks])
        return self

    def _fit(self, texte: list[str]) -> None:  # pragma: no cover - abstrakt
        raise NotImplementedError

    def scores(self, frage: str) -> np.ndarray:  # pragma: no cover - abstrakt
        raise NotImplementedError

    def suche(self, frage: str, k: int = 10) -> list[Treffer]:
        s = self.scores(frage)
        k = min(k, len(s))
        if k == 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.lexsort((idx, -s[idx]))]  # stabil: Score absteigend, dann Index
        return [Treffer(int(i), float(s[i]), self.chunks[i]) for i in idx]


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.2, b: float = 0.75, stemming: bool = True):
        self.k1 = k1
        self.b = b
        self.stemming = stemming
        self.name = "bm25"

    def _fit(self, texte):
        self.vec = CountVectorizer(analyzer=Analysator(stemming=self.stemming))
        tf = self.vec.fit_transform(texte).tocsr().astype(np.float64)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 0.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
        # Gewichtete Matrix W[d, t] = idf_t * tf (k1 + 1) / (tf + k1 (1 - b + b dl/avgdl))
        tf = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * dl[tf.row] / max(avgdl, 1e-9))
        w = self.idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm)
        self.matrix = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, frage):
        q = self.vec.transform([frage])
        q.data[:] = 1.0  # jedes Anfrage-Token einmal zaehlen
        return (self.matrix @ q.T).toarray().ravel()


class DenseRetriever(Retriever):
    """Embedding-Retriever. Benoetigt das Paket sentence-transformers.

    Standardmodell: intfloat/multilingual-e5-small. E5-Modelle erwarten die Praefixe
    "query: " und "passage: "; fuer andere Modelle bleiben sie leer, sofern nicht
    ausdruecklich angegeben. Die Embeddings des Korpus werden in cache_dir
    zwischengespeichert (Schluessel: Modell, Praefix und alle Texte), damit wiederholte Laeufe
    den Korpus nicht erneut kodieren.
    """

    def __init__(self, modell: str = "intfloat/multilingual-e5-small",
                 frage_praefix: str | None = None, text_praefix: str | None = None,
                 cache_dir: str | Path | None = None, batch_size: int = 32):
        e5 = "e5" in modell.lower()
        self.modell_name = modell
        self.frage_praefix = frage_praefix if frage_praefix is not None else ("query: " if e5 else "")
        self.text_praefix = text_praefix if text_praefix is not None else ("passage: " if e5 else "")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.name = "dense" if modell == "intfloat/multilingual-e5-small" else f"dense:{modell}"

    def _cache_pfad(self, texte: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256()
        for teil in [self.modell_name, self.text_praefix, *texte]:
            h.update(teil.encode("utf-8"))
            h.update(b"\x00")
        return self.cache_dir / f"emb_{h.hexdigest()[:20]}.npy"

    def _fit(self, texte):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - abhaengig von der Umgebung
            raise ImportError(
                "DenseRetriever benoetigt 'sentence-transformers' "
                "(pip install -e '.[dense]')") from e
        self.modell = SentenceTransformer(self.modell_name)
        pfad = self._cache_pfad(texte)
        if pfad is not None and pfad.exists():
            self.matrix = np.load(pfad)
            return
        self.matrix = np.asarray(self.modell.encode(
            [self.text_praefix + t for t in texte], batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
        if pfad is not None:
            pfad.parent.mkdir(parents=True, exist_ok=True)
            np.save(pfad, self.matrix)

    def scores(self, frage):
        q = np.asarray(self.modell.encode([self.frage_praefix + frage], normalize_embeddings=True))
        return (self.matrix @ q[0]).astype(float)


def baue_retriever(name: str, **kw) -> Retriever:
    """Fabrik fuer die Konfigurationsdateien."""
    if name == "bm25":
        return BM25Retriever(**kw)
    if name == "dense" or name.startswith("dense:"):
        # z. B. "dense" oder "dense:intfloat/multilingual-e5-base"
        if ":" in name:
            kw.setdefault("modell", name.split(":", 1)[1])
        kw.setdefault("cache_dir", _CACHE)
        return DenseRetriever(**kw)
    if name.startswith("umformuliert:"):
        # z. B. "umformuliert:bm25": Anfrage zusaetzlich von einem Sprachmodell umformulieren lassen
        from .anfrage import UmformulierRetriever
        from .antwort import OpenAIKompatibelBackend
        r = UmformulierRetriever(baue_retriever(name.split(":", 1)[1]), OpenAIKompatibelBackend())
        r.name = name
        return r
    raise ValueError(f"Unbekannter Retriever: {name}")
