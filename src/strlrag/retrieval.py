"""Retriever mit einheitlicher Schnittstelle.

Alle Retriever implementieren:
    fit(chunks)          -> self
    suche(frage, k)      -> list[Treffer] (absteigend nach Score)

Enthalten:
- TfidfRetriever       Wort-TF-IDF, optional mit CISTEM-Stemming
- ZeichenRetriever     TF-IDF auf Zeichen-n-Grammen (robust gegen Komposita)
- BM25Retriever        Okapi BM25 auf Wort-Tokens
- LsaRetriever         Latent Semantic Analysis (TF-IDF + SVD)
- HybridRetriever      Reciprocal Rank Fusion mehrerer Retriever
- DenseRetriever       Satz-Embeddings ueber sentence-transformers (optional,
                       in dieser Umgebung nicht getestet)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .chunking import Chunk
from .text import Analysator, falte


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


class TfidfRetriever(Retriever):
    def __init__(self, stemming: bool = True, sublinear_tf: bool = True):
        self.stemming = stemming
        self.sublinear_tf = sublinear_tf
        self.name = "tfidf_stem" if stemming else "tfidf"

    def _fit(self, texte):
        self.vec = TfidfVectorizer(analyzer=Analysator(stemming=self.stemming),
                                   sublinear_tf=self.sublinear_tf, norm="l2")
        self.matrix = self.vec.fit_transform(texte)

    def scores(self, frage):
        q = self.vec.transform([frage])
        return (self.matrix @ q.T).toarray().ravel()


class ZeichenRetriever(Retriever):
    def __init__(self, ngram: tuple[int, int] = (3, 5)):
        self.ngram = ngram
        self.name = "zeichen_ngramm"

    def _fit(self, texte):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram,
                                   preprocessor=falte, sublinear_tf=True, min_df=2)
        self.matrix = self.vec.fit_transform(texte)

    def scores(self, frage):
        q = self.vec.transform([frage])
        return (self.matrix @ q.T).toarray().ravel()


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


class LsaRetriever(Retriever):
    """Latent Semantic Analysis: TF-IDF (mit Stemming) -> abgeschnittene SVD -> Kosinus.

    Eine einfache dichte Repraesentation, die nur aus dem Korpus selbst gelernt wird
    und ohne zusaetzliche Pakete auskommt.
    """

    def __init__(self, komponenten: int = 200, seed: int = 0):
        self.komponenten = komponenten
        self.seed = seed
        self.name = "lsa"

    def _fit(self, texte):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        self._normalize = normalize
        self.vec = TfidfVectorizer(analyzer=Analysator(stemming=True), sublinear_tf=True, min_df=2)
        x = self.vec.fit_transform(texte)
        k = max(1, min(self.komponenten, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.matrix = normalize(self.svd.fit_transform(x))

    def scores(self, frage):
        q = self._normalize(self.svd.transform(self.vec.transform([frage])))
        return (self.matrix @ q[0]).astype(float)


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion (Cormack et al. 2009): score = sum 1 / (k_rrf + rang)."""

    def __init__(self, retriever: list[Retriever], k_rrf: int = 60, tiefe: int = 100):
        self.retriever = retriever
        self.k_rrf = k_rrf
        self.tiefe = tiefe
        self.name = "hybrid(" + "+".join(r.name for r in retriever) + ")"

    def fit(self, chunks):
        self.chunks = list(chunks)
        for r in self.retriever:
            r.fit(self.chunks)
        return self

    def scores(self, frage):
        gesamt = np.zeros(len(self.chunks))
        for r in self.retriever:
            for rang, t in enumerate(r.suche(frage, self.tiefe), start=1):
                if t.score > 0:  # Treffer ohne jede Uebereinstimmung nicht mitzaehlen
                    gesamt[t.index] += 1.0 / (self.k_rrf + rang)
        return gesamt


class DenseRetriever(Retriever):
    """Embedding-Retriever. Benoetigt das Paket sentence-transformers.

    Standardmodell: intfloat/multilingual-e5-small (erwartet die Praefixe
    "query: " und "passage: "). Nicht in CI und nicht in der Entwicklungs-
    umgebung getestet, da das Paket dort nicht installierbar war.
    """

    def __init__(self, modell: str = "intfloat/multilingual-e5-small",
                 frage_praefix: str = "query: ", text_praefix: str = "passage: "):
        self.modell_name = modell
        self.frage_praefix = frage_praefix
        self.text_praefix = text_praefix
        self.name = "dense"

    def _fit(self, texte):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - abhaengig von der Umgebung
            raise ImportError(
                "DenseRetriever benoetigt 'sentence-transformers' "
                "(pip install -e '.[dense]')") from e
        self.modell = SentenceTransformer(self.modell_name)
        self.matrix = self.modell.encode([self.text_praefix + t for t in texte],
                                         normalize_embeddings=True, show_progress_bar=False)

    def scores(self, frage):
        q = self.modell.encode([self.frage_praefix + frage], normalize_embeddings=True)
        return (self.matrix @ q[0]).astype(float)


def baue_retriever(name: str, **kw) -> Retriever:
    """Fabrik fuer die Konfigurationsdateien."""
    if name == "tfidf":
        return TfidfRetriever(stemming=False)
    if name == "tfidf_stem":
        return TfidfRetriever(stemming=True)
    if name == "zeichen_ngramm":
        return ZeichenRetriever()
    if name == "bm25":
        return BM25Retriever(**kw)
    if name == "lsa":
        return LsaRetriever(**kw)
    if name == "hybrid":
        teile = kw.get("teile", ["bm25", "zeichen_ngramm"])
        return HybridRetriever([baue_retriever(t) for t in teile])
    if name.startswith("hybrid:"):
        # z. B. "hybrid:bm25+dense"
        r = HybridRetriever([baue_retriever(t) for t in name.split(":", 1)[1].split("+")])
        r.name = name
        return r
    if name.startswith("umformuliert:"):
        # z. B. "umformuliert:bm25": Anfrage zusaetzlich von einem Sprachmodell umformulieren lassen
        from .anfrage import UmformulierRetriever
        from .antwort import OpenAIKompatibelBackend
        r = UmformulierRetriever(baue_retriever(name.split(":", 1)[1]), OpenAIKompatibelBackend())
        r.name = name
        return r
    if name == "dense":
        return DenseRetriever(**kw)
    raise ValueError(f"Unbekannter Retriever: {name}")
