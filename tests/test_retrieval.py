import numpy as np
import pytest

from strlrag.chunking import zerlege
from strlrag.retrieval import BM25Retriever, baue_retriever


@pytest.fixture
def chunks(korpus):
    return zerlege(korpus)


def test_bm25_findet_offensichtliche_stelle(chunks):
    r = baue_retriever("bm25").fit(chunks)
    t = r.suche("Wie oft muss eine sachkundige Person pr\u00fcfen?", k=3)
    assert t[0].chunk.schluessel == "TestG \u00a7 2"
    assert len(t) == 3
    assert all(t[i].score >= t[i + 1].score for i in range(len(t) - 1))


def test_stemming_hilft_bei_flexion(chunks):
    # "Bestellungen" steht so nicht im Text, nur "Bestellung"/"bestellen"
    mit = baue_retriever("bm25").fit(chunks).suche("Bestellungen anzeigen", 1)[0]
    ohne = BM25Retriever(stemming=False).fit(chunks).suche("Bestellungen anzeigen", 1)[0]
    assert mit.chunk.schluessel == "TestG \u00a7 4"
    assert ohne.score < mit.score


def test_bm25_gegen_handrechnung():
    from strlrag.chunking import Chunk
    docs = ["apfel birne", "apfel apfel kirsche", "kirsche"]
    cs = [Chunk(str(i), str(i), None, "", "", d) for i, d in enumerate(docs)]
    r = BM25Retriever(k1=1.2, b=0.75, stemming=False).fit(cs)
    n, avgdl = 3, 2.0  # Laengen 2, 3, 1
    idf = np.log((n - 2 + 0.5) / (2 + 0.5) + 1)  # "apfel" in 2 Dokumenten
    erwartet = [idf * 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / avgdl)),
                idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / avgdl)),
                0.0]
    np.testing.assert_allclose(r.scores("apfel"), erwartet)


def test_mehr_k_als_chunks(chunks):
    r = baue_retriever("bm25").fit(chunks)
    assert len(r.suche("Messger\u00e4t", k=1000)) == len(chunks)


def test_unbekannter_retriever():
    with pytest.raises(ValueError):
        baue_retriever("gibtsnicht")


def test_umformulierung_hilft_bei_fremdem_vokabular(chunks):
    from strlrag.anfrage import UmformulierRetriever
    from strlrag.antwort import MockBackend
    frage = "Wer kuemmert sich offiziell um die Geraete?"
    basis = baue_retriever("bm25")
    mock = MockBackend("Wer ist als beauftragte Person f\u00fcr die Ger\u00e4tesicherheit zu bestellen?")
    r = UmformulierRetriever(basis, mock).fit(chunks)
    assert r.suche(frage, 1)[0].chunk.schluessel == "TestG \u00a7 4"
    assert mock.letzte_nachrichten[1]["content"] == frage
    r.suche(frage, 1)
    assert list(r.cache) == [frage]  # zweiter Aufruf aus dem Cache


class _FakeModell:
    """Ersetzt SentenceTransformer: Buchstabenhaeufigkeiten als 'Embedding'."""

    aufrufe = 0

    def __init__(self, name):
        self.name = name

    def encode(self, texte, normalize_embeddings=True, **kw):
        _FakeModell.aufrufe += 1
        m = np.zeros((len(texte), 26))
        for i, t in enumerate(texte):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    m[i, ord(ch) - 97] += 1
        if normalize_embeddings:
            m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        return m


@pytest.fixture
def fake_st(monkeypatch):
    import sys
    import types
    modul = types.ModuleType("sentence_transformers")
    modul.SentenceTransformer = _FakeModell
    monkeypatch.setitem(sys.modules, "sentence_transformers", modul)
    _FakeModell.aufrufe = 0
    return _FakeModell


def test_dense_mit_cache(chunks, fake_st, tmp_path):
    r = baue_retriever("dense", cache_dir=tmp_path).fit(chunks)
    assert r.frage_praefix == "query: " and r.text_praefix == "passage: "
    t = r.suche("Temperaturmessung Kelvin", 3)
    assert len(t) == 3 and t[0].score >= t[1].score
    assert len(list(tmp_path.glob("emb_*.npy"))) == 1
    vorher = fake_st.aufrufe
    r2 = baue_retriever("dense", cache_dir=tmp_path).fit(chunks)  # Korpus aus dem Cache
    assert fake_st.aufrufe == vorher
    np.testing.assert_allclose(r2.matrix, r.matrix)


def test_dense_modellname_und_praefix(fake_st, tmp_path):
    r = baue_retriever("dense:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       cache_dir=tmp_path)
    assert r.name.startswith("dense:") and r.frage_praefix == "" and r.text_praefix == ""
