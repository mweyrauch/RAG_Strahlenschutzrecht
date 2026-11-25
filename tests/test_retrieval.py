import numpy as np
import pytest

from strlrag.chunking import zerlege
from strlrag.retrieval import BM25Retriever, baue_retriever


@pytest.fixture
def chunks(korpus):
    return zerlege(korpus, "absatz", "titel")


@pytest.mark.parametrize("name", ["tfidf", "tfidf_stem", "zeichen_ngramm", "bm25", "hybrid"])
def test_findet_offensichtliche_stelle(chunks, name):
    r = baue_retriever(name).fit(chunks)
    t = r.suche("Wie oft muss eine sachkundige Person pr\u00fcfen?", k=3)
    assert t[0].chunk.schluessel == "TestG \u00a7 2"
    assert len(t) == 3
    assert all(t[i].score >= t[i + 1].score for i in range(len(t) - 1))


def test_stemming_hilft_bei_flexion(chunks):
    # "Bestellungen" steht so nicht im Text, nur "Bestellung"/"bestellen"
    mit = baue_retriever("tfidf_stem").fit(chunks).suche("Bestellungen anzeigen", 1)[0]
    assert mit.chunk.schluessel == "TestG \u00a7 4"


def test_zeichen_ngramme_finden_kompositum_teile(chunks):
    t = baue_retriever("zeichen_ngramm").fit(chunks).suche("Temperatur", 1)[0]
    assert t.chunk.schluessel == "TestG Anlage 1"


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


def test_lsa_liefert_scores(chunks):
    r = baue_retriever("lsa", komponenten=5).fit(chunks)
    t = r.suche("sachkundige Person pr\u00fcfen", 3)
    assert len(t) == 3 and -1.0 - 1e-9 <= t[0].score <= 1.0 + 1e-9


def test_hybrid_mit_eigenen_teilen(chunks):
    r = baue_retriever("hybrid:bm25+tfidf").fit(chunks)
    assert r.name == "hybrid:bm25+tfidf"
    assert r.suche("sachkundige Person", 1)[0].chunk.schluessel == "TestG \u00a7 2"


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
