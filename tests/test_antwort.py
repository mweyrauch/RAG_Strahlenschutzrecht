import pytest

from strlrag.antwort import (KEINE_GRUNDLAGE, Antwort, ExtraktivBackend, MockBackend,
                             baue_nachrichten, beantworte)
from strlrag.chunking import zerlege
from strlrag.retrieval import baue_retriever


@pytest.fixture
def retriever(korpus):
    return baue_retriever("bm25").fit(zerlege(korpus))


def test_extraktiv_zitiert_richtige_quelle(retriever):
    a = beantworte("Wie lange ist das Ergebnis aufzubewahren?", retriever,
                   ExtraktivBackend(n_saetze=1), k=3)
    assert "f\u00fcnf Jahre" in a.text
    nr = a.zitierte_nummern[0]
    assert a.quellen[nr - 1].chunk.schluessel == "TestG \u00a7 2"
    assert a.ungueltige_zitate == []


def test_extraktiv_ohne_ueberlappung_enthaelt_sich(retriever):
    a = beantworte("Brot backen", retriever, ExtraktivBackend(), k=2)
    assert a.enthalten


def test_enthaltung_unter_schwelle(retriever):
    a = beantworte("Wie backe ich Brot?", retriever, ExtraktivBackend(), k=3, min_score=0.5)
    assert a.enthalten and a.text == KEINE_GRUNDLAGE


def test_ungueltige_zitate_erkannt(retriever):
    a = beantworte("Messger\u00e4te pr\u00fcfen", retriever, MockBackend("Siehe [1] und [9]."), k=2)
    assert a.ungueltige_zitate == [9]


def test_prompt_enthaelt_nummerierte_quellen(retriever):
    b = MockBackend()
    beantworte("Messger\u00e4te pr\u00fcfen", retriever, b, k=2)
    inhalt = b.letzte_nachrichten[1]["content"]
    assert "[1] TestG" in inhalt and "[2] TestG" in inhalt
    assert b.letzte_nachrichten[0]["role"] == "system"


def test_formatiert_listet_quellen(retriever):
    a = beantworte("Messger\u00e4te pr\u00fcfen", retriever, MockBackend(), k=2)
    assert a.formatiert().splitlines()[-1].startswith("[2] ")


def test_nachrichten_ohne_treffer():
    assert "Frage: x" in baue_nachrichten("x", [])[1]["content"]


def test_mock_text_gleich_keine_grundlage_ist_enthaltung(retriever):
    a = beantworte("Messger\u00e4te", retriever, MockBackend(KEINE_GRUNDLAGE), k=1)
    assert isinstance(a, Antwort) and a.enthalten
