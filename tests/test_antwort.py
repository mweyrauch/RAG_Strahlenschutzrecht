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


def test_enthaltung_mit_angehaengter_begruendung(retriever):
    a = beantworte("Messger\u00e4te", retriever,
                   MockBackend(KEINE_GRUNDLAGE + " Die Quellen [1] behandeln etwas anderes."), k=1)
    assert a.enthalten


def test_lange_quellen_werden_im_prompt_gekuerzt(retriever):
    from strlrag.antwort import kontext_block
    t = retriever.suche("Messger\u00e4te pr\u00fcfen", 1)
    voll = kontext_block(t, max_woerter=None)
    kurz = kontext_block(t, max_woerter=3)
    assert kurz.endswith("[...]") and len(kurz) < len(voll)


class _Antwort:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


def test_llm_backend_mit_cache(monkeypatch, tmp_path, retriever):
    import requests
    from strlrag.antwort import OpenAIKompatibelBackend
    aufrufe = []

    def fake_post(url, json, headers, timeout):
        aufrufe.append((url, json["model"]))
        return _Antwort(" Antwort [1]. ")

    monkeypatch.setattr(requests, "post", fake_post)
    b = OpenAIKompatibelBackend(base_url="http://x/v1", modell="m", cache_dir=tmp_path)
    a1 = beantworte("Messger\u00e4te pr\u00fcfen", retriever, b, k=2)
    a2 = beantworte("Messger\u00e4te pr\u00fcfen", retriever, b, k=2)
    assert a1.text == a2.text == "Antwort [1]."
    assert aufrufe == [("http://x/v1/chat/completions", "m")] and b.anfragen == 1
    # anderes Modell -> eigener Cache-Eintrag
    b2 = OpenAIKompatibelBackend(base_url="http://x/v1", modell="m2", cache_dir=tmp_path)
    beantworte("Messger\u00e4te pr\u00fcfen", retriever, b2, k=2)
    assert len(aufrufe) == 2
