import numpy as np
import pandas as pd
import pytest

from strlrag.chunking import Chunk
from strlrag.evaluation import (auc, auc_bootstrap, bootstrap_differenz, bootstrap_ki,
                                kennzahlen, lade_fragen, rang_paragraph,
                                stichwort_getroffen, werte_retrieval_aus)
from strlrag.retrieval import Treffer, baue_retriever
from strlrag.chunking import zerlege


def _t(schluessel):
    return Treffer(0, 1.0, Chunk(schluessel, schluessel, None, "", "", ""))


def test_rang_zaehlt_vorschriften_einmal():
    treffer = [_t("A"), _t("A"), _t("B"), _t("C")]
    assert rang_paragraph(treffer, frozenset({"C"})) == 3
    assert rang_paragraph(treffer, frozenset({"X"})) is None


def test_kennzahlen():
    df = pd.DataFrame({"rang": [1, 2, None, 5], "item": list("abcd")})
    k = kennzahlen(df, ks=(1, 3, 5))
    assert k["recall@1"] == 0.25 and k["recall@3"] == 0.5 and k["recall@5"] == 0.75
    assert k["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)


def test_bootstrap_intervall_umschliesst_schaetzer():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"rang": rng.integers(1, 10, 40), "item": np.repeat(np.arange(20), 2)})
    p, lo, hi = bootstrap_ki(df, "recall@3", n=300)
    assert lo <= p <= hi


def test_gepaarte_differenz_null_bei_gleichen_varianten():
    df = pd.DataFrame({"frage": list("abcd"), "item": list("aabb"), "rang": [1, 3, None, 2]})
    d, lo, hi = bootstrap_differenz(df, df.copy(), "mrr", n=200)
    assert d == lo == hi == 0


def test_differenz_verlangt_gleiche_fragen():
    a = pd.DataFrame({"frage": ["a"], "item": ["a"], "rang": [1]})
    b = pd.DataFrame({"frage": ["b"], "item": ["b"], "rang": [1]})
    with pytest.raises(ValueError):
        bootstrap_differenz(a, b, "mrr", n=10)


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [1, 2]) == 0.5
    p, lo, hi = auc_bootstrap([3, 4, 5], [1, 2, 3], n=200)
    assert lo <= p <= hi


def test_stichwort():
    assert stichwort_getroffen("... von 20  Millisievert ...", ("20 Millisievert",))
    assert stichwort_getroffen("Ma\u00dfnahmen", ("massnahmen",))
    assert stichwort_getroffen("x", ()) is None


def test_lade_fragen_und_auswertung(tmp_path, korpus):
    pfad = tmp_path / "f.yaml"
    pfad.write_text(
        "fragen:\n"
        "  - id: q1\n"
        "    fach: Pr\u00fcfintervall f\u00fcr Messger\u00e4te\n"
        "    alltag: Wie oft muss ich mein Messger\u00e4t checken lassen?\n"
        "    soll: ['TestG \u00a7 2']\n"
        "fremd: ['Wie backe ich Brot?']\n", encoding="utf-8")
    fs = lade_fragen(pfad)
    assert [f.id for f in fs.fragen] == ["q1-fach", "q1-alltag"]
    assert fs.fremd == ["Wie backe ich Brot?"]
    r = baue_retriever("bm25").fit(zerlege(korpus))
    df = werte_retrieval_aus(r, fs.fragen)
    assert df.loc[0, "rang"] == 1


def test_bootstrap_mittelwert():
    from strlrag.evaluation import bootstrap_mittelwert
    p, lo, hi = bootstrap_mittelwert([1, 0, 1, 1], ["a", "a", "b", "c"], n=300)
    assert p == 0.75 and lo <= p <= hi


def test_muster_standard_und_eigen(tmp_path):
    pfad = tmp_path / "f.yaml"
    pfad.write_text(
        "fragen:\n"
        "  - {id: q1, fach: a, soll: [X], stichworte: [s1]}\n"
        "  - {id: q2, fach: b, soll: [X], stichworte: [s2], antwort_muster: [m1, m2]}\n",
        encoding="utf-8")
    f = lade_fragen(pfad).fragen
    assert f[0].muster == ("s1",) and f[1].muster == ("m1", "m2")
