import pytest

from strlrag.chunking import zerlege


def test_paragraph(korpus):
    c = zerlege(korpus, "paragraph", "ohne")
    assert len(c) == len(korpus)
    assert c[0].praefix == ""


def test_absatz_nummern(korpus):
    c = zerlege(korpus, "absatz", "titel")
    p2 = [x for x in c if x.schluessel == "TestG \u00a7 2"]
    assert [x.absatz for x in p2] == [1, 2, 3]
    # einziger Absatz bekommt keine Nummer
    p4 = [x for x in c if x.schluessel == "TestG \u00a7 4"]
    assert p4[0].absatz is None
    assert p2[0].praefix == "TestG \u00a7 2 Pr\u00fcfintervall"
    assert "Abs. 1" in p2[0].quelle


def test_gliederung_im_praefix(korpus):
    c = zerlege(korpus, "absatz", "gliederung")
    assert c[2].praefix.startswith("Teil 2 Pflichten > Abschnitt 1")


def test_fenster_ueberdeckt_alle_woerter(korpus):
    c = zerlege(korpus, "fenster", "ohne", fenster_woerter=8, ueberlappung=3)
    for v in korpus:
        woerter = v.text.split()
        stuecke = [x.text.split() for x in c if x.schluessel == v.schluessel]
        assert stuecke[0][:8] == woerter[:8]
        assert stuecke[-1][-1] == woerter[-1]
        assert all(len(s) <= 8 for s in stuecke)


def test_ungueltige_parameter(korpus):
    with pytest.raises(ValueError):
        zerlege(korpus, "satz")
    with pytest.raises(ValueError):
        zerlege(korpus, "fenster", fenster_woerter=10, ueberlappung=10)
