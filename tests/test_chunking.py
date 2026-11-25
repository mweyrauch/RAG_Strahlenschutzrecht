from strlrag.chunking import zerlege


def test_ein_chunk_je_absatz(korpus):
    c = zerlege(korpus)
    assert len(c) == sum(len(v.absaetze) for v in korpus)


def test_absatz_nummern_und_quelle(korpus):
    c = zerlege(korpus)
    p2 = [x for x in c if x.schluessel == "TestG \u00a7 2"]
    assert [x.absatz for x in p2] == [1, 2, 3]
    assert "Abs. 1" in p2[0].quelle
    # einziger Absatz bekommt keine Nummer
    p4 = [x for x in c if x.schluessel == "TestG \u00a7 4"]
    assert p4[0].absatz is None and "Abs." not in p4[0].quelle


def test_titel_praefix(korpus):
    mit = zerlege(korpus)[2]
    ohne = zerlege(korpus, mit_titel=False)[2]
    assert mit.praefix == "TestG \u00a7 2 Pr\u00fcfintervall"
    assert mit.index_text.startswith(mit.praefix)
    assert ohne.praefix == "" and ohne.index_text == ohne.text
