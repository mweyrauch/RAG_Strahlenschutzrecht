import io
import zipfile

import pytest

from strlrag.korpus import lese_gesetz, lese_korpus, normiere_bezeichnung

from conftest import FIXTURES


def test_liest_nur_einzelvorschriften(korpus):
    # Mantel, Inhaltsuebersicht und weggefallener Paragraph fehlen
    assert [v.bezeichnung for v in korpus] == ["\u00a7 1", "\u00a7 2", "\u00a7 4", "Anlage 1"]


def test_absaetze_und_listen(korpus):
    p2 = korpus[1]
    assert len(p2.absaetze) == 3
    assert "Sichtpr\u00fcfung des Geh\u00e4uses" in p2.absaetze[1]
    assert p2.titel == "Pr\u00fcfintervall"


def test_anlage_ohne_p_liest_tabelle(korpus):
    anlage = korpus[-1]
    assert "Temperaturmessung" in anlage.text and "2 Kelvin" in anlage.text


def test_zip_wird_entpackt(tmp_path):
    puffer = io.BytesIO()
    with zipfile.ZipFile(puffer, "w") as zf:
        zf.write(FIXTURES / "testg.xml", "BJNR000000000.xml")
    pfad = tmp_path / "testg.zip"
    pfad.write_bytes(puffer.getvalue())
    assert len(lese_gesetz(pfad)) == 4


def test_doppelte_schluessel_sind_ein_fehler():
    with pytest.raises(ValueError, match="Doppelte"):
        lese_korpus([FIXTURES / "testg.xml", FIXTURES / "testg.xml"])

def test_normiere_bezeichnung():
    assert normiere_bezeichnung("\u00a7  12") == "\u00a7 12"
    assert normiere_bezeichnung("\u00a712a") == "\u00a7 12a"
