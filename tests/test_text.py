from strlrag.text import cistem, falte, saetze, tokens


def test_falte_umlaute():
    assert falte("\u00dcberpr\u00fcfung Ma\u00dfnahme") == "ueberpruefung massnahme"


def test_cistem_bekannte_faelle():
    # Flexionsformen fallen auf denselben Stamm
    assert cistem("grenzwerte") == cistem("grenzwert")
    assert cistem("personen") == cistem("person")
    assert cistem("strahlenschutzbeauftragter") == cistem("strahlenschutzbeauftragte")


def test_cistem_kurze_woerter_bleiben():
    assert cistem("dos") == "dos"
    assert cistem("") == ""


def test_tokens_entfernt_stoppwoerter():
    t = tokens("Welche Grenzwerte gelten f\u00fcr die Bev\u00f6lkerung?", stemming=False)
    assert t == ["grenzwerte", "bevoelkerung"]


def test_saetze_schuetzt_abkuerzungen():
    s = saetze("Die Pr\u00fcfung nach Abs. 3 Nr. 1 ist Pflicht. Das Ergebnis wird notiert.")
    assert len(s) == 2
    assert s[0].endswith("Pflicht.")
