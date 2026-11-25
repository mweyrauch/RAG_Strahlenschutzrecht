"""Textnormalisierung fuer deutsche Rechtstexte.

Ablauf: Kleinschreibung -> Umlaute falten (ae/oe/ue/ss) -> Tokenisierung
-> Stoppwoerter entfernen -> optional Stemming mit CISTEM
(Weissweiler & Fraser 2017, "Developing a Stemmer for German Based on a
Comparative Analysis of Publicly Available Stemmers").
"""

from __future__ import annotations

import re
from functools import lru_cache

_FALTUNG = str.maketrans({
    "\u00e4": "ae", "\u00f6": "oe", "\u00fc": "ue", "\u00df": "ss",
    "\u00c4": "ae", "\u00d6": "oe", "\u00dc": "ue",
})

_TOKEN = re.compile(r"[a-z0-9]+")

# Bewusst kleine Liste: Funktionswoerter und Frageformulierungen.
# Fachwoerter wie "nicht" bleiben drin, weil sie in Rechtstexten Bedeutung tragen koennen.
STOPPWOERTER = frozenset("""
aber alle allem allen aller alles als also am an ander andere anderem anderen
anderer anderes auch auf aus bei beim bin bis bist da damit dann das dass dein
deine dem den denen der deren des dessen die dies diese diesem diesen dieser
dieses doch dort du durch ein eine einem einen einer eines er es etwa euch
euer fuer gegen gibt hab habe haben hat hatte ich ihr ihre im in ins ist ja
jede jedem jeden jeder jedes kann koennen man mein meine mich mir mit muss
muessen nach noch nun nur ob oder ohne sein seine sich sie sind so soll sollen
ueber um und uns unser unter vom von vor waere war was welche welchem welchen
welcher welches wenn wer werden wie wieviel wird wo wurde zu zum zur
bitte genau eigentlich denn mal gilt gelten
""".split())


def falte(text: str) -> str:
    """Kleinschreibung und Umlautfaltung."""
    return text.lower().translate(_FALTUNG)


@lru_cache(maxsize=200_000)
def cistem(wort: str) -> str:
    """CISTEM-Stemmer (Variante ohne Gross/Kleinschreibungs-Information).

    Erwartet ein bereits kleingeschriebenes, gefaltetes Wort.
    """
    if len(wort) == 0:
        return wort
    w = wort
    w = re.sub(r"^ge(.{4,})", r"\1", w)
    w = w.replace("sch", "$").replace("ei", "%").replace("ie", "&")
    w = re.sub(r"(.)\1", r"\1*", w)
    while len(w) > 3:
        if len(w) > 5:
            neu = re.sub(r"e[mr]$", "", w)
            if neu != w:
                w = neu
                continue
            neu = re.sub(r"nd$", "", w)
            if neu != w:
                w = neu
                continue
        neu = re.sub(r"t$", "", w)
        if neu != w:
            w = neu
            continue
        neu = re.sub(r"[esn]$", "", w)
        if neu != w:
            w = neu
            continue
        break
    w = re.sub(r"(.)\*", r"\1\1", w)
    w = w.replace("&", "ie").replace("%", "ei").replace("$", "sch")
    return w


def tokens(text: str, stemming: bool = True, stoppwoerter: bool = True) -> list[str]:
    """Zerlegt Text in normalisierte Tokens."""
    roh = _TOKEN.findall(falte(text))
    if stoppwoerter:
        roh = [t for t in roh if t not in STOPPWOERTER]
    if stemming:
        roh = [cistem(t) for t in roh]
    return roh


class Analysator:
    """Aufrufbares Objekt fuer scikit-learn-Vektorisierer (picklebar)."""

    def __init__(self, stemming: bool = True, stoppwoerter: bool = True):
        self.stemming = stemming
        self.stoppwoerter = stoppwoerter

    def __call__(self, text: str) -> list[str]:
        return tokens(text, self.stemming, self.stoppwoerter)


_ABKUERZUNGEN = [
    "Abs.", "Nr.", "Art.", "bzw.", "z. B.", "vgl.", "Buchst.", "i. V. m.",
    "u. a.", "d. h.", "ggf.", "Anl.", "S.", "Bd.", "Ziff.", "lit.", "insb.",
]


def saetze(text: str) -> list[str]:
    """Grobe Satzzerlegung, die gaengige juristische Abkuerzungen schuetzt."""
    geschuetzt = text
    for abk in _ABKUERZUNGEN:
        geschuetzt = geschuetzt.replace(abk, abk.replace(".", "\u0000"))
    teile = re.split(r"(?<=[.;:!?])\s+(?=[A-Z\u00c4\u00d6\u00dc(])", geschuetzt)
    return [t.replace("\u0000", ".").strip() for t in teile if t.strip()]
