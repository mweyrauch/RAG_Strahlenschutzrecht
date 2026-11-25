"""Zerlegung der Vorschriften in Abschnitte (Chunks) fuer das Retrieval.

Strategien:
- "paragraph": eine Vorschrift = ein Chunk
- "absatz":    ein Absatz = ein Chunk
- "fenster":   Wortfenster fester Laenge mit Ueberlappung, innerhalb einer Vorschrift

Der Kontext-Praefix ("ohne", "titel", "gliederung") bestimmt, welche Metadaten
vor den Text des Chunks gesetzt werden. Er wird mitindexiert, aber im Antworttext
nicht als Zitat ausgegeben.
"""

from __future__ import annotations

from dataclasses import dataclass

from .korpus import Vorschrift

STRATEGIEN = ("paragraph", "absatz", "fenster")
KONTEXTE = ("ohne", "titel", "gliederung")


@dataclass(frozen=True)
class Chunk:
    id: str
    schluessel: str        # Paragraphenebene, z. B. "StrlSchG Paragraph 78" (fuer die Bewertung)
    absatz: int | None     # 1-basiert, falls eindeutig
    titel: str
    praefix: str
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.praefix}\n{self.text}" if self.praefix else self.text

    @property
    def quelle(self) -> str:
        teile = [self.schluessel]
        if self.absatz is not None:
            teile.append(f"Abs. {self.absatz}")
        s = " ".join(teile)
        return f"{s} ({self.titel})" if self.titel else s


def _praefix(v: Vorschrift, kontext: str) -> str:
    if kontext == "ohne":
        return ""
    kopf = f"{v.schluessel} {v.titel}".strip()
    if kontext == "titel":
        return kopf
    if kontext == "gliederung":
        return " > ".join([*v.gliederung, kopf])
    raise ValueError(f"Unbekannter Kontext: {kontext}")


def zerlege(
    korpus: list[Vorschrift],
    strategie: str = "absatz",
    kontext: str = "titel",
    fenster_woerter: int = 120,
    ueberlappung: int = 40,
) -> list[Chunk]:
    if strategie not in STRATEGIEN:
        raise ValueError(f"Unbekannte Strategie: {strategie}")
    if kontext not in KONTEXTE:
        raise ValueError(f"Unbekannter Kontext: {kontext}")
    if strategie == "fenster" and not (0 <= ueberlappung < fenster_woerter):
        raise ValueError("Ueberlappung muss kleiner als die Fensterlaenge sein")

    chunks: list[Chunk] = []
    for v in korpus:
        p = _praefix(v, kontext)
        if strategie == "paragraph":
            chunks.append(Chunk(f"{v.schluessel}", v.schluessel, None, v.titel, p, v.text))
        elif strategie == "absatz":
            for i, a in enumerate(v.absaetze, start=1):
                nr = i if len(v.absaetze) > 1 else None
                chunks.append(Chunk(f"{v.schluessel}#{i}", v.schluessel, nr, v.titel, p, a))
        else:
            woerter = v.text.split()
            schritt = fenster_woerter - ueberlappung
            start = 0
            j = 0
            while True:
                stueck = woerter[start:start + fenster_woerter]
                if not stueck:
                    break
                chunks.append(Chunk(f"{v.schluessel}@{j}", v.schluessel, None, v.titel, p, " ".join(stueck)))
                j += 1
                if start + fenster_woerter >= len(woerter):
                    break
                start += schritt
    return chunks
