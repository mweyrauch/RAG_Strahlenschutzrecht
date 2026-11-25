"""Zerlegung der Vorschriften in Abschnitte (Chunks) fuer das Retrieval.

Ein Absatz = ein Chunk. Absaetze sind die natuerliche Einheit, auf die Rechtstexte
verweisen ("Paragraph 78 Abs. 4"), und erlauben praezise Quellenangaben.
Vor den Text wird der Titel der Vorschrift gesetzt (z. B. "StrlSchG Paragraph 78
Grenzwerte fuer beruflich exponierte Personen"), weil einzelne Absaetze oft nicht
sagen, worum es geht. Der Praefix wird mitindexiert, aber nicht zitiert.
"""

from __future__ import annotations

from dataclasses import dataclass

from .korpus import Vorschrift


@dataclass(frozen=True)
class Chunk:
    id: str
    schluessel: str        # Paragraphenebene, z. B. "StrlSchG Paragraph 78" (fuer die Bewertung)
    absatz: int | None     # 1-basiert; None, wenn die Vorschrift nur einen Absatz hat
    titel: str
    praefix: str
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.praefix}\n{self.text}" if self.praefix else self.text

    @property
    def quelle(self) -> str:
        s = self.schluessel if self.absatz is None else f"{self.schluessel} Abs. {self.absatz}"
        return f"{s} ({self.titel})" if self.titel else s


def zerlege(korpus: list[Vorschrift], mit_titel: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    for v in korpus:
        praefix = f"{v.schluessel} {v.titel}".strip() if mit_titel else ""
        for i, a in enumerate(v.absaetze, start=1):
            nr = i if len(v.absaetze) > 1 else None
            chunks.append(Chunk(f"{v.schluessel}#{i}", v.schluessel, nr, v.titel, praefix, a))
    return chunks
