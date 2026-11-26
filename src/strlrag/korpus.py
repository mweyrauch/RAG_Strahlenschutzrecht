"""Einlesen der XML-Fassung von gesetze-im-internet.de.

Jede Datei enthaelt ein <dokumente>-Element mit einer Folge von <norm>-Elementen.
Eine Norm ist entweder der Mantel (Langtitel), eine Gliederungseinheit
(Teil, Kapitel, Abschnitt; wird uebersprungen) oder eine Einzelvorschrift mit <enbez>
(z. B. Paragraph 78).
Absaetze stehen ueblicherweise in je einem <P>-Element innerhalb von <Content>.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class Vorschrift:
    gesetz: str            # z. B. "StrlSchG"
    bezeichnung: str       # z. B. Paragraph 78 oder "Anlage 4"
    titel: str             # z. B. "Grenzwerte fuer beruflich exponierte Personen"
    absaetze: list[str] = field(default_factory=list)

    @property
    def schluessel(self) -> str:
        """Eindeutiger Bezeichner auf Paragraphenebene, z. B. StrlSchG Paragraph 78."""
        return f"{self.gesetz} {self.bezeichnung}"

    @property
    def text(self) -> str:
        return "\n".join(self.absaetze)


_LEER = re.compile(r"\s+")


def _bereinige(text: str) -> str:
    return _LEER.sub(" ", text.replace("\u00a0", " ")).strip()


def _elementtext(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return _bereinige(" ".join(el.itertext()))


def normiere_bezeichnung(bez: str) -> str:
    """Vereinheitlicht "\u00a7  78" -> "\u00a7 78"."""
    bez = _bereinige(bez)
    bez = re.sub("^(\u00a7\u00a7?)\\s*", r"\1 ", bez)
    return bez


def _xml_aus_quelle(quelle: str | Path | bytes) -> bytes:
    if isinstance(quelle, bytes):
        daten = quelle
    else:
        daten = Path(quelle).read_bytes()
    if daten[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(daten)) as zf:
            namen = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            if not namen:
                raise ValueError("ZIP-Archiv enthaelt keine XML-Datei")
            return zf.read(namen[0])
    return daten


def lese_gesetz(quelle: str | Path | bytes) -> list[Vorschrift]:
    """Liest ein Gesetz (XML oder ZIP mit XML) und gibt die Einzelvorschriften zurueck.

    Weggelassene oder aufgehobene Vorschriften ("(weggefallen)") werden uebersprungen.
    """
    wurzel = ET.fromstring(_xml_aus_quelle(quelle))
    ergebnis: list[Vorschrift] = []
    # Die amtliche Abkuerzung (z. B. "StrlSchV" statt "StrlSchV 2018") steht nur im Mantel.
    amtlich = ""

    for norm in wurzel.iter("norm"):
        meta = norm.find("metadaten")
        if meta is None:
            continue
        amtlich = amtlich or _elementtext(meta.find("amtabk"))
        gesetz = amtlich or _elementtext(meta.find("jurabk"))

        enbez = _elementtext(meta.find("enbez"))
        if not enbez:
            continue
        bezeichnung = normiere_bezeichnung(enbez)
        if bezeichnung.lower() in {"eingangsformel", "inhalts\u00fcbersicht", "inhaltsuebersicht", "schlussformel"}:
            continue
        titel = _elementtext(meta.find("titel"))

        content = norm.find("textdaten/text/Content")
        absaetze: list[str] = []
        if content is not None:
            ps = content.findall("P")
            if ps:
                absaetze = [_elementtext(p) for p in ps]
            else:
                absaetze = [_elementtext(content)]
        absaetze = [a for a in absaetze if a]
        if not absaetze:
            continue
        if len(absaetze) == 1 and absaetze[0].strip("() .-").lower() in {"weggefallen", "-"}:
            continue

        ergebnis.append(Vorschrift(
            gesetz=gesetz,
            bezeichnung=bezeichnung,
            titel=titel,
            absaetze=absaetze,
        ))
    return ergebnis


def lese_korpus(pfade: list[str | Path]) -> list[Vorschrift]:
    korpus: list[Vorschrift] = []
    for p in pfade:
        korpus.extend(lese_gesetz(p))
    schluessel = [v.schluessel for v in korpus]
    doppelt = sorted({s for s in schluessel if schluessel.count(s) > 1})
    if doppelt:
        # Im aktuellen Korpus kommt das nicht vor; die Bewertung setzt eindeutige Schluessel voraus.
        raise ValueError(f"Doppelte Vorschriften im Korpus: {', '.join(doppelt)}")
    return korpus
