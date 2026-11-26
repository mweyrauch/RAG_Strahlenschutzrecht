"""Pfade und Laden des Standardkorpus (StrlSchG + StrlSchV)."""

from __future__ import annotations

from pathlib import Path

from .korpus import Vorschrift, lese_korpus

WURZEL = Path(__file__).resolve().parents[2]
ROH = WURZEL / "data" / "roh"
FRAGEN = WURZEL / "data" / "eval" / "fragen.yaml"

QUELLEN = {
    "strlschg_xml.zip": "https://www.gesetze-im-internet.de/strlschg/xml.zip",
    "strlschv_xml.zip": "https://www.gesetze-im-internet.de/strlschv_2018/xml.zip",
}


def standardkorpus(roh: Path = ROH) -> list[Vorschrift]:
    pfade = [roh / name for name in QUELLEN]
    fehlend = [p.name for p in pfade if not p.exists()]
    if fehlend:
        raise FileNotFoundError(
            f"Rohdaten fehlen in {roh}: {', '.join(fehlend)}. "
            "Bitte zuerst 'python scripts/lade_korpus.py' ausfuehren.")
    return lese_korpus(pfade)
