"""Prueft das Fragenset gegen den Korpus.

- Gibt es jede Soll-Vorschrift im Korpus?
- Steht jedes Stichwort woertlich (nach Umlautfaltung) in mindestens einer Soll-Vorschrift?
- Kommen Item-IDs doppelt vor?
Beendet sich mit Fehlercode 1, wenn etwas nicht stimmt.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strlrag.daten import FRAGEN, standardkorpus  # noqa: E402
from strlrag.evaluation import stichwort_getroffen  # noqa: E402


def main():
    korpus = {v.schluessel: v for v in standardkorpus()}
    daten = yaml.safe_load(FRAGEN.read_text(encoding="utf-8"))
    fehler = []
    ids = [e["id"] for e in daten["fragen"]]
    for i in {x for x in ids if ids.count(x) > 1}:
        fehler.append(f"doppelte ID {i}")
    for e in daten["fragen"]:
        for s in e["soll"]:
            if s not in korpus:
                fehler.append(f"{e['id']}: Soll-Vorschrift '{s}' nicht im Korpus")
        texte = " ".join(korpus[s].text for s in e["soll"] if s in korpus)
        for w in e.get("stichworte", []):
            if not stichwort_getroffen(texte, (w,)):
                fehler.append(f"{e['id']}: Stichwort '{w}' steht in keiner Soll-Vorschrift")
    print(f"{len(daten['fragen'])} Items, {len(daten.get('fremd', []))} Fremdfragen geprueft")
    for f in fehler:
        print("FEHLER", f)
    sys.exit(1 if fehler else 0)


if __name__ == "__main__":
    main()
