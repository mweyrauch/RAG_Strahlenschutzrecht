"""Laedt die XML-Fassungen von StrlSchG und StrlSchV von gesetze-im-internet.de nach data/roh/.

Die Texte sind amtliche Werke (Paragraph 5 UrhG). Sie werden trotzdem nicht im Repository
versioniert, damit immer die aktuelle Fassung verwendet wird.
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strlrag.daten import QUELLEN, ROH  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ziel", type=Path, default=ROH)
    ap.add_argument("--neu", action="store_true", help="vorhandene Dateien ueberschreiben")
    args = ap.parse_args()
    args.ziel.mkdir(parents=True, exist_ok=True)
    for name, url in QUELLEN.items():
        pfad = args.ziel / name
        if pfad.exists() and not args.neu:
            print(f"vorhanden: {pfad}")
            continue
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        pfad.write_bytes(r.content)
        print(f"geladen:   {pfad} ({len(r.content) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
