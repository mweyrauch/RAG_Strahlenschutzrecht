"""Stellt eine Frage an den Korpus und gibt Antwort und Quellen aus.

Beispiele:
  python scripts/frage.py "Wie oft ist die Unterweisung zu wiederholen?"
  python scripts/frage.py --backend llm "Welche Grenzwerte gelten fuer Auszubildende?"
  python scripts/frage.py --nur-quellen -k 10 "Radon im Keller"
  python scripts/frage.py --retriever bm25 "Wie oft ist die Unterweisung zu wiederholen?"

--backend llm nutzt einen Chat-Server mit OpenAI-kompatibler Schnittstelle
(Standard: Ollama unter http://localhost:11434/v1; siehe LLM_BASE_URL, LLM_MODEL, LLM_API_KEY).
Keine Rechtsberatung.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strlrag.antwort import ExtraktivBackend, OpenAIKompatibelBackend, beantworte  # noqa: E402
from strlrag.chunking import zerlege  # noqa: E402
from strlrag.daten import standardkorpus  # noqa: E402
from strlrag.retrieval import baue_retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frage")
    ap.add_argument("--retriever", default="dense",
                    help="dense (e5-large), bm25 oder dense:<Modellname>")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--backend", choices=["extraktiv", "llm"], default="extraktiv")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Enthaltung, wenn der beste Treffer darunter liegt (skalenabhaengig je Retriever)")
    ap.add_argument("--nur-quellen", action="store_true")
    args = ap.parse_args()

    ret = baue_retriever(args.retriever).fit(zerlege(standardkorpus()))
    if args.nur_quellen:
        for i, t in enumerate(ret.suche(args.frage, args.k), start=1):
            print(f"[{i}] {t.score:.3f}  {t.chunk.quelle}\n    {t.chunk.text[:200]}")
        return
    backend = ExtraktivBackend() if args.backend == "extraktiv" else OpenAIKompatibelBackend()
    a = beantworte(args.frage, ret, backend, k=args.k, min_score=args.min_score)
    print(a.formatiert())
    if a.ungueltige_zitate:
        print(f"\nWarnung: Zitate ohne passende Quelle: {a.ungueltige_zitate}")


if __name__ == "__main__":
    main()
