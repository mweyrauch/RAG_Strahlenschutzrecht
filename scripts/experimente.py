"""Wertet die Retriever aus configs/experimente.yaml auf dem Fragenset aus.

Ausgaben (in --aus, Standard ergebnisse/):
  kennzahlen.csv   Recall@k, MRR und Antwort-Kennzahlen je Retriever und Frageform,
                   mit 95-%-Bootstrap-Intervallen
  raenge.csv       Rang der Soll-Vorschrift und bester Treffer je Frage (fuer die Fehleranalyse)
  differenzen.csv  gepaarte Differenzen zur Referenz (nur bei mehr als einem Retriever)
  enthaltung.csv   Trennschaerfe des Top-Scores zwischen Korpus- und Fremdfragen (AUC)
  antworten.csv    extraktive Antworten je Frage
  docs/abbildungen/*.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

WURZEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WURZEL / "src"))
from strlrag.antwort import ExtraktivBackend, beantworte  # noqa: E402
from strlrag.chunking import zerlege  # noqa: E402
from strlrag.daten import FRAGEN, standardkorpus  # noqa: E402
from strlrag.evaluation import (auc_bootstrap, bootstrap_differenz, bootstrap_ki,  # noqa: E402
                                lade_fragen, stichwort_getroffen, werte_retrieval_aus)
from strlrag.retrieval import baue_retriever  # noqa: E402
import abbildungen  # noqa: E402


def antworten(ret, fragen, k, n_saetze):
    backend = ExtraktivBackend(n_saetze)
    zeilen = []
    for f in fragen:
        a = beantworte(f.text, ret, backend, k=k)
        kontext_text = " ".join(t.chunk.text for t in a.quellen)
        zeilen.append({"frage": f.id, "item": f.item, "stil": f.stil,
                       "stichwort_im_kontext": stichwort_getroffen(kontext_text, f.stichworte),
                       "stichwort_in_antwort": stichwort_getroffen(a.text, f.stichworte),
                       "ungueltige_zitate": len(a.ungueltige_zitate),
                       "antwort": a.text})
    return pd.DataFrame(zeilen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=WURZEL / "configs" / "experimente.yaml")
    ap.add_argument("--aus", type=Path, default=WURZEL / "ergebnisse")
    ap.add_argument("--abbildungen", type=Path, default=WURZEL / "docs" / "abbildungen")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.aus.mkdir(parents=True, exist_ok=True)
    args.abbildungen.mkdir(parents=True, exist_ok=True)
    n_boot, seed, ks = cfg["bootstrap"], cfg["seed"], cfg["ks"]

    korpus = standardkorpus()
    chunks = zerlege(korpus)
    fs = lade_fragen(FRAGEN)
    print(f"Korpus: {len(korpus)} Vorschriften, {len(chunks)} Absaetze; "
          f"Fragen: {len(fs.fragen)} + {len(fs.fremd)} fremd")

    raenge, ants, kennz, enth, scores = {}, {}, [], [], {}
    for name in cfg["retriever"]:
        ret = baue_retriever(name).fit(chunks)
        df = werte_retrieval_aus(ret, fs.fragen, cfg["tiefe"])
        an = antworten(ret, fs.fragen, cfg["antwort"]["k"], cfg["antwort"]["n_saetze"])
        raenge[name], ants[name] = df, an

        for stil, teil in [("gesamt", df), *df.groupby("stil")]:
            zeile = {"retriever": name, "stil": stil, "n_fragen": len(teil)}
            for m in [f"recall@{k}" for k in ks] + ["mrr"]:
                zeile[m], zeile[f"{m}_ki_unten"], zeile[f"{m}_ki_oben"] = bootstrap_ki(teil, m, n_boot, seed)
            a = an[an.frage.isin(teil.frage)]
            zeile["stichwort_im_kontext"] = a.stichwort_im_kontext.mean()
            zeile["stichwort_in_antwort"] = a.stichwort_in_antwort.mean()
            kennz.append(zeile)

        # Enthaltung: Score-Schwellen ergeben bei RRF-Scores (nur Raenge) keinen Sinn
        if not name.startswith("umformuliert"):
            pos = df["top_score"].to_numpy()
            neg = np.array([ret.suche(q, 1)[0].score for q in fs.fremd])
            a, lo, hi = auc_bootstrap(pos, neg, n_boot, seed)
            schwelle = float(np.quantile(pos, 0.05))
            enth.append({"retriever": name, "auc": a, "ki_unten": lo, "ki_oben": hi,
                         "schwelle_q05": schwelle,
                         "fremd_ueber_schwelle": int((neg >= schwelle).sum()), "n_fremd": len(neg)})
            scores[name] = (pos, neg)
        print(f"  {name:22s} recall@5 fach={kennz[-1]['recall@5']:.2f} "
              f"alltag={kennz[-2]['recall@5']:.2f}")

    pd.DataFrame(kennz).to_csv(args.aus / "kennzahlen.csv", index=False)
    pd.concat([d.assign(retriever=n) for n, d in raenge.items()]).to_csv(args.aus / "raenge.csv", index=False)
    pd.concat([d.assign(retriever=n) for n, d in ants.items()]).to_csv(args.aus / "antworten.csv", index=False)
    pd.DataFrame(enth).to_csv(args.aus / "enthaltung.csv", index=False)

    ref = cfg["referenz"]
    andere = [n for n in cfg["retriever"] if n != ref]
    if andere:
        diffs = []
        for n in andere:
            for stil in ["gesamt", "fach", "alltag"]:
                a = raenge[n] if stil == "gesamt" else raenge[n][raenge[n].stil == stil]
                b = raenge[ref] if stil == "gesamt" else raenge[ref][raenge[ref].stil == stil]
                for m in ["recall@5", "mrr"]:
                    d, lo, hi = bootstrap_differenz(a, b, m, n_boot, seed)
                    diffs.append({"retriever": n, "referenz": ref, "stil": stil, "metrik": m,
                                  "differenz": d, "ki_unten": lo, "ki_oben": hi})
        pd.DataFrame(diffs).to_csv(args.aus / "differenzen.csv", index=False)

    abbildungen.recall_kurven(raenge, args.abbildungen / "recall_kurven.png")
    if scores:
        abbildungen.enthaltung(scores, args.abbildungen / "enthaltung.png")
    print("fertig:", args.aus)


if __name__ == "__main__":
    main()
