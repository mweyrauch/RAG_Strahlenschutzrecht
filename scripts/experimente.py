"""Rechnet das Variantenraster aus configs/experimente.yaml und schreibt Ergebnisse und Abbildungen.

Ausgaben:
  ergebnisse/raster.csv          Kennzahlen aller Varianten (gesamt und je Frageform)
  ergebnisse/vergleich.csv       Retriever bei Referenz-Zerlegung, mit 95-%-Bootstrap-Intervallen
  ergebnisse/differenzen.csv     gepaarte Differenzen zur Referenzvariante
  ergebnisse/enthaltung.csv      Trennschaerfe des Top-Scores zwischen Korpus- und Fremdfragen (AUC)
  ergebnisse/antworten.csv       extraktive Antworten der besten Variante je Frage
  ergebnisse/zusammenfassung.csv kompakte Tabelle fuer das README
  docs/abbildungen/*.png
"""

import argparse
import sys
import time
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
                                kennzahlen, lade_fragen, stichwort_getroffen,
                                werte_retrieval_aus)
from strlrag.retrieval import baue_retriever  # noqa: E402
import abbildungen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=WURZEL / "configs" / "experimente.yaml")
    ap.add_argument("--aus", type=Path, default=WURZEL / "ergebnisse")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.aus.mkdir(parents=True, exist_ok=True)
    n_boot, seed, ks, tiefe = cfg["bootstrap"], cfg["seed"], cfg["ks"], cfg["tiefe"]

    korpus = standardkorpus()
    fs = lade_fragen(FRAGEN)
    print(f"Korpus: {len(korpus)} Vorschriften, Fragen: {len(fs.fragen)} + {len(fs.fremd)} fremd")

    ergebnisse: dict[tuple, pd.DataFrame] = {}
    retriever_cache: dict[tuple, object] = {}
    zeilen = []
    r = cfg["raster"]
    for strategie in r["strategien"]:
        for kontext in r["kontexte"]:
            chunks = zerlege(korpus, strategie, kontext,
                             cfg["fenster"]["woerter"], cfg["fenster"]["ueberlappung"])
            for name in r["retriever"]:
                t0 = time.perf_counter()
                ret = baue_retriever(name).fit(chunks)
                df = werte_retrieval_aus(ret, fs.fragen, tiefe)
                dauer = time.perf_counter() - t0
                schluessel = (strategie, kontext, name)
                ergebnisse[schluessel] = df
                retriever_cache[schluessel] = ret
                for stil, teil in [("gesamt", df), *df.groupby("stil")]:
                    zeilen.append({"strategie": strategie, "kontext": kontext, "retriever": name,
                                   "stil": stil, "n_chunks": len(chunks), "n_fragen": len(teil),
                                   "sekunden": round(dauer, 2), **kennzahlen(teil, ks)})
                print(f"  {strategie:9s} {kontext:10s} {name:15s} recall@5={kennzahlen(df, ks)['recall@5']:.3f}")
    raster = pd.DataFrame(zeilen)
    raster.to_csv(args.aus / "raster.csv", index=False)

    ref = cfg["referenz"]
    ref_key = (ref["strategie"], ref["kontext"], ref["retriever"])

    # Retrieververgleich bei Referenz-Zerlegung, mit Intervallen
    vergleich = []
    for name in r["retriever"]:
        df = ergebnisse[(ref["strategie"], ref["kontext"], name)]
        for stil, teil in [("gesamt", df), *df.groupby("stil")]:
            for m in ["recall@1", "recall@5", "mrr"]:
                p, lo, hi = bootstrap_ki(teil, m, n_boot, seed)
                vergleich.append({"retriever": name, "stil": stil, "metrik": m,
                                  "wert": p, "ki_unten": lo, "ki_oben": hi})
    vergleich = pd.DataFrame(vergleich)
    vergleich.to_csv(args.aus / "vergleich.csv", index=False)

    # Gepaarte Differenzen jeder Variante zur Referenz (gesamt, Recall@5 und MRR)
    diffs = []
    for k, df in ergebnisse.items():
        if k == ref_key:
            continue
        for m in ["recall@5", "mrr"]:
            d, lo, hi = bootstrap_differenz(df, ergebnisse[ref_key], m, n_boot, seed)
            diffs.append({"strategie": k[0], "kontext": k[1], "retriever": k[2], "metrik": m,
                          "differenz": d, "ki_unten": lo, "ki_oben": hi})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(args.aus / "differenzen.csv", index=False)

    # Beste Variante nach MRR gesamt
    g = raster[raster.stil == "gesamt"].sort_values(["mrr", "recall@5"], ascending=False)
    best_key = tuple(g.iloc[0][["strategie", "kontext", "retriever"]])
    print("Beste Variante:", best_key, "Referenz:", ref_key)

    # Enthaltung: Top-Score Korpusfragen vs. Fremdfragen
    enth = []
    scores_fuer_abb = {}
    for name in r["retriever"]:
        ret = retriever_cache[(ref["strategie"], ref["kontext"], name)]
        pos = ergebnisse[(ref["strategie"], ref["kontext"], name)]["top_score"].to_numpy()
        neg = np.array([ret.suche(q, 1)[0].score for q in fs.fremd])
        a, lo, hi = auc_bootstrap(pos, neg, n_boot, seed)
        schwelle = float(np.quantile(pos, 0.05))
        enth.append({"retriever": name, "rrf_basiert": name.startswith("hybrid"), "auc": a, "ki_unten": lo, "ki_oben": hi,
                     "schwelle_q05": schwelle,
                     "fremd_ueber_schwelle": int((neg >= schwelle).sum()), "n_fremd": len(neg)})
        scores_fuer_abb[name] = (pos, neg)
    enth = pd.DataFrame(enth)
    enth.to_csv(args.aus / "enthaltung.csv", index=False)

    # Antworten (extraktiv) der besten und der Referenzvariante
    ant = []
    backend = ExtraktivBackend(cfg["antwort"]["n_saetze"])
    for label, key in [("referenz", ref_key), ("beste", best_key)]:
        ret = retriever_cache[key]
        for f in fs.fragen:
            a = beantworte(f.text, ret, backend, k=cfg["antwort"]["k"])
            kontext_text = " ".join(t.chunk.text for t in a.quellen)
            zitiert = {a.quellen[n - 1].chunk.schluessel for n in a.zitierte_nummern
                       if 1 <= n <= len(a.quellen)}
            ant.append({"variante": label, "frage": f.id, "item": f.item, "stil": f.stil,
                        "stichwort_im_kontext": stichwort_getroffen(kontext_text, f.stichworte),
                        "stichwort_in_antwort": stichwort_getroffen(a.text, f.stichworte),
                        "soll_zitiert": bool(zitiert & f.soll),
                        "ungueltige_zitate": len(a.ungueltige_zitate),
                        "antwort": a.text})
    ant = pd.DataFrame(ant)
    ant.to_csv(args.aus / "antworten.csv", index=False)

    # Zusammenfassung
    zus = []
    for label, key in [("Referenz", ref_key), ("Beste", best_key)]:
        df = ergebnisse[key]
        for stil, teil in [("gesamt", df), *df.groupby("stil")]:
            zeile = {"variante": label, "konfiguration": " / ".join(key), "stil": stil}
            for m in ["recall@1", "recall@5", "mrr"]:
                p, lo, hi = bootstrap_ki(teil, m, n_boot, seed)
                zeile[m] = f"{p:.2f} [{lo:.2f}, {hi:.2f}]"
            a = ant[(ant.variante == label.lower()) & (ant.frage.isin(teil.frage))]
            zeile["stichwort_im_kontext"] = f"{a.stichwort_im_kontext.mean():.2f}"
            zeile["stichwort_in_antwort"] = f"{a.stichwort_in_antwort.mean():.2f}"
            zus.append(zeile)
    pd.DataFrame(zus).to_csv(args.aus / "zusammenfassung.csv", index=False)

    abb = WURZEL / "docs" / "abbildungen"
    abb.mkdir(parents=True, exist_ok=True)
    abbildungen.retrieververgleich(vergleich, abb / "retrieververgleich.png")
    abbildungen.zerlegung(raster, abb / "zerlegung_kontext.png", r["retriever"])
    abbildungen.recall_kurven(ergebnisse, ref, r["retriever"], abb / "recall_kurven.png")
    abbildungen.enthaltung(scores_fuer_abb, abb / "enthaltung.png")
    print("fertig:", args.aus)


if __name__ == "__main__":
    main()
