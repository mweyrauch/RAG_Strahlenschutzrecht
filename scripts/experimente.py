"""Wertet Retriever und Antwort-Backends aus einer Konfiguration auf dem Fragenset aus.

Ausgaben (in --aus, Standard ergebnisse/):
  kennzahlen.csv        Recall@k und MRR je Retriever und Frageform, mit 95-%-Bootstrap-Intervallen
  raenge.csv            Rang der Soll-Vorschrift und bester Treffer je Frage (fuer die Fehleranalyse)
  differenzen.csv       gepaarte Differenzen zur Referenz (nur bei mehr als einem Retriever)
  enthaltung.csv        Trennschaerfe des Top-Scores zwischen Korpus- und Fremdfragen (AUC)
  fremdfragen.csv       Score und bester Treffer je Fremdfrage, mit Angabe, ob sie die Schwelle passiert
  antwortqualitaet.csv  je Retriever, Backend, Fragegruppe und Vorfilter: Anteil korrekter Antworten
                        bzw. korrekter Enthaltungen, mit 95-%-Intervallen
  antworten.csv         alle Antworten im Wortlaut
  docs/abbildungen/*.png

Eine Antwort auf eine Korpusfrage gilt als korrekt, wenn das System sich nicht enthaelt und
eines der antwort_muster des Items im Antworttext steht. Bei Fremdfragen ist die Enthaltung
korrekt. Der Vorfilter "schwelle" enthaelt sich zusaetzlich, wenn der beste Treffer unter dem
5-%-Quantil der Top-Scores der Korpusfragen liegt (auf denselben Fragen bestimmt).
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
from strlrag.antwort import ExtraktivBackend, OpenAIKompatibelBackend, beantworte  # noqa: E402
from strlrag.chunking import zerlege  # noqa: E402
from strlrag.daten import FRAGEN, standardkorpus  # noqa: E402
from strlrag.evaluation import (auc_bootstrap, bootstrap_differenz, bootstrap_ki,  # noqa: E402
                                bootstrap_mittelwert, lade_fragen, stichwort_getroffen,
                                werte_retrieval_aus)
from strlrag.retrieval import baue_retriever  # noqa: E402
import abbildungen  # noqa: E402

LLM_CACHE = WURZEL / "data" / "cache" / "llm"


def baue_backend(name: str, cfg: dict):
    if name == "extraktiv":
        return ExtraktivBackend(cfg.get("n_saetze", 2))
    if name == "llm":
        return OpenAIKompatibelBackend(cache_dir=LLM_CACHE)
    raise ValueError(f"Unbekanntes Backend: {name}")


def backend_label(name: str, backend) -> str:
    return f"llm:{backend.modell}" if name == "llm" else name


def antworten(ret, fs, backend, k, label):
    """Beantwortet alle Korpus- und Fremdfragen; eine Zeile je Frage."""
    zeilen = []
    fragen = [(f.id, f.item, f.stil, f.text, f) for f in fs.fragen]
    fragen += [(f"f{i + 1:02d}", f"f{i + 1:02d}", "fremd", q, None) for i, q in enumerate(fs.fremd)]
    t0 = time.perf_counter()
    for j, (fid, item, stil, text, f) in enumerate(fragen, start=1):
        a = beantworte(text, ret, backend, k=k)
        zitiert = {a.quellen[n - 1].chunk.schluessel for n in a.zitierte_nummern
                   if 1 <= n <= len(a.quellen)}
        zeile = {"backend": label, "frage": fid, "item": item, "stil": stil, "text": text,
                 "top_score": a.quellen[0].score if a.quellen else 0.0,
                 "enthalten": a.enthalten, "ungueltige_zitate": len(a.ungueltige_zitate),
                 "antwort": a.text}
        if f is not None:
            kontext = " ".join(t.chunk.text for t in a.quellen)
            zeile["muster_im_kontext"] = bool(stichwort_getroffen(kontext, f.muster))
            zeile["muster_in_antwort"] = bool(stichwort_getroffen(a.text, f.muster))
            zeile["soll_zitiert"] = bool(zitiert & f.soll)
            zeile["korrekt"] = (not a.enthalten) and zeile["muster_in_antwort"]
        else:
            zeile["korrekt"] = a.enthalten
        zeilen.append(zeile)
        if label != "extraktiv" and (j % 10 == 0 or j == len(fragen)):
            print(f"    {label}: {j}/{len(fragen)} Fragen ({time.perf_counter() - t0:.0f} s)", flush=True)
    return pd.DataFrame(zeilen)


def antwortqualitaet(df: pd.DataFrame, schwelle: float, n_boot: int, seed: int) -> list[dict]:
    zeilen = []
    for vorfilter in ["ohne", "schwelle"]:
        d = df.copy()
        if vorfilter == "schwelle":
            unter = d["top_score"] < schwelle
            d["enthalten"] = d["enthalten"] | unter
            fremd = d["stil"] == "fremd"
            d.loc[fremd, "korrekt"] = d.loc[fremd, "enthalten"]
            d.loc[~fremd, "korrekt"] = d.loc[~fremd, "korrekt"] & ~unter[~fremd]
        for gruppe, g in d.groupby("stil"):
            p, lo, hi = bootstrap_mittelwert(g["korrekt"].astype(float), g["item"], n_boot, seed)
            zeile = {"backend": g["backend"].iloc[0], "gruppe": gruppe, "vorfilter": vorfilter,
                     "n": len(g), "korrekt": p, "korrekt_ki_unten": lo, "korrekt_ki_oben": hi,
                     "enthalten": g["enthalten"].mean(),
                     "ungueltige_zitate": (g["ungueltige_zitate"] > 0).mean()}
            if gruppe != "fremd":
                zeile["muster_im_kontext"] = g["muster_im_kontext"].mean()
                zeile["soll_zitiert"] = g["soll_zitiert"].mean()
            zeilen.append(zeile)
    return zeilen


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
    acfg = cfg["antwort"]

    korpus = standardkorpus()
    chunks = zerlege(korpus)
    fs = lade_fragen(FRAGEN)
    print(f"Korpus: {len(korpus)} Vorschriften, {len(chunks)} Absaetze; "
          f"Fragen: {len(fs.fragen)} + {len(fs.fremd)} fremd")

    raenge, kennz, enth, fremd, scores, ants, qual = {}, [], [], [], {}, [], []
    for name in cfg["retriever"]:
        ret = baue_retriever(name).fit(chunks)
        df = werte_retrieval_aus(ret, fs.fragen, cfg["tiefe"])
        raenge[name] = df

        for stil, teil in [("gesamt", df), *df.groupby("stil")]:
            zeile = {"retriever": name, "stil": stil, "n_fragen": len(teil)}
            for m in [f"recall@{k}" for k in ks] + ["mrr"]:
                zeile[m], zeile[f"{m}_ki_unten"], zeile[f"{m}_ki_oben"] = bootstrap_ki(teil, m, n_boot, seed)
            kennz.append(zeile)
        print(f"  {name:22s} recall@5 fach={kennz[-1]['recall@5']:.2f} "
              f"alltag={kennz[-2]['recall@5']:.2f}")

        # Enthaltung: trennt der Score des besten Treffers Korpus- von Fremdfragen?
        pos = df["top_score"].to_numpy()
        fremd_treffer = [ret.suche(q, 1)[0] for q in fs.fremd]
        neg = np.array([t.score for t in fremd_treffer])
        a, lo, hi = auc_bootstrap(pos, neg, n_boot, seed)
        schwelle = float(np.quantile(pos, 0.05))
        enth.append({"retriever": name, "auc": a, "ki_unten": lo, "ki_oben": hi,
                     "schwelle_q05": schwelle,
                     "fremd_ueber_schwelle": int((neg >= schwelle).sum()), "n_fremd": len(neg)})
        for q, t in zip(fs.fremd, fremd_treffer):
            fremd.append({"retriever": name, "frage": q, "top_score": t.score,
                          "top_quelle": t.chunk.schluessel, "ueber_schwelle": t.score >= schwelle})
        scores[name] = (pos, neg)

        # Antworten je Backend
        for bname in acfg.get("backends", ["extraktiv"]):
            backend = baue_backend(bname, acfg)
            label = backend_label(bname, backend)
            an = antworten(ret, fs, backend, acfg["k"], label).assign(retriever=name)
            ants.append(an)
            qual += [{"retriever": name, **z} for z in antwortqualitaet(an, schwelle, n_boot, seed)]

    pd.DataFrame(kennz).to_csv(args.aus / "kennzahlen.csv", index=False)
    pd.concat([d.assign(retriever=n) for n, d in raenge.items()]).to_csv(args.aus / "raenge.csv", index=False)
    pd.DataFrame(enth).to_csv(args.aus / "enthaltung.csv", index=False)
    pd.DataFrame(fremd).to_csv(args.aus / "fremdfragen.csv", index=False)
    pd.concat(ants).to_csv(args.aus / "antworten.csv", index=False)
    pd.DataFrame(qual).to_csv(args.aus / "antwortqualitaet.csv", index=False)

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
    abbildungen.enthaltung(scores, args.abbildungen / "enthaltung.png")
    print("fertig:", args.aus)


if __name__ == "__main__":
    main()
