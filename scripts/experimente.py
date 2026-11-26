"""Wertet Retriever und Antwort-Backends einer Konfiguration auf dem Fragenset aus.

Aufruf:  python scripts/experimente.py configs/retrieval.yaml
         python scripts/experimente.py configs/antworten.yaml

Ausgaben in ergebnisse/<Name der Konfiguration>/:
  kennzahlen.csv        Recall@k und MRR je Retriever und Frageform, mit 95-%-Bootstrap-Intervallen
  differenzen.csv       gepaarte Differenzen zur Referenz (bei mehr als einem Retriever)
  raenge.csv            Rang der Soll-Vorschrift und bester Treffer je Frage
  enthaltung.csv        Trennschaerfe des Top-Scores zwischen Korpus- und Fremdfragen (AUC)
  fremdfragen.csv       Score und bester Treffer je Fremdfrage
  antwortqualitaet.csv  Anteil korrekter Antworten bzw. Enthaltungen je Retriever, Backend und Gruppe
  antworten.csv         alle Antworten im Wortlaut
Mit "abbildungen: true" zusaetzlich docs/abbildungen/recall_kurven.png.

Eine Antwort auf eine Korpusfrage gilt als korrekt, wenn das System sich nicht enthaelt und
eines der antwort_muster des Items im Antworttext steht. Bei Fremdfragen ist die Enthaltung
korrekt.
"""

import argparse
import gc
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
        return ExtraktivBackend(cfg.get("n_saetze", 2)), "extraktiv"
    if name == "llm":
        b = OpenAIKompatibelBackend(cache_dir=LLM_CACHE)
        return b, f"llm:{b.modell}"
    raise ValueError(f"Unbekanntes Backend: {name}")


class Vorberechnet:
    """Liefert zuvor abgerufene Treffer, damit das Embedding-Modell vor den Anfragen an das
    Sprachmodell aus dem Speicher entfernt werden kann (beide zusammen brauchen viel RAM)."""

    def __init__(self, ret, texte, k):
        self.name = ret.name
        self.treffer = {t: ret.suche(t, k) for t in texte}

    def suche(self, frage, k=10):
        return self.treffer[frage][:k]


def antworten(ret, fs, backend, k, label):
    """Beantwortet alle Korpus- und Fremdfragen; eine Zeile je Frage."""
    fragen = [(f.id, f.item, f.stil, f.text, f) for f in fs.fragen]
    fragen += [(f"f{i + 1:02d}", f"f{i + 1:02d}", "fremd", q, None) for i, q in enumerate(fs.fremd)]
    zeilen, t0 = [], time.perf_counter()
    for j, (fid, item, stil, text, f) in enumerate(fragen, start=1):
        a = beantworte(text, ret, backend, k=k)
        zeile = {"backend": label, "frage": fid, "item": item, "stil": stil, "text": text,
                 "enthalten": a.enthalten, "ungueltige_zitate": len(a.ungueltige_zitate),
                 "antwort": a.text}
        if f is None:
            zeile["korrekt"] = a.enthalten
        else:
            zitiert = {a.quellen[n - 1].chunk.schluessel for n in a.zitierte_nummern
                       if 1 <= n <= len(a.quellen)}
            kontext = " ".join(t.chunk.text for t in a.quellen)
            zeile["muster_im_kontext"] = bool(stichwort_getroffen(kontext, f.muster))
            zeile["soll_zitiert"] = bool(zitiert & f.soll)
            zeile["korrekt"] = (not a.enthalten) and bool(stichwort_getroffen(a.text, f.muster))
        zeilen.append(zeile)
        if label != "extraktiv" and (j % 10 == 0 or j == len(fragen)):
            print(f"    {label}: {j}/{len(fragen)} Fragen ({time.perf_counter() - t0:.0f} s)", flush=True)
    return pd.DataFrame(zeilen)


def antwortqualitaet(df: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    zeilen = []
    for gruppe, g in df.groupby("stil"):
        p, lo, hi = bootstrap_mittelwert(g["korrekt"].astype(float), g["item"], n_boot, seed)
        zeile = {"backend": g["backend"].iloc[0], "gruppe": gruppe, "n": len(g),
                 "korrekt": p, "korrekt_ki_unten": lo, "korrekt_ki_oben": hi,
                 "enthalten": g["enthalten"].mean(),
                 "ungueltige_zitate": (g["ungueltige_zitate"] > 0).mean()}
        if gruppe != "fremd":
            zeile["muster_im_kontext"] = g["muster_im_kontext"].mean()
            zeile["soll_zitiert"] = g["soll_zitiert"].mean()
        zeilen.append(zeile)
    return zeilen


def main():
    ap = argparse.ArgumentParser(description="Auswertung einer Konfiguration")
    ap.add_argument("config", type=Path, help="z. B. configs/retrieval.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    aus = WURZEL / "ergebnisse" / args.config.stem
    aus.mkdir(parents=True, exist_ok=True)
    n_boot, seed, ks, acfg = cfg["bootstrap"], cfg["seed"], cfg["ks"], cfg["antwort"]

    korpus = standardkorpus()
    chunks = zerlege(korpus)
    fs = lade_fragen(FRAGEN)
    print(f"Korpus: {len(korpus)} Vorschriften, {len(chunks)} Absaetze; "
          f"Fragen: {len(fs.fragen)} + {len(fs.fremd)} fremd")

    raenge, kennz, enth, fremd, ants, qual = {}, [], [], [], [], []
    for name in cfg["retriever"]:
        ret = baue_retriever(name).fit(chunks)
        df = werte_retrieval_aus(ret, fs.fragen, cfg["tiefe"])
        raenge[name] = df
        for stil, teil in [("gesamt", df), *df.groupby("stil")]:
            zeile = {"retriever": name, "stil": stil, "n_fragen": len(teil)}
            for m in [f"recall@{k}" for k in ks] + ["mrr"]:
                zeile[m], zeile[f"{m}_ki_unten"], zeile[f"{m}_ki_oben"] = bootstrap_ki(teil, m, n_boot, seed)
            kennz.append(zeile)
        print(f"  {name:40s} recall@5 fach={kennz[-1]['recall@5']:.2f} "
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

        # Antworten; bei Sprachmodell vorher Retrieval abschliessen und Embedding-Modell freigeben
        if "llm" in acfg["backends"]:
            ret = Vorberechnet(ret, [f.text for f in fs.fragen] + list(fs.fremd), acfg["k"])
            gc.collect()
        for bname in acfg["backends"]:
            backend, label = baue_backend(bname, acfg)
            an = antworten(ret, fs, backend, acfg["k"], label).assign(retriever=name)
            ants.append(an)
            qual += [{"retriever": name, **z} for z in antwortqualitaet(an, n_boot, seed)]

    pd.DataFrame(kennz).to_csv(aus / "kennzahlen.csv", index=False)
    pd.concat([d.assign(retriever=n) for n, d in raenge.items()]).to_csv(aus / "raenge.csv", index=False)
    pd.DataFrame(enth).to_csv(aus / "enthaltung.csv", index=False)
    pd.DataFrame(fremd).to_csv(aus / "fremdfragen.csv", index=False)
    pd.concat(ants).to_csv(aus / "antworten.csv", index=False)
    pd.DataFrame(qual).to_csv(aus / "antwortqualitaet.csv", index=False)

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
        pd.DataFrame(diffs).to_csv(aus / "differenzen.csv", index=False)

    if cfg.get("abbildungen"):
        ziel = WURZEL / "docs" / "abbildungen"
        ziel.mkdir(parents=True, exist_ok=True)
        abbildungen.recall_kurven(raenge, ziel / "recall_kurven.png")
    print("fertig:", aus)


if __name__ == "__main__":
    main()
