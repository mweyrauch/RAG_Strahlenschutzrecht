"""Abbildungen fuer das README. Feste Farbzuordnung je Retriever (Pastell) plus Schraffur
und Beschriftung, damit die Zuordnung nicht allein ueber die Farbe laeuft."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FARBE = {
    "tfidf": "#9EC5E8",
    "tfidf_stem": "#C3B1E1",
    "bm25": "#F4B6C2",
    "zeichen_ngramm": "#FFC9A3",
    "hybrid": "#A8D8B9",
    "lsa": "#D9C2A7",
    "dense": "#F7D794",
}
SCHRAFFUR = {"tfidf": "", "tfidf_stem": "//", "bm25": "", "zeichen_ngramm": "..", "hybrid": "xx",
             "lsa": "--", "dense": "\\\\"}
MARKER = {"tfidf": "o", "tfidf_stem": "s", "bm25": "D", "zeichen_ngramm": "^", "hybrid": "v",
          "lsa": "X", "dense": "P"}
NAME = {
    "tfidf": "TF-IDF (Wort)",
    "tfidf_stem": "TF-IDF + Stemming",
    "bm25": "BM25 + Stemming",
    "zeichen_ngramm": "Zeichen-3-5-Gramme",
    "hybrid": "Hybrid (BM25 + Zeichen, RRF)",
    "lsa": "LSA (TF-IDF + SVD)",
    "dense": "Embeddings (e5-small)",
}
# Weitere Varianten (z. B. "hybrid:bm25+dense") bekommen die Reservefarben der Palette.
RESERVE = ["#B5EAEA", "#F7D794", "#D9C2A7"]


def _stil(n):
    if n not in FARBE:
        FARBE[n] = RESERVE[len([k for k in FARBE if k not in SCHRAFFUR]) % len(RESERVE)]
        SCHRAFFUR[n], MARKER[n], NAME[n] = "++", "*", n
    return FARBE[n]
STIL = {"fach": "Fachsprachliche Fragen", "alltag": "Alltagssprachliche Fragen"}
KANTE = "#4a4a4a"
TEXT = "#333333"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#eeeeee",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def _speichern(fig, pfad: Path):
    fig.tight_layout()
    fig.savefig(pfad, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def retrieververgleich(vergleich, pfad: Path):
    """Recall@5 und MRR je Retriever, getrennt nach Frageform, mit 95-%-Intervallen."""
    namen = list(dict.fromkeys(vergleich.retriever))
    for n in namen:
        _stil(n)
    fig, achsen = plt.subplots(2, 2, figsize=(10, 7), sharey="row", sharex=True)
    for zeile, metrik in enumerate(["recall@5", "mrr"]):
        for spalte, stil in enumerate(["fach", "alltag"]):
            ax = achsen[zeile, spalte]
            d = vergleich[(vergleich.stil == stil) & (vergleich.metrik == metrik)].set_index("retriever").loc[namen]
            y = np.arange(len(namen))[::-1]
            for yi, n in zip(y, namen):
                w, lo, hi = d.loc[n, ["wert", "ki_unten", "ki_oben"]]
                ax.barh(yi, w, color=FARBE[n], hatch=SCHRAFFUR[n], edgecolor=KANTE, linewidth=0.6, height=0.62)
                ax.errorbar(w, yi, xerr=[[w - lo], [hi - w]], color=KANTE, capsize=3, linewidth=1)
                ax.text(hi + 0.025, yi, f"{w:.2f}", va="center", fontsize=9, color=TEXT)
            ax.set_yticks(y, [NAME[n] for n in namen])
            ax.set_xlim(0, 1.15)
            ax.grid(axis="y", visible=False)
            if zeile == 0:
                ax.set_title(STIL[stil], fontsize=11, color=TEXT)
            if spalte == 0:
                ax.set_ylabel("Recall@5" if metrik == "recall@5" else "MRR")
    fig.suptitle("Retriever bei Absatz-Zerlegung mit Titel-Pr\u00e4fix (95-%-Bootstrap-Intervalle, 40 Items je Frageform)",
                 fontsize=11, color=TEXT)
    _speichern(fig, pfad)


def zerlegung(raster, pfad: Path, namen):
    """MRR je Zerlegung und Kontext-Praefix, alle Fragen."""
    g = raster[raster.stil == "gesamt"]
    strategien = list(dict.fromkeys(g.strategie))
    kontexte = list(dict.fromkeys(g.kontext))
    titel = {"paragraph": "Ganze Vorschrift", "absatz": "Absatz", "fenster": "Wortfenster (120/40)"}
    ktitel = {"ohne": "ohne", "titel": "Titel", "gliederung": "Gliederung + Titel"}
    fig, achsen = plt.subplots(1, len(strategien), figsize=(11, 3.8), sharey=True)
    x = np.arange(len(kontexte))
    achsen = np.atleast_1d(achsen)
    for n in namen:
        _stil(n)
    for ax, s in zip(achsen, strategien):
        for n in namen:
            d = g[(g.strategie == s) & (g.retriever == n)].set_index("kontext").loc[kontexte]
            ax.plot(x, d["mrr"], marker=MARKER[n], color=FARBE[n], markeredgecolor=KANTE,
                    markersize=8, linewidth=2, label=NAME[n])
        ax.set_xticks(x, [ktitel[k] for k in kontexte])
        ax.set_title(titel.get(s, s), fontsize=11, color=TEXT)
        ax.set_xlim(-0.3, len(kontexte) - 0.7)
    achsen[0].set_ylabel("MRR (alle 80 Fragen)")
    achsen[0].set_ylim(0, 1)
    griffe, labels = achsen[0].get_legend_handles_labels()
    fig.legend(griffe, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Einfluss von Zerlegung und Kontext-Pr\u00e4fix", fontsize=11, color=TEXT)
    _speichern(fig, pfad)


def recall_kurven(ergebnisse, ref, namen, pfad: Path, kmax: int = 20):
    fig, achsen = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ks = np.arange(1, kmax + 1)
    for n in namen:
        _stil(n)
    for ax, stil in zip(achsen, ["fach", "alltag"]):
        for n in namen:
            df = ergebnisse[(ref["strategie"], ref["kontext"], n)]
            r = df[df.stil == stil]["rang"].to_numpy(dtype=float)
            r = np.nan_to_num(r, nan=np.inf)
            ax.plot(ks, [(r <= k).mean() for k in ks], color=FARBE[n], linewidth=2,
                    marker=MARKER[n], markevery=[0, 4, 9, 19], markeredgecolor=KANTE,
                    markersize=7, label=NAME[n])
        ax.set_title(STIL[stil], fontsize=11, color=TEXT)
        ax.set_xlabel("k (abgerufene Vorschriften)")
        ax.set_xticks([1, 5, 10, 15, 20])
    achsen[0].set_ylabel("Recall@k")
    achsen[0].set_ylim(0, 1.02)
    achsen[0].legend(loc="lower right", fontsize=8, frameon=False)
    _speichern(fig, pfad)


def enthaltung(scores, pfad: Path):
    """Top-1-Score von Korpusfragen und Fremdfragen je Retriever."""
    # RRF-Scores bilden nur R\u00e4nge ab und eignen sich nicht als Schwelle
    namen = [n for n in scores if not n.startswith("hybrid")]
    for n in namen:
        _stil(n)
    fig, achsen = plt.subplots(1, len(namen), figsize=(11, 3.4))
    rng = np.random.default_rng(0)
    for ax, n in zip(achsen, namen):
        pos, neg = scores[n]
        for i, (werte, label) in enumerate([(pos, "Korpus"), (neg, "fremd")]):
            jitter = rng.uniform(-0.12, 0.12, len(werte))
            ax.scatter(np.full(len(werte), i) + jitter, werte, s=22,
                       color=FARBE[n] if i == 0 else "#d9d9d9", edgecolor=KANTE, linewidth=0.5)
        ax.axhline(np.quantile(pos, 0.05), color=KANTE, linestyle="--", linewidth=1)
        ax.set_xticks([0, 1], ["Korpus", "fremd"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(NAME[n], fontsize=9, color=TEXT)
        ax.grid(axis="x", visible=False)
    achsen[0].set_ylabel("Score des besten Treffers")
    fig.suptitle("Trennt der Top-Score beantwortbare von fremden Fragen? (gestrichelt: 5-%-Quantil der Korpusfragen)",
                 fontsize=10, color=TEXT)
    _speichern(fig, pfad)
