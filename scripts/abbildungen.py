"""Abbildungen fuer das README.

Feste Farbzuordnung je Retriever (Pastell). Weil die Pastelltoene auf Weiss wenig
Kontrast haben, tragen Markerformen, dunkle Konturen und die Legende die Zuordnung mit.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FARBE = {
    "bm25": "#F4B6C2",
    "dense": "#9EC5E8",
    "umformuliert:dense": "#FFC9A3",
}
MARKER = {"bm25": "D", "dense": "o", "umformuliert:dense": "^"}
NAME = {
    "bm25": "BM25 (Basislinie)",
    "dense": "Embeddings (multilingual-e5-small)",
    "umformuliert:dense": "Embeddings + Umformulierung",
}
RESERVE = ["#F7D794", "#B5EAEA", "#D9C2A7"]
STIL = {"fach": "Fachsprachliche Fragen", "alltag": "Alltagssprachliche Fragen"}
KANTE = "#4a4a4a"
TEXT = "#333333"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#eeeeee",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def _stil(n):
    """Unbekannte Varianten bekommen der Reihe nach eine Reservefarbe."""
    if n not in FARBE:
        unbekannt = [k for k in FARBE if k not in NAME]
        FARBE[n] = RESERVE[len(unbekannt) % len(RESERVE)]
        MARKER[n] = "X"
    return FARBE[n], MARKER[n], NAME.get(n, n)


def _speichern(fig, pfad: Path):
    fig.tight_layout()
    fig.savefig(pfad, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def recall_kurven(raenge: dict, pfad: Path, kmax: int = 20):
    """Recall@k je Retriever, getrennt nach Frageform."""
    fig, achsen = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ks = np.arange(1, kmax + 1)
    for ax, stil in zip(achsen, ["fach", "alltag"]):
        for n, df in raenge.items():
            farbe, marker, name = _stil(n)
            r = np.nan_to_num(df[df.stil == stil]["rang"].to_numpy(dtype=float), nan=np.inf)
            ax.plot(ks, [(r <= k).mean() for k in ks], color=farbe, linewidth=2.2,
                    marker=marker, markevery=[0, 4, 9, 19], markeredgecolor=KANTE,
                    markersize=7, label=name)
        ax.set_title(STIL[stil], fontsize=11, color=TEXT)
        ax.set_xlabel("k (abgerufene Vorschriften)")
        ax.set_xticks([1, 5, 10, 15, 20])
    achsen[0].set_ylabel("Recall@k (Anteil der 40 Fragen)")
    achsen[0].set_ylim(0, 1.02)
    achsen[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Wie oft ist die richtige Vorschrift unter den ersten k Treffern?",
                 fontsize=11, color=TEXT)
    _speichern(fig, pfad)


def enthaltung(scores: dict, pfad: Path):
    """Top-1-Score von Korpusfragen und Fremdfragen je Retriever."""
    namen = list(scores)
    fig, achsen = plt.subplots(1, len(namen), figsize=(3.2 * len(namen) + 1, 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, n in zip(achsen[0], namen):
        farbe, _, name = _stil(n)
        pos, neg = scores[n]
        for i, werte in enumerate([pos, neg]):
            jitter = rng.uniform(-0.12, 0.12, len(werte))
            ax.scatter(np.full(len(werte), i) + jitter, werte, s=24,
                       color=farbe if i == 0 else "#d9d9d9", edgecolor=KANTE, linewidth=0.5)
        ax.axhline(np.quantile(pos, 0.05), color=KANTE, linestyle="--", linewidth=1)
        ax.set_xticks([0, 1], ["Korpusfragen", "Fremdfragen"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(name, fontsize=10, color=TEXT)
        ax.grid(axis="x", visible=False)
    achsen[0][0].set_ylabel("Score des besten Treffers")
    fig.suptitle("Trennt der Top-Score beantwortbare von fremden Fragen?\n"
                 "(gestrichelt: Schwelle, die 95 % der Korpusfragen durchl\u00e4sst)",
                 fontsize=10, color=TEXT)
    _speichern(fig, pfad)
