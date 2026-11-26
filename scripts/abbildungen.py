"""Abbildungen fuer das README.

Feste Farbzuordnung je Retriever (Pastell). Weil die Pastelltoene auf Weiss wenig
Kontrast haben, tragen Markerformen, dunkle Konturen und die Legende die Zuordnung mit.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E5 = "dense:intfloat/multilingual-e5-"
FARBE = {"bm25": "#F4B6C2", E5 + "small": "#C3B1E1", E5 + "base": "#FFC9A3", "dense": "#9EC5E8"}
MARKER = {"bm25": "D", E5 + "small": "s", E5 + "base": "^", "dense": "o"}
NAME = {"bm25": "BM25 (Basislinie)", E5 + "small": "e5-small", E5 + "base": "e5-base",
        "dense": "e5-large"}
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
