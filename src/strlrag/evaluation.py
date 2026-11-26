"""Bewertung von Retrieval und Antworten.

Retrieval wird auf Paragraphenebene bewertet: Ein Treffer zaehlt, wenn ein
abgerufener Chunk zu einer der Soll-Vorschriften gehoert. Mehrere Chunks
derselben Vorschrift zaehlen fuer den Rang nur einmal, damit feine Zerlegungen
nicht kuenstlich bevorzugt oder benachteiligt werden.

Unsicherheit: Bootstrap ueber Fragen-Items (Cluster: beide Formulierungen eines
Items werden gemeinsam gezogen), gepaart fuer Variantenvergleiche.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .retrieval import Retriever, Treffer
from .text import falte


@dataclass(frozen=True)
class Frage:
    id: str            # z. B. "q07-alltag"
    item: str          # z. B. "q07"
    stil: str          # "fach" oder "alltag"
    text: str
    soll: frozenset[str]
    stichworte: tuple[str, ...] = ()
    thema: str = ""
    muster: tuple[str, ...] = ()   # akzeptierte Antwortformen, z. B. "20 mSv"; Standard: stichworte


@dataclass
class Fragenset:
    fragen: list[Frage]
    fremd: list[str] = field(default_factory=list)  # Fragen ausserhalb des Korpus


def lade_fragen(pfad: str | Path) -> Fragenset:
    daten = yaml.safe_load(Path(pfad).read_text(encoding="utf-8"))
    fragen: list[Frage] = []
    for e in daten["fragen"]:
        for stil in ("fach", "alltag"):
            if e.get(stil):
                fragen.append(Frage(
                    id=f"{e['id']}-{stil}", item=e["id"], stil=stil, text=e[stil],
                    soll=frozenset(e["soll"]), stichworte=tuple(e.get("stichworte", [])),
                    thema=e.get("thema", ""),
                    muster=tuple(e.get("antwort_muster") or e.get("stichworte", [])),
                ))
    return Fragenset(fragen, list(daten.get("fremd", [])))


def rang_paragraph(treffer: list[Treffer], soll: frozenset[str]) -> int | None:
    """1-basierter Rang der ersten Soll-Vorschrift unter den eindeutigen Vorschriften."""
    gesehen: list[str] = []
    for t in treffer:
        s = t.chunk.schluessel
        if s in gesehen:
            continue
        gesehen.append(s)
        if s in soll:
            return len(gesehen)
    return None


def werte_retrieval_aus(retriever: Retriever, fragen: list[Frage], tiefe: int = 50) -> pd.DataFrame:
    zeilen = []
    for f in fragen:
        treffer = retriever.suche(f.text, tiefe)
        zeilen.append({
            "frage": f.id, "item": f.item, "stil": f.stil, "thema": f.thema,
            "rang": rang_paragraph(treffer, f.soll),
            "top_score": treffer[0].score if treffer else 0.0,
            "top_quelle": treffer[0].chunk.schluessel if treffer else "",
        })
    return pd.DataFrame(zeilen)


def recall_at(raenge: pd.Series, k: int) -> float:
    r = raenge.to_numpy(dtype=float)  # None -> nan
    return float(np.mean(np.nan_to_num(r, nan=np.inf) <= k))


def mrr(raenge: pd.Series) -> float:
    r = raenge.to_numpy(dtype=float)
    return float(np.mean(np.where(np.isnan(r), 0.0, 1.0 / np.nan_to_num(r, nan=1.0))))


def _wert_je_frage(df: pd.DataFrame, name: str) -> np.ndarray:
    """Beitrag jeder Frage zur Metrik (Metriken sind Mittelwerte ueber Fragen)."""
    r = np.nan_to_num(df["rang"].to_numpy(dtype=float), nan=np.inf)
    if name == "mrr":
        return 1.0 / r
    if name.startswith("recall@"):
        return (r <= int(name.split("@")[1])).astype(float)
    raise ValueError(name)


def _cluster_bootstrap(werte: np.ndarray, items: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Cluster-Bootstrap fuer einen Mittelwert ueber Fragen.

    Gezogen werden Items mit Zuruecklegen; der Mittelwert einer Ziehung ist
    (Summe der Werte der gezogenen Items) / (Anzahl ihrer Fragen).
    """
    eindeutig, inv = np.unique(items, return_inverse=True)
    summe = np.bincount(inv, weights=werte, minlength=len(eindeutig))
    anzahl = np.bincount(inv, minlength=len(eindeutig)).astype(float)
    rng = np.random.default_rng(seed)
    zieh = rng.integers(0, len(eindeutig), size=(n, len(eindeutig)))
    return summe[zieh].sum(axis=1) / anzahl[zieh].sum(axis=1)


def bootstrap_mittelwert(werte, items, n: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """Mittelwert und Perzentil-Intervall fuer beliebige Werte je Frage (Cluster: items)."""
    w = np.asarray(werte, dtype=float)
    b = _cluster_bootstrap(w, np.asarray(items), n, seed)
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    return float(w.mean()), float(lo), float(hi)


def bootstrap_ki(df: pd.DataFrame, metrik: str, n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Punktschaetzer und Perzentil-Intervall, Cluster-Bootstrap ueber Items."""
    w = _wert_je_frage(df, metrik)
    werte = _cluster_bootstrap(w, df["item"].to_numpy(), n, seed)
    lo, hi = np.quantile(werte, [alpha / 2, 1 - alpha / 2])
    return float(w.mean()), float(lo), float(hi)


def bootstrap_differenz(a: pd.DataFrame, b: pd.DataFrame, metrik: str, n: int = 2000,
                        seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Gepaarte Differenz metrik(a) - metrik(b) auf denselben Fragen."""
    a = a.sort_values("frage").reset_index(drop=True)
    b = b.sort_values("frage").reset_index(drop=True)
    if list(a["frage"]) != list(b["frage"]):
        raise ValueError("Varianten wurden nicht auf denselben Fragen ausgewertet")
    d = _wert_je_frage(a, metrik) - _wert_je_frage(b, metrik)
    werte = _cluster_bootstrap(d, a["item"].to_numpy(), n, seed)
    lo, hi = np.quantile(werte, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def auc(positiv: np.ndarray, negativ: np.ndarray) -> float:
    """Flaeche unter der ROC-Kurve (Mann-Whitney), Bindungen zaehlen halb."""
    p = np.asarray(positiv, dtype=float)[:, None]
    q = np.asarray(negativ, dtype=float)[None, :]
    return float(np.mean((p > q) + 0.5 * (p == q)))


def auc_bootstrap(positiv, negativ, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    p, q = np.asarray(positiv), np.asarray(negativ)
    werte = [auc(rng.choice(p, len(p)), rng.choice(q, len(q))) for _ in range(n)]
    lo, hi = np.quantile(werte, [alpha / 2, 1 - alpha / 2])
    return auc(p, q), float(lo), float(hi)


def stichwort_getroffen(text: str, stichworte: tuple[str, ...]) -> bool | None:
    """True, wenn mindestens ein Stichwort (z. B. "20 Millisievert") im Text steht."""
    if not stichworte:
        return None
    t = " ".join(falte(text).split())
    return any(" ".join(falte(s).split()) in t for s in stichworte)
