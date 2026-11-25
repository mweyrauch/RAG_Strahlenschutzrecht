"""Anfrage-Umformulierung: Ein Sprachmodell uebersetzt eine alltagssprachliche Frage in
die Begriffe des Strahlenschutzrechts. Gesucht wird mit Original und Umformulierung;
die Ranglisten werden per Reciprocal Rank Fusion zusammengefuehrt.

Nicht in CI gegen ein echtes Modell getestet (nur mit MockBackend).
"""

from __future__ import annotations

import numpy as np

from .antwort import ChatBackend
from .chunking import Chunk
from .retrieval import Retriever

UMFORMULIER_PROMPT = (
    "Formuliere die folgende Frage so um, wie sie in der Fachsprache des deutschen "
    "Strahlenschutzgesetzes und der Strahlenschutzverordnung gestellt wuerde. Verwende die "
    "dort ueblichen Rechtsbegriffe (z. B. 'beruflich exponierte Person', 'effektive Dosis', "
    "'Strahlenschutzverantwortlicher'). Gib nur die umformulierte Frage aus, ohne Antwort "
    "und ohne Paragraphennummern."
)


class UmformulierRetriever(Retriever):
    def __init__(self, basis: Retriever, backend: ChatBackend, k_rrf: int = 60, tiefe: int = 100):
        self.basis = basis
        self.backend = backend
        self.k_rrf = k_rrf
        self.tiefe = tiefe
        self.name = f"umformuliert:{basis.name}"
        self.cache: dict[str, str] = {}

    def fit(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        self.basis.fit(self.chunks)
        return self

    def umformuliere(self, frage: str) -> str:
        if frage not in self.cache:
            self.cache[frage] = self.backend.chat([
                {"role": "system", "content": UMFORMULIER_PROMPT},
                {"role": "user", "content": frage},
            ]).strip()
        return self.cache[frage]

    def scores(self, frage):
        gesamt = np.zeros(len(self.chunks))
        for q in (frage, self.umformuliere(frage)):
            for rang, t in enumerate(self.basis.suche(q, self.tiefe), start=1):
                if t.score > 0:  # Treffer ohne jede Uebereinstimmung nicht mitzaehlen
                    gesamt[t.index] += 1.0 / (self.k_rrf + rang)
        return gesamt
