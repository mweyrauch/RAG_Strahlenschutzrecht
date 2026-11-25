"""Antworterzeugung aus abgerufenen Abschnitten.

Backends:
- ExtraktivBackend:      ohne Sprachmodell; waehlt die zur Frage passendsten Saetze
                         aus den Quellen und zitiert sie. Dient als nachvollziehbare
                         Basislinie und fuer Tests.
- OpenAIKompatibelBackend: beliebiger Server mit /v1/chat/completions-Schnittstelle
                         (z. B. Ollama, LM Studio, vLLM, OpenAI). Nicht in CI getestet.
- MockBackend:           feste Antwort, fuer Tests.

Zusaetzlich: Enthaltung bei zu schwachem Retrieval und eine Pruefung, ob die
im Antworttext zitierten Quellennummern tatsaechlich im Kontext vorkommen.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .retrieval import Retriever, Treffer
from .text import Analysator, saetze

KEINE_GRUNDLAGE = "Dazu finde ich in den Quellen keine Grundlage."

SYSTEM_PROMPT = (
    "Du beantwortest Fragen zum deutschen Strahlenschutzrecht. "
    "Nutze ausschliesslich die nummerierten Quellen. Belege jede Aussage mit der "
    "Quellennummer in eckigen Klammern, z. B. [2]. Wenn die Quellen die Frage nicht "
    f"beantworten, antworte genau: '{KEINE_GRUNDLAGE}' Antworte knapp auf Deutsch. "
    "Dies ist keine Rechtsberatung."
)


@dataclass
class Antwort:
    frage: str
    text: str
    quellen: list[Treffer] = field(default_factory=list)
    enthalten: bool = False  # True = bewusst nicht beantwortet

    @property
    def zitierte_nummern(self) -> list[int]:
        return sorted({int(n) for n in re.findall(r"\[(\d+)\]", self.text)})

    @property
    def ungueltige_zitate(self) -> list[int]:
        """Quellennummern im Text, die es im Kontext nicht gibt."""
        return [n for n in self.zitierte_nummern if not 1 <= n <= len(self.quellen)]

    def formatiert(self) -> str:
        zeilen = [self.text, ""]
        for i, t in enumerate(self.quellen, start=1):
            zeilen.append(f"[{i}] {t.chunk.quelle}")
        return "\n".join(zeilen).rstrip()


def kontext_block(treffer: list[Treffer]) -> str:
    return "\n\n".join(f"[{i}] {t.chunk.quelle}\n{t.chunk.text}" for i, t in enumerate(treffer, start=1))


def baue_nachrichten(frage: str, treffer: list[Treffer]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Quellen:\n\n{kontext_block(treffer)}\n\nFrage: {frage}"},
    ]


class Backend:
    def antworte(self, frage: str, treffer: list[Treffer]) -> str:  # pragma: no cover
        raise NotImplementedError


class ChatBackend(Backend):
    """Backend mit einem Sprachmodell; chat() nimmt Nachrichten und gibt Text zurueck."""

    def chat(self, nachrichten: list[dict]) -> str:  # pragma: no cover - abstrakt
        raise NotImplementedError

    def antworte(self, frage, treffer):
        return self.chat(baue_nachrichten(frage, treffer))


class MockBackend(ChatBackend):
    def __init__(self, text: str = "Testantwort [1]."):
        self.text = text
        self.letzte_nachrichten: list[dict] | None = None

    def chat(self, nachrichten):
        self.letzte_nachrichten = nachrichten
        return self.text


class ExtraktivBackend(Backend):
    """Waehlt bis zu n Saetze mit der hoechsten TF-IDF-Aehnlichkeit zur Frage."""

    def __init__(self, n_saetze: int = 2):
        self.n_saetze = n_saetze

    def antworte(self, frage, treffer):
        kandidaten: list[tuple[int, str]] = []
        for i, t in enumerate(treffer, start=1):
            for s in saetze(t.chunk.text):
                kandidaten.append((i, s))
        if not kandidaten:
            return KEINE_GRUNDLAGE
        vec = TfidfVectorizer(analyzer=Analysator(stemming=True), sublinear_tf=True)
        try:
            m = vec.fit_transform([s for _, s in kandidaten] + [frage])
        except ValueError:  # nur Stoppwoerter
            return KEINE_GRUNDLAGE
        sim = (m[:-1] @ m[-1].T).toarray().ravel()
        if sim.max() <= 0:
            return KEINE_GRUNDLAGE
        # leichte Bevorzugung hoeher gerankter Quellen bei Gleichstand
        rang = np.array([i for i, _ in kandidaten], dtype=float)
        ordnung = np.lexsort((rang, -sim))[: self.n_saetze]
        ordnung = sorted(ordnung, key=lambda j: (kandidaten[j][0], j))
        return " ".join(f"{kandidaten[j][1]} [{kandidaten[j][0]}]" for j in ordnung)


class OpenAIKompatibelBackend(ChatBackend):
    """Ruft einen Chat-Completions-Endpunkt auf.

    Umgebungsvariablen: LLM_BASE_URL (Standard http://localhost:11434/v1 fuer Ollama),
    LLM_MODEL, LLM_API_KEY (optional).
    """

    def __init__(self, base_url: str | None = None, modell: str | None = None,
                 api_key: str | None = None, temperatur: float = 0.0, timeout: float = 120):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.modell = modell or os.environ.get("LLM_MODEL", "llama3.1:8b")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.temperatur = temperatur
        self.timeout = timeout

    def chat(self, nachrichten):  # pragma: no cover - braucht Netzwerk
        import requests

        kopf = {"Content-Type": "application/json"}
        if self.api_key:
            kopf["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.modell, "temperature": self.temperatur,
                  "messages": nachrichten},
            headers=kopf, timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def beantworte(frage: str, retriever: Retriever, backend: Backend, k: int = 5,
               min_score: float | None = None) -> Antwort:
    """Retrieval + Antwort. Bei min_score wird unterhalb der Schwelle enthalten."""
    treffer = retriever.suche(frage, k)
    if not treffer or (min_score is not None and treffer[0].score < min_score):
        return Antwort(frage, KEINE_GRUNDLAGE, treffer, enthalten=True)
    text = backend.antworte(frage, treffer)
    return Antwort(frage, text, treffer, enthalten=text.strip() == KEINE_GRUNDLAGE)
