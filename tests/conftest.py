from pathlib import Path

import pytest

from strlrag.antwort import ChatBackend
from strlrag.korpus import lese_gesetz

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def korpus():
    return lese_gesetz(FIXTURES / "testg.xml")


class MockBackend(ChatBackend):
    """Sprachmodell-Ersatz fuer Tests: gibt einen festen Text zurueck und merkt sich den Prompt."""

    def __init__(self, text: str = "Testantwort [1]."):
        self.text = text
        self.letzte_nachrichten: list[dict] | None = None

    def chat(self, nachrichten):
        self.letzte_nachrichten = nachrichten
        return self.text
