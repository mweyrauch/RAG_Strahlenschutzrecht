from pathlib import Path

import pytest

from strlrag.korpus import lese_gesetz

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def korpus():
    return lese_gesetz(FIXTURES / "testg.xml")
