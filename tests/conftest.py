import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[3]
CORE_PATH = ROOT / 'packages' / 'accuralai-core'
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))
@pytest.fixture
def anyio_backend():
    return 'asyncio'
