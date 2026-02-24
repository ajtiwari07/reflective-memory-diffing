from __future__ import annotations

import asyncio
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
src_path = root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from reflective_memory_diffing_agent.app import main  # noqa: E402


if __name__ == "__main__":
    asyncio.run(main())
