"""Launch the bundled causal light-transport explorer."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nerf_lightpath_explorer.causal_explorer.server import main


if __name__ == "__main__":
    main()
