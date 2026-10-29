"""macOS entry point. Same commands as run.py, with the shim loaded first.

    python3 mac/run_mac.py detect
    python3 mac/run_mac.py calibrate
    python3 mac/run_mac.py play

The shim must be imported before anything from chessagent, so the import
order in this file is deliberate and should not be tidied.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import macshim                       # noqa: E402,F401  -- must be first

sys.path.insert(0, macshim.ROOT)

from run import main                 # noqa: E402


if __name__ == "__main__":
    if "--quiet-shim" not in sys.argv:
        print(macshim.describe())
        print()

    # Stop here rather than let the failure surface as a traceback from deep
    # inside asyncio, which names the symptom and not the cause. Commands
    # that never touch the engine are still allowed through.
    needs_engine = any(cmd in sys.argv for cmd in ("play", "watch", "loopback"))
    if needs_engine and macshim.engine_problem():
        sys.exit(1)

    sys.exit(main())
