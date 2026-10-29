#!/bin/bash
# The macOS equivalent of play.bat. Double-click it in Finder, or run:
#     bash mac/play.command
#     bash mac/play.command --strength 6
#     bash mac/play.command --delay 1 2
cd "$(dirname "$0")/.." || exit 1

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
    echo "No python3 found. Run mac/setup.command first."
    read -r -p "Press return to close." _
    exit 1
fi

echo "------------------------------------------------------------"
echo " Chess Agent"
echo "------------------------------------------------------------"
echo "  1. Open a game on your chess site"
echo "  2. Leave the board fully visible, nothing covering it"
echo "  3. Do not touch the mouse once it starts"
echo
echo "  Press F9 to stop, or Ctrl-C in this window."
echo "------------------------------------------------------------"
echo

"$PY" mac/run_mac.py calibrate --retry 400
if [ $? -ne 0 ]; then
    echo
    echo "Calibration failed."
    echo "The board must be within a couple of moves of the start of a game."
    echo "If a game is well underway, start a new one and run this again."
    echo
    echo "If every screenshot looks blank, Terminal is missing Screen Recording"
    echo "permission: System Settings > Privacy & Security > Screen Recording."
    echo
    read -r -p "Press return to close." _
    exit 1
fi

echo
"$PY" mac/run_mac.py play --think 0.4 "$@"

echo
read -r -p "Press return to close." _
