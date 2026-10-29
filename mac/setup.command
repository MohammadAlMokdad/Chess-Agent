#!/bin/bash
# Run this once on a Mac. Double-click it in Finder, or: bash mac/setup.command
cd "$(dirname "$0")/.." || exit 1

echo "------------------------------------------------------------"
echo " Chess Agent - macOS setup"
echo "------------------------------------------------------------"
echo

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
    echo "No python3 found. Install it with:  brew install python"
    echo "(or from https://www.python.org/downloads/)"
    read -r -p "Press return to close." _
    exit 1
fi
echo "Python: $($PY --version) at $(command -v $PY)"
echo

echo "Installing Python packages..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt || {
    echo "Package install failed."; read -r -p "Press return to close." _; exit 1; }

# pyautogui cannot even be imported on macOS without pyobjc -- it loads a
# Quartz-backed platform module and asserts out if that is missing.
echo "Installing pyobjc (required by pyautogui on macOS)..."
"$PY" -m pip install --quiet pyobjc-core pyobjc-framework-Quartz || {
    echo "  pyobjc failed to install - pyautogui will not import."
    echo "  Try:  $PY -m pip install pyobjc-core pyobjc-framework-Quartz"
    read -r -p "Press return to close." _; exit 1; }

# pynput is how the F9 kill switch works off Windows.
echo "Installing pynput (for the F9 kill switch)..."
"$PY" -m pip install --quiet pynput || echo "  pynput failed - F9 will not work, Ctrl-C still will."
echo

if [ -x "engine/stockfish" ]; then
    echo "Engine already present: engine/stockfish"
elif command -v stockfish >/dev/null 2>&1; then
    echo "Using Stockfish from PATH: $(command -v stockfish)"
else
    echo "Downloading Stockfish (universal macOS build, ~78 MB)..."
    mkdir -p engine
    URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-macos-universal.tar.gz"
    if curl -sL --fail -o engine/sf.tar.gz "$URL"; then
        # Extract into a scratch directory. The archive's top-level folder is
        # itself called "stockfish", so extracting in place and then moving
        # the binary to engine/stockfish just moves it back inside that
        # folder -- leaving a DIRECTORY where the executable should be, and a
        # baffling "Permission denied" when Python tries to run it.
        rm -rf engine/_unpack
        mkdir -p engine/_unpack
        tar -xzf engine/sf.tar.gz -C engine/_unpack

        FOUND=$(find engine/_unpack -type f -name "stockfish*" ! -name "*.gz" | head -1)
        if [ -n "$FOUND" ]; then
            rm -rf engine/stockfish            # may be a stale directory
            mv "$FOUND" engine/stockfish
            chmod +x engine/stockfish
            rm -rf engine/_unpack engine/sf.tar.gz

            if echo quit | ./engine/stockfish >/dev/null 2>&1; then
                echo "Engine installed and working: engine/stockfish"
            else
                echo "Engine installed but will not run. Try:  brew install stockfish"
            fi
        else
            echo "Could not find a binary inside the archive."
            echo "Alternative:  brew install stockfish"
            rm -rf engine/_unpack
        fi
    else
        echo "Download failed. Alternative:  brew install stockfish"
    fi
fi
echo

echo "Checking it all loads..."
"$PY" mac/run_mac.py selftest --games 2 2>&1 | tail -6
echo
echo "------------------------------------------------------------"
echo " IMPORTANT - macOS permissions"
echo "------------------------------------------------------------"
echo " System Settings > Privacy & Security, grant your TERMINAL app:"
echo "   * Screen Recording   (without it, screenshots come back blank)"
echo "   * Accessibility      (without it, the mouse will not move)"
echo
echo " Quit and reopen Terminal after granting them, or they do not apply."
echo "------------------------------------------------------------"
echo
echo "Setup done. Now run:  mac/play.command"
read -r -p "Press return to close." _
