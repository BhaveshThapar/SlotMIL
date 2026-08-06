#!/usr/bin/env bash
# Write ~/.kaggle/kaggle.json from a username + API key.
#
# Kaggle's "Create New Token" button hands you a ready-made kaggle.json, but if
# you only copied the key string, the file is trivial to reconstruct -- it has
# exactly two fields.
#
# The key is read with `read -s`, so it is not echoed to the terminal and does
# not land in shell history. Usage:
#
#     bash scripts/setup_kaggle.sh
set -euo pipefail

DEST="$HOME/.kaggle/kaggle.json"

if [ -f "$DEST" ]; then
    read -rp "$DEST already exists. Overwrite? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "left unchanged"; exit 0; }
fi

# The username is the profile slug from kaggle.com/<username>, not your email.
read -rp "Kaggle username: " KAGGLE_USER
read -rsp "Kaggle API key:  " KAGGLE_KEY
echo

if [ -z "$KAGGLE_USER" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "both fields are required" >&2
    exit 1
fi

mkdir -p "$(dirname "$DEST")"
printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USER" "$KAGGLE_KEY" > "$DEST"
chmod 600 "$DEST"   # kaggle refuses to run on a world-readable token

echo "wrote $DEST"

PROJ=/fs/nexus-scratch/bthapar/SlotMIL
if [ -x "$PROJ/.venv/bin/python" ]; then
    "$PROJ/.venv/bin/python" - <<'PY'
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    print("kaggle auth: OK")
except ImportError:
    print("kaggle auth: not verified (pip install kaggle)")
except Exception as e:
    raise SystemExit(f"kaggle auth FAILED: {e}")
PY
fi
