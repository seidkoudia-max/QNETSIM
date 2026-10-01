#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/qnetsim"
python -m streamlit run quasar_gui.py --server.port 8504
