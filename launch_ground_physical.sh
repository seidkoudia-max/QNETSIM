#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/qnetsim"
python -m streamlit run ground_physical_studio.py --server.port 8505
