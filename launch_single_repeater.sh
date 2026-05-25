#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/qnetsim"
python -m streamlit run paper_figure_studio.py --server.port 8501
