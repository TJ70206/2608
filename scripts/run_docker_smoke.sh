#!/usr/bin/env bash
set -euo pipefail

python scripts/check_project.py
python scripts/check_demo_inputs.py
python scripts/check_html_demo.py
python scripts/check_pre_demo_readiness.py
python -m unittest discover -v

echo "Docker smoke checks passed."
