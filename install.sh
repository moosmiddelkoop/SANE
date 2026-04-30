#!/bin/bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install "numpy<2" ray==2.6.1 pyarrow imageio
pip install -r requirements.txt
