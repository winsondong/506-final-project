# Makefile

DATA ?= ./airbnbListingsData.csv
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) src/train.py --data $(DATA) --out models/best_rf.joblib

vis: install
	$(VENV)/bin/jupyter lab notebooks/

test: install
	$(VENV)/bin/pytest -q

clean:
	rm -rf $(VENV) .pytest_cache __pycache__ models reports

help:
	@echo "Usage:"
	@echo "  make install    - Create venv and install dependencies"
	@echo "  make run        - Run full pipeline (preprocess + train + evaluate)"
	@echo "  make vis        - Open Jupyter Lab for data exploration"
	@echo "  make test       - Run test suite"
	@echo "  make clean      - Remove all generated files"
