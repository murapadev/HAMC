PY ?= python
WIDTH ?= 4
HEIGHT ?= 3
SUBGRID ?= 2
LOCAL ?= 4
OUT ?= output
SEED ?= 42
HOST ?= 127.0.0.1
PORT ?= 8765

.PHONY: demo dashboard test lint

demo:
	$(PY) -m hamc generate --width $(WIDTH) --height $(HEIGHT) --subgrid $(SUBGRID) --local $(LOCAL) --seed $(SEED) -o $(OUT)

dashboard:
	$(PY) -m hamc dashboard --host $(HOST) --port $(PORT) --open

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
