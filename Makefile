PY ?= python
WIDTH ?= 4
HEIGHT ?= 3
SUBGRID ?= 2
LOCAL ?= 4
OUT ?= output
SEED ?=
HOST ?= 127.0.0.1
PORT ?= 8765

.PHONY: demo dashboard test

demo:
ifdef SEED
	$(PY) main.py --width $(WIDTH) --height $(HEIGHT) --subgrid $(SUBGRID) --local $(LOCAL) --output $(OUT) --seed $(SEED) --debug
else
	$(PY) main.py --width $(WIDTH) --height $(HEIGHT) --subgrid $(SUBGRID) --local $(LOCAL) --output $(OUT) --debug
endif

dashboard:
	$(PY) dashboard.py --host $(HOST) --port $(PORT) --open

test:
	$(PY) run_tests.py
