PY ?= python
WIDTH ?= 4
HEIGHT ?= 3
SUBGRID ?= 2
LOCAL ?= 4
OUT ?= output
SEED ?=

.PHONY: demo
demo:
ifdef SEED
	$(PY) main.py --width $(WIDTH) --height $(HEIGHT) --subgrid $(SUBGRID) --local $(LOCAL) --output $(OUT) --seed $(SEED) --debug
else
	$(PY) main.py --width $(WIDTH) --height $(HEIGHT) --subgrid $(SUBGRID) --local $(LOCAL) --output $(OUT) --debug
endif

