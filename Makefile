PYTHON ?= python
VENV ?= .venv
IMAGE ?= travel-intent
DATASET ?= data/intent-detection-train.jsonl
TRAIN_DATASET ?= data/splits/train_expanded.jsonl
EVAL_DATASET ?= data/splits/train.jsonl
TEXT ?= Je recherche un vol
MODEL ?= classic

ifeq ($(OS),Windows_NT)
	VENV_PYTHON = $(VENV)/Scripts/python
else
	VENV_PYTHON = $(VENV)/bin/python
endif

.PHONY: venv install prepare-data augment-data train predict evaluate app docker-build docker-train docker-predict docker-evaluate docker-app

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

prepare-data:
	$(VENV_PYTHON) -m src.cli.main prepare-data --dataset $(DATASET)

augment-data:
	$(VENV_PYTHON) -m src.cli.main augment-data --dataset data/splits/train.jsonl

train:
	$(VENV_PYTHON) -m src.cli.main train --model $(MODEL) --dataset $(TRAIN_DATASET)

predict:
	$(VENV_PYTHON) -m src.cli.main predict --model $(MODEL) --text "$(TEXT)"

evaluate:
	$(VENV_PYTHON) -m src.cli.main evaluate --model $(MODEL) --dataset $(EVAL_DATASET)

app:
	$(VENV_PYTHON) -m streamlit run apps/app.py

docker-build:
	docker build -t $(IMAGE) .

docker-train: docker-build
	docker run --rm \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/reports:/app/reports \
		-v $(CURDIR)/logs:/app/logs \
		$(IMAGE) python -m src.cli.main train --model $(MODEL) --dataset /app/$(TRAIN_DATASET)

docker-predict: docker-build
	docker run --rm \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/logs:/app/logs \
		$(IMAGE) python -m src.cli.main predict --model $(MODEL) --text "$(TEXT)"

docker-evaluate: docker-build
	docker run --rm \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/reports:/app/reports \
		-v $(CURDIR)/logs:/app/logs \
		$(IMAGE) python -m src.cli.main evaluate --model $(MODEL) --dataset /app/$(EVAL_DATASET)

docker-app: docker-build
	docker run --rm -p 8501:8501 \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/reports:/app/reports \
		-v $(CURDIR)/logs:/app/logs \
		$(IMAGE)
