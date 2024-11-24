VENV ?= venv
PYTHON ?= $(shell which python3)
PIP = $(shell which pip3)
PYTHON_FILES ?= $(shell find . -path ./venv -prune -o -name '*.py' -print)

# Tools
AUTOPEP8 ?= $(shell which autopep8)

.PHONY: venv
venv: # Create virtual environment
	@echo ">> setting up environment"
	$(PYTHON) -m venv $(VENV)

.PHONY: setup
setup: # Initial project setup
setup: requirements.txt
	$(PIP) install -r requirements.txt

.PHONY: build
build: # Build project
build: setup
	$(PIP) install --editable .

.PHONY: build-ui
build-ui: # Build UI
	cd ui && npm run build

.PHONY: fmt
fmt: # Format all python files
fmt: setup
	$(AUTOPEP8) --recursive --in-place --aggressive --aggressive $(PYTHON_FILES)

.PHONY: run-prom
run-prom: # Run prometheus
run-prom:
	@echo ">> running prometheus in docker at http://localhost:9090"
	@docker run -d --name prometheus -p 9090:9090 -v $(shell pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus