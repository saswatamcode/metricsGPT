VENV ?= venv
PYTHON ?= $(shell which python3)
PIP = $(shell which pip3)
PYTHON_FILES ?= $(shell find . -path ./venv -prune -o -name '*.py' -print)

# Tools
AUTOPEP8 ?= $(shell which autopep8)

.PHONY: setup
setup: # Initial project setup
setup: requirements.txt
	@echo ">> setting up environment"
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

.PHONY: build
build: # Build project
build: setup
	$(PIP) install --editable .

.PHONY: fmt
fmt: # Format all python files
fmt: setup
	$(AUTOPEP8) --recursive --in-place --aggressive --aggressive $(PYTHON_FILES)
