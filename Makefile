PROJECT_NAME = delius
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python$(PYTHON_VERSION)
ENV_NAME=delius_env
PIP_FILE=requirements.in
VENV_NAME=.venv

## Dump pip dependencies to file
.PHONY: export_env
export_env:
	pip freeze > $(PIP_FILE)
	pip-compile $(PIP_FILE)

## Update pip dependencies from file
.PHONY: pip_install
pip_install:
	pip install -U pip
	pip install -r $(PIP_FILE)

## Recreate the conda environment from file and install pip dependencies.
.PHONY: create_env
create_env:
	$(PYTHON_INTERPRETER) -m venv $(VENV_NAME)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
