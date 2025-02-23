PROJECT_NAME = delius
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
ENV_NAME=delius_env
CONDA_FILE=environment.yml
PIP_FILE=requirements.txt

## Dump conda dependencies to file
.PHONY: export_conda
export_conda:
	conda env export --name $(ENV_NAME) > $(CONDA_FILE)

## Dump pip dependencies to file
.PHONY: export_pip
export_pip:
	pip freeze > $(PIP_FILE)

## Dump both pip and conda dependencies to different files
.PHONY: export_env
export_env: export_conda export_pip
	@echo "Exported conda environment to '$(CONDA_FILE)' and pip dependencies to '$(PIP_FILE)'"

## Update conda dependencies from file
.PHONY: conda_install
conda_install:
	conda env update --file $(CONDA_FILE) --prune

## Update pip dependencies from file
.PHONY: pip_install
pip_install:
	pip install -r $(PIP_FILE)

## Recreate the conda environment from file and install pip dependencies.
.PHONY: create_env
create_env:
	conda env create -f $(CONDA_FILE)
	pip_install

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
