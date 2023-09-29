#!/bin/bash

set -e 

# Install development dependencies
poetry install --with dev

# Install pre-commit hooks & their virtual environments
pre-commit install --install-hooks

# Install Azure ML extension
az extension add -n ml -y

# Install Jupyter kernel
poetry run python -m ipykernel install --user
