[![CI:main](https://github.com/EliaGroup-Innersource/codespaces-template/actions/workflows/ci.yaml/badge.svg)](https://github.com/EliaGroup-Innersource/codespaces-template/actions/workflows/ci.yaml) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

# Quick start

## Requirements

- [Poetry](https://python-poetry.org/)
- [Python 3.11](https://www.python.org/downloads/release/python-3110/)

## Setup

Setup instructions only apply if working on a local machine. 

If working in GitHub Codespaces, development environment is automatically configured.

Configure and activate virtual environment:

```console
poetry env use /path/to/python3.11
poetry shell
```

Install dependencies:

```console
poetry install
```

Install with optional development dependencies:

```console
poetry install --with dev
```

Install pre-commit hooks:

```console
pre-commit install
```

## Dependency management

Add a package:
```console
poetry add <package>
```

Add a package as an optional development dependency:
```console
poetry add <package> --group dev
```

Remove a package:
```console
poetry remove <package>
```

Any poetry command that adds/removes/updates dependencies modifies `pyproject.toml` and `poetry.lock` files. Therefore, these files need to be commited to the repository in order to guarantee a reproducible environment.