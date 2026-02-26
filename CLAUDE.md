# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tradingo is a declarative signal and portfolio construction framework for quantitative trading. It processes trading workflows as DAGs (Directed Acyclic Graphs), with support for multiple data providers (Yahoo Finance, IG Trading), backtesting, portfolio optimization, and trading execution.

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest test/tradingo/test_dag.py

# Run a specific test
uv run pytest test/tradingo/test_dag.py::test_function_name -v

# Linting and formatting
uv run black .
uv run isort .
uv run ruff check .
uv run mypy .

# Run pre-commit hooks
uv run pre-commit run --all-files

# CLI usage
uv run tradingo-cli --config "<path/to/config.yaml>" task run "<task-name>" --end-date "<date>"
```

## Architecture

### Core Components

- **DAG Engine (`dag.py`)**: Task orchestration with dependency resolution. `Task` wraps functions with config, I/O, and state management. `TaskState` tracks PENDING → FAILED/SUCCESS.

- **Symbol Management (`symbols.py`)**: Decorators for data I/O with ArcticDB:
  - `@symbol_provider`: Injects data from ArcticDB libraries into function args
  - `@symbol_publisher`: Writes function results to ArcticDB
  - `@lib_provider`: Injects library handles

- **Data Layer (`api.py`)**: `Tradingo` class extends ArcticDB.Arctic with hierarchical namespace support and fluent read API.

- **Configuration (`config.py`, `settings.py`)**: Jinja2 templating for YAML/JSON configs. `EnvProvider` dataclass for environment variable mapping.

### Pipeline Stages

1. **Universe**: Symbol definitions with timing/sampling intervals
2. **Prices**: Data sampling via provider templates (Yahoo Finance, IG Trading)
3. **Signals**: Custom signal computation (user-defined)
4. **Portfolio**: Position sizing and optimization via `riskfolio-lib`
5. **Trading**: Execution strategy and order placement

### Key Data Flow

Tasks read data via `symbols_in` (from ArcticDB libraries) and write via `symbols_out`. Configuration templates in `tradingo.templates` handle provider-specific sampling and portfolio construction.

### Module Structure

```
src/tradingo/
├── api.py              # Tradingo ArcticDB wrapper
├── dag.py              # DAG and Task classes
├── config.py           # Jinja2 config loading
├── symbols.py          # I/O decorators
├── backtest.py         # Cython-accelerated backtesting
├── portfolio.py        # Portfolio construction
├── sampling/           # Data providers (yf.py, ig.py)
└── templates/          # YAML config templates
```

## Environment Variables

Required for full functionality:
- `TP_CONFIG_HOME`: Configuration directory
- `TP_TEMPLATES`: Path to templates
- `TP_ARCTIC_URI`: ArcticDB connection URI
- `IG_SERVICE_*`: IG Trading API credentials (api_key, acc_type, username, password)

## Type Checking

Project uses strict mypy. Some external libraries are allowlisted for missing imports in `pyproject.toml`. The `src/monitor` directory is excluded from type checking.
