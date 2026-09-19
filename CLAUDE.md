# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tradingo is a declarative task-graph framework for quantitative trading. Workflows are defined as YAML configs, executed as DAGs with ArcticDB-backed I/O, and dispatched locally (thread/process pool) or distributed (Celery). It handles data sampling, signal computation, portfolio construction, and order execution.

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file or specific test
uv run pytest test/tradingo/test_dag.py
uv run pytest test/tradingo/test_dag.py::test_function_name -v

# All code checks (linting, formatting, type checking)
uv run pre-commit run --all-files

# CLI — run a task
uv run tradingo-cli --config "<path/to/config.yaml>" task run "<task-name>" \
  --start-date "2025-01-01" --end-date "2025-01-15"

# CLI — run with parallelism and batching
uv run tradingo-cli --config "<path/to/config.yaml>" task run "<task-name>" \
  --with-deps 1 --n-workers 4 --batch-interval 1month --batch-mode stepped

# CLI — run with Celery distributed execution
uv run tradingo-cli --config "<path/to/config.yaml>" task run "<task-name>" \
  --executor celery --broker-url redis://localhost:6379/0

# CLI — list tasks in a config
uv run tradingo-cli --config "<path/to/config.yaml>" task list

# CLI — stop a running Celery execution plan
uv run tradingo-cli --config "<path/to/config.yaml>" task stop <plan_key>

# Monitor dashboard
uv run tradingo-monitor

# Trading execution (IB Gateway)
uv run tradingo-engine-ib
```

## Architecture

### Core Components

**`dag.py`** — Task graph engine:
- `Task` — wraps a Python function with config-driven I/O (via `symbol_provider`/`symbol_publisher`) and dependency resolution. State: `PENDING → SUCCESS | FAILED`.
- `Stage` — group of tasks executed sequentially in one process. Config syntax: `stage: [task.a, task.b, ...]`. Eliminates inter-task overhead (e.g. Celery round-trips) while still persisting intermediate ArcticDB writes.
- `DAG(dict[str, Task])` — the full task graph. Constructed via `DAG.from_config(config)`. Supports `DAG.run()` (sequential), `DAG.run_parallel()` (thread/process/Celery), and `DAG.update_state()` (reconcile with ArcticDB).
- `DAGRun` — live handle returned by `run_parallel(..., background=True)`. Exposes `.summary()`, `.steps()`, `.wait()`, `.stop()`.
- `BatchMode` — controls how time chunks are ordered vs. dependencies (`STEPPED`, `TASK`, `DEPS_FIRST`).

**`symbols.py`** — ArcticDB I/O decorators applied automatically by `Task.function`:
- `@symbol_provider(**symbols_in)` — injects DataFrames from ArcticDB into function args
- `@symbol_publisher(*symbols_out)` — writes function return value(s) to ArcticDB
- `@lib_provider` — injects library handles directly

**`api.py`** — `Tradingo` class: extends `arcticdb.Arctic` with a fluent dotted-namespace read API. `api.prices.im_multi_asset_3()` reads a symbol without knowing the raw key.

**`config.py`** — Jinja2 config loading. `read_config_template(path, variables)` handles `include:` directives and template substitution.

**`worker.py`** — Celery application (`tradingo.worker`). Exposes `run_tradingo_task` Celery task; handles task serialisation/deserialisation including `pd.Timestamp`, `re.Pattern`, and `Arctic` round-trips. Gracefully absent if `celery` not installed.

**`execution_plan.py`** — `ExecutionPlan` persists batched run state to `~/.tradingo/plans/`. Tracks per-step status (`PENDING`, `SUBMITTED`, `SUCCESS`, `FAILED`) and Celery task IDs for recovery via `--recover`.

**`settings.py`** — `TradingoConfig` and `IGTradingConfig` dataclasses for environment variable binding.

**`backtest.py`** — Cython-accelerated backtesting loop.

### Module Structure

```
src/tradingo/
├── api.py              # Tradingo ArcticDB wrapper (fluent namespace reads)
├── dag.py              # DAG, Task, Stage, DAGRun, BatchMode
├── config.py           # Jinja2 config loading with include: support
├── symbols.py          # @symbol_provider / @symbol_publisher decorators
├── settings.py         # TradingoConfig, IGTradingConfig env binding
├── worker.py           # Celery app and task serialisation
├── execution_plan.py   # Batched run state persistence (~/.tradingo/plans/)
├── cli.py              # tradingo-cli entrypoint
├── backtest.py         # Cython-accelerated backtesting
├── portfolio.py        # Portfolio construction helpers
├── plotting.py         # Visualisation utilities
├── utils.py            # Shared utilities
├── sampling/           # Data providers
│   ├── ig.py           # IG Markets OHLCV sampling
│   ├── yf.py           # Yahoo Finance sampling
│   ├── dukascopy.py    # Dukascopy tick data sampling
│   ├── ib.py           # Interactive Brokers sampling
│   ├── instruments.py  # Instrument metadata
│   └── quality.py      # Data quality checks
├── backfill/           # Historical data loaders
│   ├── dukascopy.py    # Dukascopy bulk backfill
│   └── forexsb.py      # ForexSB CSV backfill
├── engine/             # Live trading execution
│   ├── ig.py           # IG Markets order execution
│   └── ib.py           # Interactive Brokers execution (tradingo-engine-ib)
├── notifications/      # Alerting
│   └── email.py        # SMTP email notifications
└── templates/          # Bundled YAML config templates
    ├── instruments/     # Universe definition templates
    ├── downstream_tasks.yaml
    └── portfolio_construction.yaml
```

```
src/monitor/            # Plotly Dash portfolio dashboard (tradingo-monitor)
```

### CLI Flags Reference

`tradingo-cli --config <path> task run <task>`:

| Flag | Default | Description |
|------|---------|-------------|
| `--with-deps` | `False` | Run dependency tasks too (int = depth limit) |
| `--start-date` | — | Start of time window |
| `--end-date` | — | End of time window |
| `--force-rerun` | `True` | Re-run even if state is SUCCESS |
| `--dry-run` | — | Skip ArcticDB writes |
| `--clean` | — | Delete output symbols before running |
| `--skip-deps` | — | Regex pattern: skip matching dependency names |
| `--n-workers` | `1` | Parallel workers (thread/process pool) |
| `--batch-interval` | — | Chunk interval: `3days`, `2months`, `1hour` |
| `--batch-mode` | `stepped` | `stepped` / `task` / `deps-first` |
| `--recover` | — | Resume a failed batched run (skips SUCCESS steps) |
| `--executor` | `thread` | `thread`, `process`, or `celery` |
| `--broker-url` | `TP_CELERY_BROKER_URL` | Celery broker (for `--executor celery`) |

### Config Task Syntax

```yaml
my.task.name:
  depends_on: ["upstream.task"]         # DAG edges
  function: "module.path.function"      # Python callable
  symbols_in:                           # ArcticDB inputs → function args
    close: "prices/universe"
  symbols_out:                          # ArcticDB outputs (one per return value)
    - "signals/universe"
  params:                               # Extra kwargs passed to function
    speed1: 16

my.stage.name:                          # Stage: run tasks a, b, c in one process
  stage: [my.task.a, my.task.b]
  depends_on: ["upstream.task"]         # External deps (auto-computed if omitted)
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `TP_ARCTIC_URI` | Yes | — | ArcticDB connection URI |
| `TP_CONFIG_HOME` | Yes | — | Config directory |
| `TP_TEMPLATES` | No | bundled | Override bundled template path |
| `TP_CELERY_BROKER_URL` | Worker only | `redis://localhost:6379/0` | Celery broker |
| `TP_CELERY_RESULT_BACKEND` | Worker only | `redis://localhost:6379/1` | Celery result backend |
| `TP_CELERY_QUEUE` | Worker only | `tradingo` | Queue name |
| `IG_SERVICE_*` | IG only | — | IG Trading API credentials |
| `TRADINGO_LOG_CONFIG` | No | bundled | Logging YAML dictConfig path |

## Dependency Extras

```bash
uv sync                          # Core only
uv sync --group dev              # + pytest, mypy, pre-commit
uv sync --group worker           # + celery, redis
uv sync --group monitor          # + dash, gunicorn
uv sync --group serve            # + fastapi, croniter
uv sync --group ib               # + ib_insync
uv sync --group research         # + dtale, jupyter
```

## Type Checking

Project uses strict mypy. The `src/monitor/` directory is excluded. External libraries are allowlisted in `pyproject.toml`. Ignore Pyright diagnostics from the IDE — mypy is the authoritative checker.
