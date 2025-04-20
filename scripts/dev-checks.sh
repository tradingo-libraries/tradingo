# alike a github workflow, run before commit
echo Pytest tradingo
uv run pytest
echo Formatting check tradingo
uv run black ./ --check --config pyproject.toml
echo Import check tradingo
uv run isort ./ --check
echo Ruff check tradingo
uv run ruff check --config pyproject.toml ./
echo Type check tradingo
uv run mypy --config pyproject.toml ./src
