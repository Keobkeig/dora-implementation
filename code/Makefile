.PHONY: format test clean

format:
	uv run black .
	uv run ruff check . --fix

test: format
	uv run pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf __pycache__ demo/__pycache__ demo/tests/__pycache__
	rm -rf dora/__pycache__ dora/layers/__pycache__ dora/models/__pycache__ dora/utils/__pycache__
	rm -rf scripts/__pycache__ tests/__pycache__ tests/unit/__pycache__ training/__pycache__
	rm -rf htmlcov .coverage coverage.xml
