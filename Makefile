.PHONY: lint format check test

lint:
	ruff check src/ tests/ notebooks/
	flake8 src/ tests/ --max-line-length=120

format:
	ruff format src/ tests/ notebooks/

check: lint

test:
	pytest tests/ -v
