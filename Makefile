install-deps:
	brew install portaudio

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check --fix --select I .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix .

.PHONY: check
check: lint
	uv run ruff format --check .

.PHONY: test
test:
	uv run pytest

.PHONY: test-verbose
test-verbose:
	uv run pytest -v

.PHONY: test-watch
test-watch:
	uv run pytest-watch
