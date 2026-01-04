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
