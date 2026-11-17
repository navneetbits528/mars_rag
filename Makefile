.PHONY: install db-up db-down migrate run test
install:
	uv sync --extra dev

db-up:
	docker compose up -d db

db-down:
	docker compose down

migrate:
	uv run python -m marsrag.db migrate

run:
	uv run uvicorn marsrag.api.main:app --reload

test:
	uv run pytest -q
