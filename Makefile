.PHONY: setup corpus ingest calibrate retrieve dev services down

setup:
	uv sync --python 3.11

corpus:
	uv run tokendiet fetch-corpus

ingest:
	uv run tokendiet ingest

calibrate:
	uv run tokendiet calibrate-tokenizer

# Usage: make retrieve Q="what are the principal risk factors?"
retrieve:
	uv run tokendiet retrieve "$(Q)"

# Optional: the deployed topology from the deck (Qdrant + Redis).
services:
	docker compose up -d

down:
	docker compose down

dev: setup corpus ingest
	@echo "Ready. Try: make retrieve Q=\"what are the principal risk factors?\""
