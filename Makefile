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

# Usage: make ab Q="..."   (needs GROQ_API_KEY in .env)
ab:
	uv run tokendiet ab "$(Q)"

web-build:
	cd web && npm install && npm run build

api:
	uv run uvicorn tokendiet.api:app --host 127.0.0.1 --port 8000

dev: setup corpus ingest web-build
	@echo "Ready. Run 'make api' and open http://127.0.0.1:8000"
