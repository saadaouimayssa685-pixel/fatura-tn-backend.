# Render free evaluation profile

This profile is for evaluation, not durable invoice storage. Select Free explicitly.

- Branch: main
- Docker build context: repository root
- Dockerfile: backend/Dockerfile.free
- Health check: /health
- ALLOWED_ORIGINS: https://fatura-tn-app.vercel.app

The image excludes PyTorch, YOLO, PaddleOCR and local LLM weights. Tesseract,
PDF text extraction and hashing-based retrieval remain available. No semantic
embedding model is loaded. Disabled detection is recorded in the agent trace.
Memory use under real OCR workloads still needs measurement on the host.

Do not upload production invoices to this evaluation service. SQLite accounts,
dataset indexes and uploaded documents are local and disappear on restart.
DATABASE_URL alone does not migrate authentication or binary documents.
Before production use, migrate authentication to persistent storage, add durable
document storage, enforce authorization on document APIs and migrate existing data.
The local demo accounts must also be disabled for production.

The Vercel API URL must only be switched after the backend and persistence have
been tested. Never store credentials in Git or Docker build arguments.
