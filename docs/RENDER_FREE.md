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

Configure DATABASE_URL as a secret before starting this profile. Authentication,
sessions and invoice metadata use PostgreSQL. Startup fails if PostgreSQL is
unavailable, and demo users are not created. Existing local accounts are not migrated.
Do not upload production invoices to this evaluation service: dataset indexes
and uploaded documents are still local and disappear on restart.
Before production use, add durable
document storage, enforce authorization on document APIs and migrate existing data.
Existing demo accounts in an imported database need separate review.

The Vercel API URL must only be switched after the backend and persistence have
been tested. Never store credentials in Git or Docker build arguments.
