# Render free evaluation profile

This profile is for evaluation, not equivalent to the full local extraction
pipeline. Select Free explicitly. Do not upload confidential invoices until
document access controls have been verified.

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
With the PostgreSQL file store, new source files are stored as BYTEA in
`stored_files`; see `backend/FILE_STORAGE.md` for quotas and limitations.
Existing local document references and dataset indexes are not migrated
automatically and do not survive an ephemeral filesystem reset.
The current branch protects all document APIs, including reads and downloads.
It uses a single private workspace, not per-customer database isolation.
Set `WORKSPACE_ALLOWED_EMAILS` to an explicit comma-separated list of trusted,
existing FATURA accounts. Every listed account shares this workspace's data.
An empty list denies access; public signup is disabled. Known seeded demo
accounts cannot be authorized and are never seeded on PostgreSQL.
Do not merge/deploy until the intended owner's real account is identified and
the allowlist is configured. No account/password is created or migrated by this
change. If no real account exists, provision it privately before deployment;
do not enable public signup as a workaround.

The Vercel API URL must only be switched after the backend and persistence have
been tested. Never store credentials in Git or Docker build arguments.

## Deployment checklist (2026-10-19)

1. Sign in to the existing Render workspace. Locate `fatura-tn-backend` and
   record its deployed commit, instance plan, build context and Dockerfile.
   Do not create a replacement service or upgrade a plan automatically.
2. Identify the provider and plan for the existing `DATABASE_URL` using only
   its hostname. Keep the full connection string secret. Check storage usage,
   free-tier expiry and backup availability in that provider's dashboard.
3. Keep the current local SQLite databases and originals unchanged. Back up
   the hosted database before any migration. A SQLite backup alone does not
   contain source images/PDFs stored on local disk.
4. Configure `WORKSPACE_ALLOWED_EMAILS` for the existing trusted account(s).
   Block confidential-data migration until unauthenticated catalog, detail,
   download, RAG and database read routes are denied. Also verify registration
   policy and access between accounts. A login page alone is not sufficient.
5. Review the build context before deployment: no `.env` secrets, SQLite
   copies, `qa-results`, local uploads or logs. Ignore files do not untrack
   files already committed; inspect the actual Git/deployment contents.
6. Confirm the intended demo scope. `Dockerfile.free` disables PaddleOCR,
   YOLO and sentence-transformer embeddings. Do not present this profile as
   the validated local pipeline and do not silently deploy it as a substitute.
7. Only after the preceding checks, deploy the agreed backend version with
   secrets in Render environment settings, `REQUIRE_POSTGRES=true` and
   `SEED_DEMO_USERS=false`. Test login, denial without authentication, save,
   validation and source downloads with non-confidential test data.
8. Verify that test records and source files survive a redeploy. Then deploy
   the current frontend from the repository's `web/` directory, not the
   older `frontend/` directory. Set the Vercel project root to `web` when
   deploying this repository. Its
   `VITE_API_BASE_URL` must point to the verified HTTPS backend.
9. Migrate real SQLite data and source files only after explicit confirmation
   of the destination. Verify counts and document checksums, including human
   corrections and validation history. Never seed known demo credentials on
   the public instance.

Public checks on 2026-10-19: the Vercel frontend returned HTTP 200; Render
`/health` reported PostgreSQL and PostgreSQL BYTEA file storage, but no
PaddleOCR or YOLO. This does not verify the billing plan, database contents,
authorization, persistence across restarts, or equivalence with the local app.
The access-control changes on the review branch are not evidence that the
currently hosted instance is already secured. Repeat the checks after release.

## Offline access-control checks

```powershell
cd backend
python -m unittest test_workspace_access -v
```

These tests use in-memory handlers and sessions, never production accounts or
documents. They cover every declared API path without OCR/LLM calls. Browser
source previews fetch images/PDFs with a Bearer header and create temporary
blob URLs; session tokens are not included in download URLs.

Provider references (recheck before changing plans):
- https://render.com/docs/free
- https://vercel.com/docs/plans/hobby
- https://neon.com/pricing
