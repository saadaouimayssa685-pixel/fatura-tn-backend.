# Hosted file storage

PostgreSQL deployments store new invoice, document and RAG source files in
`stored_files` as BYTEA. Metadata references the SHA-256 digest; repeated
content is stored once. Downloads read the bytes directly from PostgreSQL.
SQLite deployments continue to use local files.

Limits default to 10 MiB per file (`MAX_FILE_SIZE`) and 100 MiB total file
content (`FILE_STORAGE_QUOTA_BYTES`). The quota excludes database overhead,
invoice metadata, and RAG chunks. It is not a guarantee about provider billing
or total database usage. This is intended for a small deployment, not bulk
import of the entire invoice dataset. Larger collections need object storage.

Existing local file references are not migrated automatically. They remain
downloadable only while their files exist. Reimport originals to store them
durably. Unreferenced blobs are retained and still consume quota; automatic
garbage collection is not implemented.

Unit tests use mocked database connections. A hosted upload/download test
across a redeploy remains necessary to verify the deployment end to end.
Authentication and per-user document authorization require a separate audit
before confidential invoices can be uploaded to a public instance.
