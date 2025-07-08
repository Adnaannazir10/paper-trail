# Ingestion Pipeline Design (Pseudocode Only)

This document sketches the end‑to‑end logic for transforming newly uploaded journal PDFs into search‑ready vectors stored in **Qdrant**.

---

## 1  Detect new journal files

| Source                              | Strategy                                                                          | Notes                                                         |
| ----------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Local/Network folder                | Filesystem watcher (`watchdog` / `inotifywait`) fires on **create/modify** events | Store `(path, mtime, sha256)` in manifest to skip duplicates. |
| Cloud object store (S3, GCS, Azure) | Object‑created notification → queue message                                       | Serverless; no polling.                                       |
| Google Drive / SharePoint / Dropbox | Webhook → n8n / Zapier automation → queue ingest job                              | Low‑code, business‑friendly; platform notifies instantly.     |
| Any other location                  | Hourly cron list → diff against manifest table                                    | Fallback when events aren’t available.                        |

**Manifest schema**\
`file_id (PK) ▸ revision_tag ▸ sha256_hash ▸ status ▸ ingested_at`

---

## 2  Chunk the content & attach metadata

1. Extract text using a PDF‑parsing framework (e.g., **PyMuPDF**, pdfplumber) that preserves headings.
2. Split by heading where possible; otherwise fall back to a token‑based splitter (e.g., `tiktoken`) with a **chunk\_size = 256 tokens**\*\* with a **chunk\_overlap = 64 tokens** (≈ 25 % of the chunk size)\*\* to preserve context across boundaries.
3. Yield **dictionaries** that already include the chunk text **and** all required metadata fields (`id`, `source_doc_id`, `chunk_index`, `section_heading`, `journal`, `publish_year`, `usage_count`, `attributes`, `link`).

**Example resulting chunk:**

```json
{
  "id": "mucuna_01_intro",
  "source_doc_id": "extension_brief_mucuna.pdf",
  "chunk_index": 1,
  "section_heading": "Velvet bean description",
  "journal": "ILRI extension brief",
  "publish_year": 2016,
  "usage_count": 42,
  "attributes": [
    "Botanical description",
    "Morphology"
  ],
  "link": "https://cgspace.cgiar.org/server/api/core/bitstreams/68bfaec0-8d32-4567-9133-7df9ec7f3e23/content",
  "text": "Velvet bean–Mucuna pruriens var. utilis, also known as mucuna—is a twining annual leguminous vine common to most parts of the tropics. Its growth is restricted to the wet-season as it dies at the onset of the cold season. It has large trifoliate leaves (i.e. has three leaflets) and very long vigorous twining stems that can extend over two–three metres depending on growth conditions. When planted at the beginning of the growing season, flowers normally form at the end of March/early April. These flowers are deep purple and appear underneath the foliage. Seeds are large, ovoid shaped (±10 mm long) and of different colours, ranging from white, grey, brown to black and mottled."
}
```

---

## 3  Generate embeddings

```python
EMBED_MODEL = "avsolatorio/NoInstruct-small-Embedding-v0"
vectors = embedder.batch_encode([c.text for c in chunks])
```

**Why this model?**

- *Open source & permissive*: no vendor lock‑in, easy to self‑host.
- *Lightweight*: < 100 M parameters → small memory footprint and fast inference.
- *384‑d vectors*: lower storage cost in Qdrant while retaining strong semantic quality.
- *Leaderboard‑proven*: ranked #1 on the Hugging Face embedding leaderboard for models under 100 M parameters, so we get top performance per parameter.

---

---

## 5  Upsert into Qdrant

```python
qdrant.points.upsert(
    collection_name="journal_chunks",
    points=[
        {
            "id": meta["id"],
            "vector": vec,
            "payload": meta
        } for vec, meta in zip(vectors, metadatas)
    ]
)
```

*Key is deterministic* (`source_sha256:order_idx`) → idempotent re‑ingestion.

---

## 6  Full pipeline pseudocode

```pseudo
function ingest_file(file_url, schema_version):
    path, sha = download(file_url)
    if manifest.exists(path, sha):
        return  # duplicate
    text = extract_text(path)
    chunks = split(text, strategy=schema_version)
    vectors = embed(chunks)
        qdrant_upsert(vectors, chunks)
    manifest.record(file_id=path, revision=sha, status="ingested")
```

---

## Why Qdrant?

- **Open‑source with hosting options** – start self‑hosted, migrate to Qdrant Cloud later with almost zero code change.
- **Easy to host** – drop‑in Docker container; straightforward deploys on AWS, Azure, GCP, or bare‑metal.
- **Developer speed** – single‑Docker image for local dev; clean Python client.
- **Rich payload filtering** – complex nested JSON filters let us slice by `journal`, `publish_year`, etc.
- **HNSW defaults** – good p95 latency (< 25 ms on 1 M points) without manual tuning.

This design keeps ingestion idempotent, horizontally scalable, and ready for millions of chunks while leveraging Qdrant’s straightforward API and filtering power.

