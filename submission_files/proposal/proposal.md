# Stage 1 — Proposal: Multilingual RAG Retrieval Performance Analysis

## 1. Description of the Computational Object

**Computational object:** A **Retrieval-Augmented Generation (RAG)** system evaluated in a multilingual setting. The system comprises:

- **Document base:** A fixed set of base documents in text format (10 documents), used as the sole knowledge source for retrieval.
  
  The same **10 documents** (same content, same structure) are translated into **20 different languages** and indexed; retrieval and RAG experiments are run on this multilingual corpus so that the impact of language on embedding quality, retrieval performance, and answer quality can be compared under controlled conditions.
- **Embedding model:** `intfloat/multilingual-e5-large`, used to encode documents and queries into a common vector space.
- **Vector store:** FAISS (Facebook AI Similarity Search) index storing embeddings with metadata (document ID, language).
- **Retrieval:** Top-*k* nearest-neighbour search in the FAISS index given a query embedding.
- **Chat LLM:** Qwen2.5 7B (via Ollama), which receives the retrieved context and the user query to produce the final answer.
- **Translator LLM:** Gemini 1.5 Pro.

**Pipeline:** Documents → Translation → Embeddings → Vector index → Query → Retrieval → LLM response.

**Comparative focus:** All metrics are recorded **per language** so that they can be **compared across languages** for the same content. For example: embedding generation time for a document in Portuguese vs. the *same* document in English; retrieval latency, embedding dimensionality, storage size, and retrieval/RAG quality are measured for each language version and then compared (e.g. in tables and plots). The controlled factor is language; the document set and structure are identical across the 20 language versions.

**Justification of complexity:** The object is complex enough to study real-world trade-offs (embedding quality, retrieval speed, memory, answer quality) and the effect of language on each stage, but remains tractable: all components run locally (no external APIs) and the pipeline is well-defined.

---

## 2. Choice of Analysis Method

**Method:** **Measurement**.

**Environment:**

- **Hardware:** Intel i5-12400F (CPU), NVIDIA GeForce RTX 4060 (GPU), 32 GB RAM, 1 TB NVMe SSD (Kingston NV2).
- **Execution:** All models and experiments run **locally** (no external APIs).

---

## 3. Definition of Metrics and Methodology

Each metric below is measured **per language** (for the same documents and queries in each language version). Results are then **compared across languages** (e.g. Portuguese vs. English vs. Russian) via tables and plots to assess the impact of language on performance, efficiency, and quality.

### 3.1 Performance metrics (recording and calculation)

| Metric | Description | Recording / calculation |
|--------|-------------|-------------------------|
| **Embedding generation time** | Time to encode each document with the chosen model. | Measure with a high-resolution timer around the encode call; report mean and, if relevant, standard deviation over repeated runs. Unit: seconds (or ms). |
| **Retrieval latency** | Time from query embedding to returning top-*k* document IDs from the FAISS index. | Timer around the index search call. Report mean (and dispersion) over the evaluation query set. Unit: seconds (or ms). |
| **Memory usage** | RAM and GPU consumed by the embedding model, FAISS index, and/or full RAG process. | Use OS or runtime tools to sample memory before/after loading the model and index and during runs. Unit: MB. |

### 3.2 Efficiency metrics

| Metric | Description | Recording / calculation |
|--------|-------------|-------------------------|
| **Embedding dimensionality** | Size of each embedding vector. | Read from the model/output shape (e.g. `embedding.shape[1]`). Unit: dimension count. |
| **Storage size** | Disk space used by the FAISS index and any serialised embeddings or metadata. | Measure file size(s) of the index and related files. Unit: KB or MB. |

### 3.3 RAG answer quality

| Metric | Description |
|--------|-------------|
| **Relevance (retrieval)** | Whether the retrieved documents are relevant to the query. For a defined set of queries (~20) with known relevant document(s), record which document(s) are in top-*k*; evaluate relevance manually |
| **Completeness** | Whether the answer addresses all parts of the question. | Manual rating (e.g. full / partial / missing). |
| **Relevance** | Whether the answer stays on-topic and uses the provided context. | Manual rating |
