# Semantic Visual Search Engine

Search an image collection with plain-language text or with another image. Images are embedded with OpenCLIP, stored in Qdrant, and retrieved by cosine similarity. An optional LLM step (Anthropic Claude) rewrites chatty queries such as *"show me a red flower please"* into short caption-style text that CLIP matches better.

## Features

- **Text → image search**: natural-language query, rewritten by Claude into a CLIP-friendly caption.
- **Image → image search**: upload a picture and find visually similar ones.
- **Folder ingestion**: walks a directory tree, embeds images in batches, and uses each sub-folder name as the image's `category`.
- **REST API** (FastAPI) and a **Streamlit UI** on top of it.
- Optional saving of retrieved results to disk.
- Structured logging (`structlog`) and a custom exception type across the backend.

## How it works

```
                 ┌─────────────── ingestion ───────────────┐
 images/<category>/*.jpg ──► OpenCLIP image encoder ──► Qdrant (512-d, cosine)
                                                              ▲
 text query ──► Claude rewrite ──► OpenCLIP text encoder ─────┤  top-k
 query image ─────────────────► OpenCLIP image encoder ───────┘
```

Defaults: model `ViT-B-32`, checkpoint `laion2b_s34b_b79k`, vector size 512.

## Project layout

```
.
├── images/                       # Sample dataset (animal, flower, general, ...)
├── data/                         # Query uploads and saved results (query_images/, retrieved/)
├── semantic_visual_search/
│   ├── backend/                  # The working implementation
│   │   ├── config.py             # Env-driven settings (Config)
│   │   ├── embeddings.py         # OpenCLIP text/image embedding
│   │   ├── qdrant_client.py      # Qdrant client singleton + collection setup
│   │   ├── ingestion.py          # IndexService: single image / folder indexing
│   │   ├── retriever.py          # ImageSearchService: text & image search
│   │   ├── query_translator.py   # Claude query rewriter
│   │   ├── main.py               # FastAPI app
│   │   ├── logger/, exception/   # Logging and error handling
│   ├── ui/app.py                 # Streamlit front end
│   ├── notebooks/                # Experiments, model comparison, Recall@K evaluation
│   ├── src/, tests/, infra/, config/   # Planned modular layout (currently placeholders)
├── pyproject.toml
└── requirements.txt
```

## Getting started

### Prerequisites

- Python (the project pins 3.14 in `.python-version` / `pyproject.toml`; the dependencies should also work on recent earlier versions, but that is untested here)
- A [Qdrant](https://qdrant.tech/) instance (Qdrant Cloud or self-hosted) and its API key
- An [Anthropic API key](https://console.anthropic.com/) (only needed for text queries, which are rewritten by Claude)

### Install

```bash
python -m venv env
# Windows: env\Scripts\activate    macOS/Linux: source env/bin/activate
python -m pip install -r requirements.txt
```

Use the same interpreter for everything (`python -m pip`, `python -m uvicorn`, `python -m streamlit`) so the venv's Python is the one that runs.

`requirements.txt` ends with `-e .`, which installs the package from `pyproject.toml`, so it needs a Python version that satisfies `requires-python`. If that blocks you, remove that line and run everything from the repository root so the `semantic_visual_search` package is importable.

### Configure

Create a `.env` file in the repository root:

```env
QDRANT_URL=https://<your-cluster>.qdrant.io
QDRANT_API_KEY=<your-qdrant-key>
ANTHROPIC_API_KEY=<your-anthropic-key>

# Optional (defaults shown)
QDRANT_COLLECTION=semantic-visual-search
CLIP_MODEL_NAME=ViT-B-32
CLIP_CHECKPOINT=laion2b_s34b_b79k
VECTOR_SIZE=512
DEVICE=cpu                     # or cuda
ANTHROPIC_MODEL=claude-sonnet-4-6
IMAGES_ROOT=./images
QUERY_IMAGE_ROOT=./data/query_images
RETRIEVED_ROOT=./data/retrieved
```

If you change `CLIP_MODEL_NAME`, set `VECTOR_SIZE` to match its embedding dimension.

### Run

From the repository root, start the API:

```bash
python -m uvicorn semantic_visual_search.backend.main:app --reload
```

Interactive docs are at http://localhost:8000/docs. Then start the UI in a second terminal:

```bash
python -m streamlit run semantic_visual_search/ui/app.py
```

The first start downloads the CLIP weights (a few hundred MB).

Index your images once before searching:

```bash
curl -X POST "http://localhost:8000/ingest"
# or a custom folder
curl -X POST "http://localhost:8000/ingest?folder_path=path/to/images"
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`. Each image's category is the name of the folder it sits in.

## API

| Method | Endpoint        | Description |
|--------|-----------------|-------------|
| POST   | `/ingest`       | Index a folder (`folder_path`, defaults to `IMAGES_ROOT`). |
| GET    | `/translate`    | Show how Claude rewrites a query (`q`). |
| GET    | `/search-text`  | Text search (`q`, `k=5`, `category`, `save_results`). |
| POST   | `/search-image` | Image search; multipart `file` plus `k`, `category`, `save_results`. |

Example:

```bash
curl "http://localhost:8000/search-text?q=a%20big%20cat%20with%20a%20mane&k=3"
```

```json
{
  "query": "a big cat with a mane",
  "translated": "Lion with a thick mane",
  "k": 3,
  "saved_folder": null,
  "results": [
    { "filename": "lion.jpeg", "path": "images/animal/lion.jpeg", "category": "animal", "score": 0.31 }
  ]
}
```

(Values above are illustrative.)

## Troubleshooting

- **`pip.exe` blocked by an Application Control policy (Windows):** run pip through Python instead: `python -m pip install -r requirements.txt`.
- **Package tries to build from source / `Cython ... blocked`:** your Python is newer than the available wheels. Create the venv with Python 3.14 (`py -3.14 -m venv env --clear`) and check with `python --version`.
- **`DLL load failed while importing cygrpc` (or another `.pyd`) blocked:** Application Control is blocking an unsigned native extension from a very new release. Install an older version of that package, for example `python -m pip install "grpcio==1.76.0"`, or ask the machine's administrator to allow the project folder.
- **`torch_python.dll` blocked (`WinError 4551`):** the newest torch (2.15) was blocked on the development machine, so `requirements.txt` pins `torch==2.12.1` and `torchvision==0.27.1`. If torch is blocked on yours, try another version. Stop the API server first, because a running server locks the DLLs and the install fails with `Access is denied`. Restart the server after any failed torch import, since torch cannot be re-imported in the same process.
- **`WinError 10054` / connection reset when contacting Qdrant:** the cluster may be suspended or deleted (free clusters are removed after inactivity), or the network blocks it. Test with `curl.exe https://<your-cluster-host>/healthz`. Any response, even `{"error":"forbidden"}`, means the host is reachable. A new cluster needs a new `QDRANT_URL` and `QDRANT_API_KEY`, and the images must be re-ingested.

## Notebooks

`semantic_visual_search/notebooks/` contains `experiments.ipynb`, `model_comparison.ipynb` and `evaluate_recall_at_k.ipynb`, plus sample retrieved results.

## Status and known limitations

This is a work in progress.

- The `category` filter is built in `retriever.py` but not yet passed to `query_points`, so filtering by category currently has no effect.
- `clear_collection` calls `client.delete(..., filter=...)`, which does not match the Qdrant client API and should be reviewed before use.
- `src/`, `tests/`, `infra/` (Docker, Kubernetes, CI/CD, Nginx) and `config/*.yaml` are empty placeholders for planned work: modular services, tests, containerisation and deployment.
- The Streamlit UI expects the API at `http://localhost:8000` (hard-coded in `ui/app.py`) and loads result images from local paths, so it must run on the same machine as the API.
- The API has no authentication or rate limiting.

## Tech stack

Python · FastAPI · Streamlit · OpenCLIP (via LangChain) · Qdrant · Anthropic Claude (LangChain) · Pillow · structlog

## Author

Nikhil Srivastava
