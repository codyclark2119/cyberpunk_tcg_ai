"""Generic embedding-index build/query over a chunked text corpus (RAG).

Brute-force cosine
similarity over an in-memory array -- fine for corpora up to roughly tens of
thousands of chunks; swap in a real vector index if a game's corpus outgrows
that.

Fully game-agnostic: operates on any jsonl of `{"chunk_id": ..., "text": ...}`
records. All paths and the embedding model id are parameters rather than
hardcoded constants, since a per-game module owns its own corpus location.
"""

import hashlib
from pathlib import Path

import numpy as np
from mlx_embeddings import generate, load

from harness.core.io import read_jsonl

DEFAULT_MODEL_ID = "mlx-community/all-MiniLM-L6-v2-4bit"


def fingerprint(chunks: list[dict]) -> str:
    """Identify the exact chunk set an index was built from.

    Re-chunking a corpus with different budgets rewrites the chunks file
    while leaving a stale index in place. Chunk ids are positional, so stale
    vectors still *resolve* -- retrieval would silently return text that was
    never embedded. Comparing a content hash turns that into an error instead.
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(c["text"].encode())
    return h.hexdigest()


def load_chunks(path: Path) -> list[dict]:
    return read_jsonl(path, missing_ok=False)


def embed_texts(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        output = generate(model, tokenizer, texts=batch)
        vectors.append(np.array(output.text_embeds))
    return np.concatenate(vectors, axis=0)


def build_index(chunks_path: Path, index_path: Path, model_id: str = DEFAULT_MODEL_ID) -> None:
    chunks = load_chunks(chunks_path)
    model, tokenizer = load(model_id)
    embeddings = embed_texts(model, tokenizer, [c["text"] for c in chunks])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        index_path,
        embeddings=embeddings,
        chunk_ids=np.array([c["chunk_id"] for c in chunks]),
        model_id=model_id,
        chunks_fingerprint=fingerprint(chunks),
    )
    print(f"embedded {len(chunks)} chunks ({embeddings.shape[1]}-dim) -> {index_path}")


def retrieve(
    query: str, k: int, chunks_path: Path, index_path: Path,
    model_id: str = DEFAULT_MODEL_ID, model_and_tokenizer: tuple | None = None,
) -> list[dict]:
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found -- build the index first")
    stored = np.load(index_path, allow_pickle=True)
    embeddings, chunk_ids = stored["embeddings"], stored["chunk_ids"]

    chunks = load_chunks(chunks_path)
    stored_model = str(stored["model_id"]) if "model_id" in stored else None
    if stored_model and stored_model != model_id:
        raise ValueError(
            f"{index_path} was built with {stored_model}, but this run uses {model_id}. "
            f"Vectors from different models are not comparable -- rebuild the index."
        )
    stored_fp = str(stored["chunks_fingerprint"]) if "chunks_fingerprint" in stored else None
    if stored_fp and stored_fp != fingerprint(chunks):
        raise ValueError(
            f"{index_path} is stale: {chunks_path} has changed since it was embedded. "
            f"Rebuild the index."
        )

    # Loading the embedding model takes real time, so a caller doing many
    # retrievals in a loop should load it once and pass it in.
    model, tokenizer = model_and_tokenizer if model_and_tokenizer else load(model_id)
    query_vec = embed_texts(model, tokenizer, [query])[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    scores = embeddings @ query_vec  # both sides pre-normalized -> cosine similarity
    top_k = np.argsort(-scores)[:k]

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    return [{**chunks_by_id[str(chunk_ids[i])], "score": float(scores[i])} for i in top_k]
