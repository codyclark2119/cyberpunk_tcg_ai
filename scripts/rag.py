"""CLI: embed a chunked corpus and query it.

Usage:
    python scripts/rag.py index --chunks data/processed/chunks.jsonl --out data/processed/chunk_embeddings.npz
    python scripts/rag.py query "..." --chunks data/processed/chunks.jsonl --index data/processed/chunk_embeddings.npz [--k 5]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.core.rag.index import DEFAULT_MODEL_ID, build_index, retrieve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    idx_p = sub.add_parser("index", help="Embed all chunks and build the vector store")
    idx_p.add_argument("--chunks", type=Path, required=True)
    idx_p.add_argument("--out", type=Path, required=True)
    idx_p.add_argument("--model-id", default=DEFAULT_MODEL_ID)

    q_p = sub.add_parser("query", help="Retrieve the top-k most relevant chunks for a query")
    q_p.add_argument("text")
    q_p.add_argument("--k", type=int, default=5)
    q_p.add_argument("--chunks", type=Path, required=True)
    q_p.add_argument("--index", type=Path, required=True)
    q_p.add_argument("--model-id", default=DEFAULT_MODEL_ID)

    args = parser.parse_args()

    if args.command == "index":
        build_index(args.chunks, args.out, model_id=args.model_id)
    else:
        for r in retrieve(args.text, args.k, args.chunks, args.index, model_id=args.model_id):
            print(f"[{r['score']:.3f}] {r['chunk_id']}")
            print(r["text"][:300].replace("\n", " "))
            print()


if __name__ == "__main__":
    main()
