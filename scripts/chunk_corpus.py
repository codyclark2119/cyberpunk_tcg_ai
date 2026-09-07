#!/usr/bin/env python3
"""Chunk the rules corpus and card database for retrieval.

A thin CLI over `games.cyberpunk.chunking`, which holds the chunking policy
itself (why two chunk files, how rules are packed, what a card's retrievable
text looks like). Run after `scripts/ingest.py` and `scripts/build_cards.py`.

Usage:
    python scripts/chunk_corpus.py
    python scripts/chunk_corpus.py --max-chars 800
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk import chunking

if __name__ == "__main__":
    chunking.main()
