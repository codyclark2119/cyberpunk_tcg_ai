"""The Cyberpunk TCG `GameConfig` -- wires this game's corpus, prompts, and
retrieval into the generic `harness.core` engine.

See `games/base/game_interface.py` for the contract this fulfills. Import
`GAME` from this module wherever a script needs "the current game's config",
scoped to this one game module rather than shared across games.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.base.game_interface import GameConfig, RetrievalConfig
from games.cyberpunk import prompts

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models" / "cyberpunk"

RULES_PATH = DATA_DIR / "processed" / "rules.jsonl"
RULE_SECTIONS_PATH = DATA_DIR / "processed" / "rule_sections.jsonl"
RULES_CHUNKS_PATH = DATA_DIR / "processed" / "rule_chunks.jsonl"
CARDS_CHUNKS_PATH = DATA_DIR / "processed" / "card_chunks.jsonl"
CARD_DATABASE_PATH = DATA_DIR / "processed" / "card_database.jsonl"

# The single index this game retrieves over. Unlike the One Piece instance of
# this template, which needs two independently-budgeted indexes because card
# chunks outnumber rule chunks 48:1, this corpus is 2:1 and a plain top-k over
# one merged index still returns a mix on a rules question. The per-source
# chunk files above are still written, so the dual-budget router can be added
# later without re-shaping the corpus -- see `games/cyberpunk/chunking.py` for
# the ratio at which that becomes necessary.
CORPUS_CHUNKS_PATH = DATA_DIR / "processed" / "corpus_chunks.jsonl"
CORPUS_INDEX_PATH = DATA_DIR / "processed" / "corpus_index.npz"

RETRIEVAL = RetrievalConfig(
    chunks_path=CORPUS_CHUNKS_PATH,
    index_path=CORPUS_INDEX_PATH,
)

GAME = GameConfig(
    name="cyberpunk",
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    retrieval=RETRIEVAL,
    build_messages=prompts.build_messages,
    judge_system_prompt=prompts.JUDGE_SYSTEM_PROMPT,
    extract_citations=prompts.extract_citations,
    extra={
        "rules_path": RULES_PATH,
        "rule_sections_path": RULE_SECTIONS_PATH,
        "rules_chunks_path": RULES_CHUNKS_PATH,
        "cards_chunks_path": CARDS_CHUNKS_PATH,
        "card_database_path": CARD_DATABASE_PATH,
        "rules_version": prompts.RULES_VERSION,
    },
)
