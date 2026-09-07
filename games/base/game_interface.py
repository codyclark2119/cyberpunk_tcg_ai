"""The contract every game module implements to plug into the generic harness.

This is deliberately a plain-data + callable protocol rather than a class
hierarchy requiring inheritance -- a game module builds one `GameConfig` and
passes it to `harness.core` functions, rather than subclassing anything.

A new game (e.g. One Piece TCG) implements this by providing:

  - Corpus paths: where its rules text, card database, and chunked/derived
    files live (see `data/<game>/...` convention in the top-level README).
  - A retrieval config: chunk path + embedding index path (`harness.core.rag`
    works unmodified once these are supplied).
  - Prompt builders: `build_messages(question, context=None) -> list[dict]`,
    used identically for training-data assembly and for inference, so the two
    can never drift apart (see `harness/core/sft/build.py` and
    `harness/core/eval/harness.py`).
  - A judge system prompt (or prompts) naming the game and its rules
    convention, passed to `harness.core.eval.judge.judge_batch_rubric`.
  - A gold-data schema: this template ships a single generic schema (see
    `data/gold/SCHEMA.md`) intended to be used as-is per game, rather than
    each game inventing its own fields -- keep new fields additive and
    document them there if a game genuinely needs one.
  - Gameplay hooks (if the game has a "board state" eval track): an action
    grammar/parser, a position renderer, and a legality checker. There is no
    generic implementation of these -- `games/<name>/gameplay/` is 100%
    game-specific (its own `actions.py`, `positions.py`, etc.).

This module intentionally contains no game-specific code. It is the seam:
fill in `games/<name>/` to stand up a new game.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class RetrievalConfig:
    chunks_path: Path
    index_path: Path
    embed_model_id: str = "mlx-community/all-MiniLM-L6-v2-4bit"


@dataclass
class GameConfig:
    """Everything `harness.core` needs to run the generic pipeline for one game."""

    name: str                              # e.g. "one_piece"
    data_dir: Path                         # data/<name>/
    models_dir: Path                       # models/<name>/

    retrieval: Optional[RetrievalConfig] = None

    # (question: str, context: str | None) -> chat messages, EXCLUDING the
    # final assistant turn. Used identically by SFT dataset assembly and by
    # eval-time generation.
    build_messages: Optional[Callable[[str, Optional[str]], list]] = None

    # System prompt(s) for the rubric judge. Must contain the literal phrase
    # "labeled A, B, C, D" wherever it names the candidate labels, so
    # `harness.core.eval.judge.judge_prompt_for` can rewrite it for other
    # arm counts.
    judge_system_prompt: str = ""

    # Optional: a regex-or-callable that extracts citation ids (e.g. rule
    # numbers) from an answer, for citation-validity scoring. None if the
    # game has no citation convention.
    extract_citations: Optional[Callable[[str], set]] = None

    extra: dict = field(default_factory=dict)  # game-specific escape hatch
