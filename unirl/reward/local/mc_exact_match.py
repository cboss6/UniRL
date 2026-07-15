"""Multiple-choice exact-match reward scorer for VLM QA tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

_ANSWER_TAG = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)

_ANSWER_PATTERN = re.compile(
    r"(?:(?:answer|option)\s*(?:is|:)\s*)\(?([A-D])\)?",
    re.IGNORECASE,
)

_STANDALONE_LETTER = re.compile(r"\b([A-D])\b")


def _normalize_answer(answer: str) -> str:
    """Normalize answer to A/B/C/D letter.

    Handles: "A"/"B"/"C"/"D" → "A"/"B"/"C"/"D"
                 "1"/"2"/"3"/"4" → "A"/"B"/"C"/"D"
    """
    a = answer.strip().upper()
    # Numeric → letter
    if len(a) == 1 and a in "1234":
        return chr(ord("A") + ord(a) - ord("1"))
    # Already a letter
    if len(a) == 1 and a in "ABCD":
        return a
    return a  # fallback


def _extract_answer_letter(text: str) -> str:
    text = text.strip()
    # Handle numeric answers: "1"→"A", "2"→"B", "3"→"C", "4"→"D"
    if len(text) == 1 and text in "1234":
        return chr(ord("A") + ord(text) - ord("1"))
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    match = _ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    matches = _STANDALONE_LETTER.findall(text)
    if matches:
        return matches[-1].upper()
    return ""


def _extract_answer_letter_graded(text: str) -> Tuple[str, float]:
    """Extract A/B/C/D AND a format-quality weight.

    Returns ``(letter, format_weight)`` — the caller multiplies the weight in
    only when ``letter == gt`` (a wrong answer always scores 0):

      1. ``<answer>X</answer>``  → weight **1.0** (prompt-mandated format)
      2. any other extraction    → weight **0.5** ("answer is X", a bare letter,
         or a standalone A-D in free-form CoT)
      no A-D letter anywhere     → ``("", 0.0)``

    The full-weight ``<answer>`` tier nudges the model toward the mandated
    format while still giving partial credit to correct-but-unformatted replies.
    """
    text = text.strip()

    # 1. LAST <answer>X</answer> — the prompt-mandated format.
    tag_matches = _ANSWER_TAG.findall(text)
    if tag_matches:
        return tag_matches[-1].upper(), 1.0

    # 2. "answer is X" / "option: X" natural-language phrasing.
    m = _ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).upper(), 0.5

    # 3a. Bare single-letter / digit reply.
    if len(text) == 1 and text in "1234":
        return chr(ord("A") + ord(text) - ord("1")), 0.5
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper(), 0.5

    # 3b. Last standalone A/B/C/D anywhere in the text.
    matches = _STANDALONE_LETTER.findall(text)
    if matches:
        return matches[-1].upper(), 0.5

    return "", 0.0


class MCExactMatchRewardScorer(LocalRewardBackend):
    """Multiple-choice exact-match reward for VLM QA tasks."""

    canonical_model_name = "mc_exact_match"
    input_kind = "text"

    def __init__(self, *, config: "MCExactMatchSpec", base_device: str) -> None:
        del base_device
        super().__init__()
        # Graded format reward: when True, a CORRECT answer is scored by format
        # quality — <answer>X</answer> → 1.0, any other correct extraction → 0.5
        # (a wrong answer is always 0.0).
        self.graded_format_reward = bool(getattr(config, "graded_format_reward", False))

    def _load_model(self) -> None:
        self.model = "mc_exact_match"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("MCExactMatchRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            gt = _normalize_answer(str(meta["answer"]))
            if self.graded_format_reward:
                # Correct answer scored by format quality; wrong answer → 0.0.
                predicted, fmt_weight = _extract_answer_letter_graded(text)
                rewards.append(fmt_weight if predicted == gt else 0.0)
            else:
                predicted = _extract_answer_letter(text)
                rewards.append(1.0 if predicted == gt else 0.0)

        return rewards


@dataclass
class MCExactMatchSpec(BaseRewardComponentSpec):
    """Config for the MC exact-match scorer.

    ``graded_format_reward``: when True, a CORRECT answer is scored by how it was
    formatted — ``<answer>X</answer>`` → 1.0, any other correct extraction → 0.5;
    a wrong answer is always 0.0. Keeps a positive-but-smaller signal for
    correct-yet-unformatted replies instead of the hard 1/0 exact match.
    """

    graded_format_reward: bool = False
