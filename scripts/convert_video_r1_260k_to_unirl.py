#!/usr/bin/env python
"""Convert Video-R1-260k → UniRL jsonl (VIDEO-only, multiple-choice, letter GT).

Source: the HF dataset ``Video-R1/Video-R1-data``. Its ``Video-R1-260k.json`` is a
single list mixing image+video across 5 answer types; each row looks like::

    {"problem_id": 2,
     "problem": "What appears on the screen ...?",
     "data_type": "video",              # image | video
     "problem_type": "multiple choice", # multiple choice | free-form | numerical | OCR | regression
     "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
     "solution": "<answer>A</answer>",
     "path": "./LLaVA-Video-178K/.../ytb_xxx.mp4",
     "data_source": "LLaVA-Video-178K/30_60_s_youtube_v0_1"}

We keep ONLY ``data_type == "video"`` AND ``problem_type == "multiple choice"`` so
the letter ground_truth is scored by unirl.reward.local.mc_exact_match (A-D). The
mp4 ``path`` is resolved to an ABSOLUTE path under ``--data-root`` and written
straight into ``media_refs[].uri`` (unirl.data.datasets._resolve_media_uri returns
absolute URIs as-is → the Qwen3-Omni processor samples frames itself, fps-driven).

Output (under ``--out-dir``, default ``datasets/video_r1_260k``)::

    train.jsonl / val.jsonl   # {prompt, prompt_id, media_refs:[(video,prompt,uri)], metadata:{answer}}

No symlinks — uri is the absolute mp4 path, so the jsonl is location-independent.

PREREQUISITE — unzip the per-source video archives first (they ship as
``<Source>/<Source>_partN.zip`` and extract INTO ``<Source>/``)::

    ROOT=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/HF_Models/hub/\
datasets--Video-R1--Video-R1-data/snapshots/9ecf5eff38945e9ae4958058b83c9337f54aadd4
    for d in CLEVRER STAR NeXT-QA PerceptionTest LLaVA-Video-178K General Spatial; do
      for z in "$ROOT/$d"/*_part*.zip; do [ -f "$z" ] && unzip -o -q "$z" -d "$ROOT/$d"; done
    done

Rows whose mp4 is not on disk are skipped (so a PARTIAL download/unzip still yields
a usable jsonl). Use ``--keep-missing`` to emit them anyway.

Usage (video MC, cap ~20k, hold out 200 for val, only the harder reasoning sources)::

    python scripts/convert_video_r1_260k_to_unirl.py \
        --sources CLEVRER,STAR,NeXT-QA,PerceptionTest \
        --max-total 20000 --val-count 200
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
from typing import Dict, List, Optional

# Where the HF snapshot lives (contains Video-R1-260k.json + the per-source dirs).
DEFAULT_DATA_ROOT = (
    "/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/HF_Models/hub/"
    "datasets--Video-R1--Video-R1-data/snapshots/9ecf5eff38945e9ae4958058b83c9337f54aadd4"
)

_ANSWER_TAG = re.compile(r"<answer>\s*([A-Da-d])\s*</answer>")
_ANY_LETTER = re.compile(r"\b([A-Da-d])\b")


def _extract_letter(solution: str) -> Optional[str]:
    """Pull the A-D ground-truth letter out of ``solution`` (``<answer>D</answer>``)."""
    if not solution:
        return None
    m = _ANSWER_TAG.search(solution)
    if not m:
        m = _ANY_LETTER.search(solution)
    return m.group(1).upper() if m else None


def _build_prompt(problem: str, options: List[str]) -> str:
    """Question + labelled options + a reason-then-answer format directive.

    Qwen3-Omni Instruct ignores ``<think>...</think>`` tags in the prompt (its
    SFT distribution never saw them, so 75%+ of rollouts drop the opening tag).
    Ask for free-form reasoning instead, then a single ``<answer>X</answer>``
    line the scorer can lock onto. The scorer parses the ``<answer>`` tag
    strictly (see mc_exact_match._extract_answer_letter), so the reasoning
    body may mention option letters freely without polluting the reward.
    """
    opts = "\n".join(str(o).strip() for o in (options or []))
    return (
        f"{problem.strip()}\n{opts}\n"
        "First reason step by step about which option is correct. "
        "Then output the final answer letter (A, B, C, or D) on its own "
        "in the exact format:\n"
        "<answer>X</answer>"
    )


def _iter_rows(data_root: str, sources: Optional[set], keep_missing: bool):
    """Yield (source, abs_video_path, prompt, answer) for eligible video-MC rows."""
    with open(os.path.join(data_root, "Video-R1-260k.json"), encoding="utf-8") as f:
        data = json.load(f)

    stats = collections.Counter()
    for row in data:
        if row.get("data_type") != "video" or row.get("problem_type") != "multiple choice":
            continue
        rel = str(row.get("path", "")).lstrip("./")
        source = rel.split("/")[0] if rel else ""
        if sources and source not in sources:
            stats["filtered_source"] += 1
            continue
        answer = _extract_letter(str(row.get("solution", "")))
        if answer is None:
            stats["bad_answer"] += 1
            continue
        abs_path = os.path.join(data_root, rel)
        if not keep_missing and not os.path.isfile(abs_path):
            stats[f"missing:{source}"] += 1
            continue
        prompt = _build_prompt(str(row.get("problem", "")), row.get("options") or [])
        stats[f"kept:{source}"] += 1
        yield source, abs_path, prompt, answer, int(row.get("problem_id", -1))

    # stash stats on the generator's closure via attribute is awkward; print here.
    _iter_rows.last_stats = stats  # type: ignore[attr-defined]


def _write_split(rows: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {path}: {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="HF snapshot dir (holds Video-R1-260k.json)")
    ap.add_argument("--out-dir", default="datasets/video_r1_260k", help="output dataset dir")
    ap.add_argument(
        "--sources",
        default="",
        help="comma-separated source folders to KEEP (e.g. CLEVRER,STAR,NeXT-QA,PerceptionTest); empty = all",
    )
    ap.add_argument("--max-per-source", type=int, default=0, help="cap rows per source (0 = no cap)")
    ap.add_argument("--max-total", type=int, default=0, help="cap total kept rows (0 = no cap)")
    ap.add_argument("--val-count", type=int, default=200, help="hold out last N (post-shuffle) rows for val.jsonl")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed (deterministic split)")
    ap.add_argument("--keep-missing", action="store_true", help="emit rows even if the mp4 is not on disk yet")
    args = ap.parse_args()

    sources = {s.strip() for s in args.sources.split(",") if s.strip()} or None

    # Collect, applying per-source cap.
    per_source: Dict[str, List[Dict]] = collections.defaultdict(list)
    for source, abs_path, prompt, answer, pid in _iter_rows(args.data_root, sources, args.keep_missing):
        if args.max_per_source and len(per_source[source]) >= args.max_per_source:
            continue
        per_source[source].append(
            {
                "prompt": prompt,
                "prompt_id": f"video_r1_260k:{source}:{pid}",
                "media_refs": [{"modality": "video", "role": "prompt", "uri": abs_path}],
                "metadata": {"answer": answer},
            }
        )

    stats = getattr(_iter_rows, "last_stats", collections.Counter())
    kept = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("kept:")}
    missing = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("missing:")}
    print(f"[stats] kept per source: {dict(sorted(kept.items(), key=lambda kv: -kv[1]))}")
    if missing:
        print(f"[stats] skipped (mp4 not on disk) per source: {dict(sorted(missing.items(), key=lambda kv: -kv[1]))}")
        print("        → run the unzip step (see module docstring) or pass --keep-missing.")
    print(f"[stats] bad/no answer: {stats.get('bad_answer', 0)}")

    rows: List[Dict] = [r for rs in per_source.values() for r in rs]
    if not rows:
        raise SystemExit(
            "No usable video multiple-choice rows found. The videos are likely still "
            "zipped/downloading — unzip the per-source archives first (see docstring)."
        )

    random.Random(args.seed).shuffle(rows)
    if args.max_total and len(rows) > args.max_total:
        rows = rows[: args.max_total]

    n_val = max(0, min(int(args.val_count), len(rows) - 1))
    val_rows = rows[len(rows) - n_val :] if n_val else rows[:1]
    train_rows = rows[: len(rows) - n_val] if n_val else rows

    os.makedirs(args.out_dir, exist_ok=True)
    _write_split(train_rows, os.path.join(args.out_dir, "train.jsonl"))
    _write_split(val_rows, os.path.join(args.out_dir, "val.jsonl"))
    print(f"[done] {len(rows)} rows → {args.out_dir} (train={len(train_rows)}, val={len(val_rows)})")


if __name__ == "__main__":
    main()
