#!/usr/bin/env python3
"""Find the first exact tensor divergence in Qwen3 stage dumps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

_NAME = re.compile(
    r"layer(?P<layer>\d+)\.call(?P<call>\d+)\.[^.]+\.(?P<stage>[^.]+)\.pt$"
)
_STAGE_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "layer_input",
            "attention_input",
            "q_proj",
            "k_proj",
            "v_proj",
            "q_norm",
            "k_norm",
            "q_rope",
            "k_rope",
            "v_attn",
            "attn_core_output",
            "oproj_input",
            "oproj_output",
            "attention_output",
            "router_logits",
            "router_topk_weights",
            "topk",
            "routing_weights",
            "moe_input",
            "canonical_dispatch",
            "fc1",
            "activation",
            "fc2",
            "rank_partial",
            "final",
            "moe_output",
            "layer_output",
        )
    )
}


def collect(directory: Path) -> dict[tuple[int, int, str], Path]:
    output = {}
    for path in directory.glob("*.pt"):
        match = _NAME.match(path.name)
        if match:
            key = (
                int(match.group("layer")),
                int(match.group("call")),
                match.group("stage"),
            )
            output[key] = path
    return output


def canonicalize(value: torch.Tensor) -> torch.Tensor:
    while value.ndim > 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    return value.contiguous()


def align(vllm: torch.Tensor, actor: torch.Tensor, stage: str):
    if stage in {
        "q_norm",
        "k_norm",
        "q_rope",
        "k_rope",
        "v_attn",
        "attn_core_output",
    }:
        if actor.ndim == 4 and actor.shape[0] == 1:
            actor = actor.squeeze(0)
            if actor.shape[1:] != vllm.shape[1:]:
                actor = actor.permute(1, 0, 2)
        elif (
            actor.ndim == 3
            and vllm.ndim == 3
            and actor.shape[0] == vllm.shape[1]
            and actor.shape[2] == vllm.shape[2]
        ):
            actor = actor.permute(1, 0, 2)
        if (
            stage in {"k_rope", "v_attn"}
            and actor.ndim == 3
            and vllm.ndim == 3
            and actor.shape[0] > vllm.shape[0]
        ):
            actor = actor[-1:]
        if (
            vllm.ndim == 3
            and actor.ndim == 3
            and vllm.shape[1:] == actor.shape[1:]
        ):
            vllm = vllm[: actor.shape[0]]
        return vllm.contiguous(), actor.contiguous()
    vllm = canonicalize(vllm)
    actor = canonicalize(actor)
    if vllm.ndim == actor.ndim and vllm.shape[1:] == actor.shape[1:]:
        if vllm.shape[0] >= actor.shape[0]:
            vllm = vllm[: actor.shape[0]]
    return vllm, actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vllm-rank", type=int, default=0)
    parser.add_argument(
        "--vllm-tp-size",
        type=int,
        default=None,
        help="Number of rank directories that are TP shards; use 1 for TP1 DP replicas.",
    )
    parser.add_argument("--actor-side", choices=("fsdp", "veomni"), default="fsdp")
    parser.add_argument("--actor-rank", type=int, default=0)
    parser.add_argument("--fsdp-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-stages", default="")
    parser.add_argument("--output")
    args = parser.parse_args()

    run = Path(args.root) / args.run_id
    vllm_root = run / "vllm"
    vllm_files_by_rank = {
        int(path.name.removeprefix("rank")): collect(path)
        for path in vllm_root.glob("rank*")
        if path.is_dir()
    }
    vllm_files = vllm_files_by_rank.get(args.vllm_rank, {})
    actor_rank = (
        args.fsdp_rank
        if args.fsdp_rank is not None and args.actor_side == "fsdp"
        else args.actor_rank
    )
    actor_files = collect(run / args.actor_side / f"rank{actor_rank:02d}")
    skipped_stages = {
        stage.strip()
        for stage in args.skip_stages.split(",")
        if stage.strip()
    }
    shared = {
        key
        for key in set(vllm_files) & set(actor_files)
        if key[2] not in skipped_stages
    }
    ordered = sorted(
        shared,
        key=lambda key: (
            key[1],
            key[0],
            _STAGE_ORDER.get(key[2], 999),
            key[2],
        ),
    )
    report = {
        "compared": 0,
        "skipped_shape": [],
        "first_divergence": None,
    }
    for layer, call, stage in ordered:
        reconstruct_tp = (
            args.vllm_tp_size
            if args.vllm_tp_size is not None
            else len(vllm_files_by_rank)
        )
        if stage in {
            "q_proj",
            "k_proj",
            "v_proj",
            "q_norm",
            "k_norm",
            "q_rope",
            "k_rope",
            "v_attn",
            "attn_core_output",
            "oproj_input",
        } and reconstruct_tp > 1:
            shards = [
                torch.load(files[(layer, call, stage)], map_location="cpu")
                for rank, files in sorted(vllm_files_by_rank.items())
                if rank < reconstruct_tp
                if (layer, call, stage) in files
            ]
            dim = (
                -2
                if stage
                in {
                    "q_norm",
                    "k_norm",
                    "q_rope",
                    "k_rope",
                    "v_attn",
                    "attn_core_output",
                }
                else -1
            )
            vllm_value = torch.cat(shards, dim=dim)
        else:
            vllm_value = torch.load(
                vllm_files[(layer, call, stage)],
                map_location="cpu",
            )
        vllm, actor = align(
            vllm_value,
            torch.load(actor_files[(layer, call, stage)], map_location="cpu"),
            stage,
        )
        if vllm.shape != actor.shape:
            report["skipped_shape"].append(
                {
                    "layer": layer,
                    "call": call,
                    "stage": stage,
                    "vllm": list(vllm.shape),
                    "actor": list(actor.shape),
                }
            )
            continue
        report["compared"] += 1
        if torch.equal(vllm, actor):
            continue
        different = (vllm != actor).reshape(-1)
        first = int(torch.nonzero(different, as_tuple=False)[0])
        report["first_divergence"] = {
            "layer": layer,
            "call": call,
            "stage": stage,
            "shape": list(vllm.shape),
            "different": int(different.sum()),
            "elements": int(different.numel()),
            "first_flat_index": first,
            "vllm_value": float(vllm.reshape(-1)[first]),
            "actor_value": float(actor.reshape(-1)[first]),
            "max_abs": float((vllm.float() - actor.float()).abs().max()),
            "vllm_file": str(vllm_files[(layer, call, stage)]),
            "actor_file": str(actor_files[(layer, call, stage)]),
        }
        break

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if report["first_divergence"] is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
