"""Runtime knobs for UniRL's VeOmni DeepEP high-throughput EP path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Mapping


class DeepEPHTConfigError(ValueError):
    """Raised before an unsupported DeepEP-HT configuration can run."""


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise DeepEPHTConfigError(f"{name} must be an integer, got {value!r}") from exc


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DeepEPHTConfigError(f"{name} must be a boolean value, got {value!r}")


@dataclass(frozen=True)
class DeepEPHTConfig:
    """Intranode DeepEP high-throughput buffer settings.

    Expert counts and EP size are taken from the live VeOmni mesh / module, not
    hard-coded here. The Qwen3-30B-A3B smoke is EP=4 / 128 experts / 32 local.
    """

    nvl_buffer_bytes: int = 200_000_000
    expert_alignment: int = 128
    num_sms: int = 24
    async_finish: bool = False
    single_node: bool = True
    mode: str = "ht"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DeepEPHTConfig":
        values = os.environ if env is None else env
        config = cls(
            nvl_buffer_bytes=_integer(values, "UNIRL_DEEPEP_NVL_BUFFER_BYTES", 200_000_000),
            expert_alignment=_integer(values, "UNIRL_DEEPEP_EXPERT_ALIGNMENT", 128),
            num_sms=_integer(values, "UNIRL_DEEPEP_NUM_SMS", 24),
            async_finish=_boolean(values, "UNIRL_DEEPEP_ASYNC_FINISH", False),
            single_node=_boolean(values, "UNIRL_DEEPEP_SINGLE_NODE", True),
            mode=values.get("UNIRL_DEEPEP_MODE", "ht").strip().lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.nvl_buffer_bytes <= 0:
            raise DeepEPHTConfigError("nvl_buffer_bytes must be positive")
        if self.expert_alignment <= 0:
            raise DeepEPHTConfigError("expert_alignment must be positive")
        if self.num_sms <= 0 or self.num_sms % 2:
            raise DeepEPHTConfigError("num_sms must be a positive even integer")
        if self.mode not in {"ht", "high_throughput", "high-throughput"}:
            raise DeepEPHTConfigError(
                f"UniRL DeepEP-HT requires high-throughput mode, not {self.mode!r}"
            )

    def manifest(self) -> dict[str, object]:
        return asdict(self)


def validate_runtime_contract(
    config: DeepEPHTConfig,
    *,
    ep_size: int,
    hidden_dtype: object,
    hidden_device_type: str,
    topk_dtype: object,
    routing_dtype: object,
    num_experts: int,
    gate_up_shape: tuple[int, ...],
    down_shape: tuple[int, ...],
) -> None:
    """Validate live tensors against the DeepEP-HT + local-shard contract."""

    config.validate()
    if ep_size < 2:
        raise DeepEPHTConfigError(f"DeepEP-HT requires ep_size>=2, got {ep_size}")
    if str(hidden_dtype) not in {"torch.bfloat16", "bfloat16", "bf16"}:
        raise DeepEPHTConfigError(f"hidden states must be BF16, got {hidden_dtype}")
    if hidden_device_type != "cuda":
        raise DeepEPHTConfigError(f"DeepEP-HT requires CUDA tensors, got {hidden_device_type!r}")
    if str(topk_dtype) not in {"torch.int64", "int64", "torch.int32", "int32"}:
        raise DeepEPHTConfigError(f"top-k IDs must be integer, got {topk_dtype}")
    if str(routing_dtype) not in {
        "torch.float32",
        "torch.bfloat16",
        "float32",
        "bfloat16",
    }:
        raise DeepEPHTConfigError(f"routing weights must be FP32 or BF16, got {routing_dtype}")
    if num_experts < 1 or num_experts % ep_size != 0:
        raise DeepEPHTConfigError(
            f"num_experts={num_experts} must be positive and divisible by ep_size={ep_size}"
        )
    local_experts = num_experts // ep_size
    if len(gate_up_shape) != 3 or gate_up_shape[0] != local_experts:
        raise DeepEPHTConfigError(
            f"gate_up_proj must be local [{local_experts},2I,H], got {gate_up_shape}"
        )
    if len(down_shape) != 3 or down_shape[0] != local_experts:
        raise DeepEPHTConfigError(
            f"down_proj must be local [{local_experts},H,I], got {down_shape}"
        )
    if gate_up_shape[1] % 2:
        raise DeepEPHTConfigError("gate_up_proj 2I dimension must be even")
    intermediate = gate_up_shape[1] // 2
    if down_shape[2] != intermediate or down_shape[1] != gate_up_shape[2]:
        raise DeepEPHTConfigError(
            f"local shards must be gate_up [E,2I,H] and down [E,H,I], got {gate_up_shape} and {down_shape}"
        )


__all__ = ["DeepEPHTConfig", "DeepEPHTConfigError", "validate_runtime_contract"]
