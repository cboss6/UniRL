"""DeepEP high-throughput dispatch/combine with explicit autograd."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import socket
import threading

import torch

from unirl.train.backend.veomni.ep.deepep_ht.config import (
    DeepEPHTConfig,
    DeepEPHTConfigError,
    validate_runtime_contract,
)

logger = logging.getLogger(__name__)


class DeepEPBackendError(RuntimeError):
    """A fail-fast DeepEP API or runtime contract error."""


@dataclass
class _BufferEntry:
    buffer: object
    dispatch_config: object
    combine_config: object
    ep_size: int


class DeepEPBufferCache:
    """Persistent buffers keyed by process group, device, and model shape."""

    def __init__(self) -> None:
        self._entries: dict[tuple[object, ...], _BufferEntry] = {}
        self._lock = threading.Lock()

    def get(
        self,
        *,
        group: object,
        device: torch.device,
        num_experts: int,
        hidden: int,
        topk: int,
        config: DeepEPHTConfig,
    ) -> _BufferEntry:
        key = (
            id(group),
            device.type,
            device.index,
            num_experts,
            hidden,
            topk,
        )
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry
            try:
                import deep_ep
                import torch.distributed as dist
            except Exception as exc:
                raise DeepEPBackendError(
                    "ep_comm_backend=deepep_ht requires an importable DeepEP build "
                    "with high-throughput Buffer.dispatch/combine"
                ) from exc
            try:
                ep_size = dist.get_world_size(group)
                deep_ep.Buffer.set_num_sms(config.num_sms)
                buffer = deep_ep.Buffer(
                    group,
                    config.nvl_buffer_bytes,
                    0,
                    low_latency_mode=False,
                    explicitly_destroy=True,
                )
                dispatch_config = deep_ep.Buffer.get_dispatch_config(ep_size, hidden * 2)
                combine_config = deep_ep.Buffer.get_combine_config(ep_size, hidden * 2)
            except Exception as exc:
                raise DeepEPBackendError(
                    "failed to create the persistent intranode DeepEP-HT "
                    f"buffer for {device}, (E={num_experts},H={hidden},K={topk})"
                ) from exc
            if bool(getattr(buffer, "low_latency_mode", False)):
                raise DeepEPBackendError("DeepEP Buffer unexpectedly entered low-latency mode")
            entry = _BufferEntry(buffer, dispatch_config, combine_config, ep_size)
            self._entries[key] = entry
            return entry

    def clear(self) -> None:
        """Destroy every explicit DeepEP buffer; idempotent and best effort."""

        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            destroy = getattr(entry.buffer, "destroy", None)
            if destroy is None:
                continue
            try:
                destroy()
            except Exception:
                logger.exception("Failed to destroy a DeepEP-HT buffer during shutdown")


_BUFFER_CACHE = DeepEPBufferCache()


def close_deepep_ht_buffers() -> None:
    """Release process-local native DeepEP buffers before PG teardown."""

    _BUFFER_CACHE.clear()


@dataclass
class _Invocation:
    entry: _BufferEntry | None = None
    handle: object | None = None
    recv_shape: tuple[int, int] | None = None
    recv_device: torch.device | None = None
    num_experts: int = 0


def _wait_event(event: object, enabled: bool) -> None:
    if not enabled:
        return
    wait = getattr(event, "current_stream_wait", None)
    if wait is None:
        raise DeepEPBackendError("DeepEP async_finish returned no current-stream-waitable event")
    wait()


class _DeepEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dispatcher: "DeepEPDispatcher",
        invocation: _Invocation,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ):
        recv_x, recv_ids, recv_weights = dispatcher._dispatch(
            invocation, hidden_states, topk_indices, routing_weights
        )
        ctx.dispatcher = dispatcher
        ctx.invocation = invocation
        ctx.routing_dtype = routing_weights.dtype
        return recv_x, recv_ids, recv_weights

    @staticmethod
    def backward(ctx, grad_recv_x, _grad_recv_ids, grad_recv_weights):
        grad_hidden, grad_routing = ctx.dispatcher._reverse_dispatch(
            ctx.invocation, grad_recv_x, grad_recv_weights
        )
        if grad_routing is not None:
            grad_routing = grad_routing.to(ctx.routing_dtype)
        return None, None, grad_hidden, None, grad_routing


class _DeepEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dispatcher: "DeepEPDispatcher",
        invocation: _Invocation,
        partial: torch.Tensor,
    ) -> torch.Tensor:
        ctx.dispatcher = dispatcher
        ctx.invocation = invocation
        return dispatcher._combine(invocation, partial)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            None,
            ctx.dispatcher._reverse_combine(ctx.invocation, grad_output),
        )


class DeepEPDispatcher:
    """Validate and execute the differentiable DeepEP-HT sequence."""

    def __init__(self, config: DeepEPHTConfig) -> None:
        config.validate()
        self.config = config
        self._validated_groups: set[int] = set()
        self._logged_ready = False

    @staticmethod
    def _distributed():
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise DeepEPBackendError("an initialized VeOmni EP process group is required")
        return dist

    def _validate_single_node(self, group: object, dist: object, ep_size: int) -> None:
        if not self.config.single_node:
            return
        key = id(group)
        if key in self._validated_groups:
            return
        hostnames: list[str | None] = [None] * ep_size
        dist.all_gather_object(hostnames, socket.gethostname(), group=group)
        if len(set(hostnames)) != 1:
            raise DeepEPHTConfigError(f"UniRL DeepEP-HT is single-node only; hosts={hostnames}")
        self._validated_groups.add(key)

    def validate_inputs(
        self,
        *,
        group: object,
        num_experts: int,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> None:
        dist = self._distributed()
        ep_size = dist.get_world_size(group)
        validate_runtime_contract(
            self.config,
            ep_size=ep_size,
            hidden_dtype=hidden_states.dtype,
            hidden_device_type=hidden_states.device.type,
            topk_dtype=topk_indices.dtype,
            routing_dtype=routing_weights.dtype,
            num_experts=num_experts,
            gate_up_shape=tuple(gate_up_proj.shape),
            down_shape=tuple(down_proj.shape),
        )
        if hidden_states.dim() != 2 or topk_indices.dim() != 2:
            raise DeepEPHTConfigError("expected hidden [tokens,H] and top-k [tokens,K]")
        if routing_weights.shape != topk_indices.shape:
            raise DeepEPHTConfigError("routing weights and top-k IDs must match")
        if topk_indices.shape[0] != hidden_states.shape[0]:
            raise DeepEPHTConfigError("hidden and routing token counts differ")
        if bool(((topk_indices < 0) | (topk_indices >= num_experts)).any()):
            raise DeepEPHTConfigError(f"top-k entries must be global IDs in [0,{num_experts})")
        if gate_up_proj.device != hidden_states.device or down_proj.device != hidden_states.device:
            raise DeepEPHTConfigError("hidden states and local expert shards must share one CUDA device")
        self._validate_single_node(group, dist, ep_size)

    def _entry(
        self,
        group: object,
        hidden_states: torch.Tensor,
        num_experts: int,
        topk: int,
    ) -> _BufferEntry:
        return _BUFFER_CACHE.get(
            group=group,
            device=hidden_states.device,
            num_experts=num_experts,
            hidden=hidden_states.shape[-1],
            topk=topk,
            config=self.config,
        )

    def _dispatch(
        self,
        invocation: _Invocation,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ):
        if invocation.entry is None:
            raise DeepEPBackendError("dispatch invocation has no buffer")
        entry = invocation.entry
        asynchronous = self.config.async_finish
        layout = entry.buffer.get_dispatch_layout(
            topk_indices,
            invocation.num_experts,
            async_finish=asynchronous,
            allocate_on_comm_stream=asynchronous,
        )
        (
            tokens_per_rank,
            tokens_per_rdma_rank,
            tokens_per_expert,
            token_in_rank,
            layout_event,
        ) = layout
        result = entry.buffer.dispatch(
            x=hidden_states,
            num_tokens_per_rank=tokens_per_rank,
            num_tokens_per_rdma_rank=tokens_per_rdma_rank,
            is_token_in_rank=token_in_rank,
            num_tokens_per_expert=tokens_per_expert,
            topk_idx=topk_indices,
            topk_weights=routing_weights.float(),
            expert_alignment=self.config.expert_alignment,
            config=entry.dispatch_config,
            previous_event=layout_event if asynchronous else None,
            async_finish=asynchronous,
            allocate_on_comm_stream=asynchronous,
        )
        recv_x, recv_ids, recv_weights, _counts, handle, event = result
        _wait_event(event, asynchronous)
        if recv_ids is None or recv_weights is None or handle is None:
            raise DeepEPBackendError("DeepEP HT dispatch did not return top-k metadata and a handle")
        invocation.handle = handle
        invocation.recv_shape = tuple(recv_x.shape)
        invocation.recv_device = recv_x.device
        return recv_x, recv_ids, recv_weights

    def _combine(self, invocation: _Invocation, partial: torch.Tensor):
        if invocation.entry is None or invocation.handle is None:
            raise DeepEPBackendError("combine called before dispatch")
        if partial.dtype != torch.bfloat16:
            raise DeepEPBackendError("DeepEP combine requires a BF16 local partial")
        output, _weights, event = invocation.entry.buffer.combine(
            x=partial,
            handle=invocation.handle,
            config=invocation.entry.combine_config,
            async_finish=self.config.async_finish,
            allocate_on_comm_stream=self.config.async_finish,
        )
        _wait_event(event, self.config.async_finish)
        return output

    def _reverse_combine(self, invocation: _Invocation, grad_output: torch.Tensor) -> torch.Tensor:
        if invocation.entry is None or invocation.handle is None:
            raise DeepEPBackendError("combine backward lost its handle")
        recv_x, _ids, _weights, _counts, _handle, event = invocation.entry.buffer.dispatch(
            x=grad_output.to(torch.bfloat16).contiguous(),
            handle=invocation.handle,
            config=invocation.entry.dispatch_config,
            async_finish=self.config.async_finish,
            allocate_on_comm_stream=self.config.async_finish,
        )
        _wait_event(event, self.config.async_finish)
        return recv_x

    def _reverse_dispatch(
        self,
        invocation: _Invocation,
        grad_recv_x: torch.Tensor | None,
        grad_recv_weights: torch.Tensor | None,
    ):
        if (
            invocation.entry is None
            or invocation.handle is None
            or invocation.recv_shape is None
            or invocation.recv_device is None
        ):
            raise DeepEPBackendError("dispatch backward lost invocation state")
        if grad_recv_x is None:
            grad_recv_x = torch.zeros(
                invocation.recv_shape,
                dtype=torch.bfloat16,
                device=invocation.recv_device,
            )
        optional = {}
        if grad_recv_weights is not None:
            optional["topk_weights"] = grad_recv_weights.float().contiguous()
        grad_hidden, grad_routing, event = invocation.entry.buffer.combine(
            x=grad_recv_x.to(torch.bfloat16).contiguous(),
            handle=invocation.handle,
            config=invocation.entry.combine_config,
            async_finish=self.config.async_finish,
            allocate_on_comm_stream=self.config.async_finish,
            **optional,
        )
        _wait_event(event, self.config.async_finish)
        if grad_recv_weights is not None and grad_routing is None:
            raise DeepEPBackendError("this DeepEP combine API cannot propagate routing-weight gradients")
        return grad_hidden, grad_routing

    def dispatch(
        self,
        *,
        group: object,
        num_experts: int,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):
        self.validate_inputs(
            group=group,
            num_experts=num_experts,
            hidden_states=hidden_states,
            topk_indices=topk_indices,
            routing_weights=routing_weights,
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
        )
        invocation = _Invocation(
            entry=self._entry(group, hidden_states, num_experts, topk_indices.shape[1]),
            num_experts=num_experts,
        )
        if not self._logged_ready:
            dist = self._distributed()
            message = (
                "UniRL VeOmni DeepEP-HT ready: ep_comm_backend=deepep_ht "
                f"low_latency_mode=False rank={dist.get_rank(group)} "
                f"ep_size={invocation.entry.ep_size if invocation.entry is not None else '?'} "
                f"num_experts={num_experts} local_experts={gate_up_proj.shape[0]} "
                f"hidden={hidden_states.shape[-1]} topk={topk_indices.shape[1]}"
            )
            print(message, flush=True)
            logger.warning(message)
            self._logged_ready = True
        recv_x, recv_ids, recv_weights = _DeepEPDispatch.apply(
            self,
            invocation,
            hidden_states.contiguous(),
            topk_indices.contiguous(),
            routing_weights.contiguous(),
        )
        return invocation, recv_x, recv_ids, recv_weights

    def combine(self, invocation: _Invocation, partial: torch.Tensor) -> torch.Tensor:
        return _DeepEPCombine.apply(self, invocation, partial.contiguous())


__all__ = [
    "DeepEPBackendError",
    "DeepEPBufferCache",
    "DeepEPDispatcher",
    "close_deepep_ht_buffers",
]
