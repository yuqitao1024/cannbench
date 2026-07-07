from __future__ import annotations

import importlib
import math
from typing import Any


def lightning_indexer(**kwargs: Any) -> Any:
    """Route CannBench DSA index selection to DeepGEMM by workflow phase."""
    phase = _resolve_phase(kwargs)
    deep_gemm = _import_required("deep_gemm", op_name="lightning_indexer")
    if phase == "decode":
        return _call_required(
            deep_gemm,
            "fp8_paged_mqa_logits",
            op_name="lightning_indexer",
            kwargs=_deep_gemm_decode_indexer_kwargs(deep_gemm, kwargs),
        )
    if phase == "prefill":
        return _call_required(
            deep_gemm,
            "fp8_mqa_logits",
            op_name="lightning_indexer",
            kwargs=_deep_gemm_prefill_indexer_kwargs(kwargs),
        )
    raise _unsupported_phase(phase, "lightning_indexer")


def sparse_attention(**kwargs: Any) -> Any:
    """Route CannBench DSA sparse attention to FlashMLA by workflow phase."""
    phase = _resolve_phase(kwargs)
    flash_mla = _import_required("flash_mla", op_name="sparse_attention")
    if phase == "decode":
        return _call_required(
            flash_mla,
            "flash_mla_with_kvcache",
            op_name="sparse_attention",
            kwargs=_flash_mla_decode_attention_kwargs(flash_mla, kwargs),
        )
    if phase == "prefill":
        return _call_required(
            flash_mla,
            "flash_mla_sparse_fwd",
            op_name="sparse_attention",
            kwargs=_flash_mla_prefill_attention_kwargs(kwargs),
        )
    raise _unsupported_phase(phase, "sparse_attention")


def _resolve_phase(kwargs: dict[str, Any]) -> str:
    payload = kwargs.get("payload")
    if isinstance(payload, dict):
        phase = payload.get("phase")
        if isinstance(phase, str) and phase:
            return phase
    phase = kwargs.get("phase")
    if isinstance(phase, str) and phase:
        return phase
    case = kwargs.get("case")
    phase = getattr(case, "phase", None)
    if isinstance(phase, str) and phase:
        return phase
    raise RuntimeError(
        "cannbench_cuda_dsa_flashmla_deepgemm requires a DSA workflow phase "
        "('decode' or 'prefill') from payload['phase'], phase, or case.phase."
    )


def _import_required(module_name: str, *, op_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"cannbench_cuda_dsa_flashmla_deepgemm {op_name} requires "
            f"the CUDA library module {module_name!r} to be installed."
        ) from exc


def _call_required(module, function_name: str, *, op_name: str, kwargs: dict[str, Any]):
    candidate = getattr(module, function_name, None)
    if not callable(candidate):
        raise RuntimeError(
            f"cannbench_cuda_dsa_flashmla_deepgemm {op_name} requires "
            f"{module.__name__}.{function_name} to be callable."
        )
    return candidate(**kwargs)


def _deep_gemm_prefill_indexer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    torch = _require_torch(kwargs)
    payload = _require_payload(kwargs)
    query = kwargs["query"].reshape(
        payload["query_shape"][0] * payload["query_shape"][1],
        payload["query_shape"][2],
        payload["query_shape"][3],
    )
    keys = kwargs["keys"].reshape(
        payload["key_shape"][0] * payload["key_shape"][1],
        payload["key_shape"][2],
    )
    weights = kwargs["weights"].reshape(
        payload["weight_shape"][0] * payload["weight_shape"][1],
        payload["weight_shape"][2],
    ).to(torch.float32)
    q_fp8 = query.to(torch.float8_e4m3fn)
    kv_fp8 = _per_custom_dims_cast_to_fp8(torch, keys, dims=(0,), use_ue8m0=False)
    batch, query_tokens = payload["query_shape"][0], payload["query_shape"][1]
    context_tokens = payload["key_shape"][1]
    starts: list[int] = []
    ends: list[int] = []
    for batch_index in range(batch):
        starts.extend([batch_index * context_tokens] * query_tokens)
        ends.extend([(batch_index + 1) * context_tokens] * query_tokens)
    return {
        "q": q_fp8,
        "kv": kv_fp8,
        "weights": weights,
        "cu_seq_len_k_start": torch.tensor(
            starts, device=query.device, dtype=torch.int32
        ),
        "cu_seq_len_k_end": torch.tensor(ends, device=query.device, dtype=torch.int32),
        "clean_logits": False,
        "max_seqlen_k": 0,
    }


def _deep_gemm_decode_indexer_kwargs(deep_gemm, kwargs: dict[str, Any]) -> dict[str, Any]:
    torch = _require_torch(kwargs)
    payload = _require_payload(kwargs)
    block_kv = 64
    query = kwargs["query"]
    batch, query_tokens, index_heads, index_dim = payload["query_shape"]
    context_tokens = payload["key_shape"][1]
    if query_tokens != 1:
        raise RuntimeError("DeepGEMM decode indexer requires query_tokens == 1")
    q_fp8 = query.reshape(batch, query_tokens, index_heads, index_dim).to(
        torch.float8_e4m3fn
    )
    block_table, max_context_len = _sequential_block_table(
        torch, batch=batch, context_tokens=context_tokens, block_size=block_kv, device=query.device
    )
    kv_cache_bf16 = _blocked_kv_cache(
        torch,
        kwargs["keys"],
        batch=batch,
        context_tokens=context_tokens,
        kv_heads=1,
        head_dim=index_dim,
        block_size=block_kv,
        max_context_len=max_context_len,
    ).to(torch.bfloat16)
    fused_kv_cache = _deep_gemm_fp8_kv_cache(torch, kv_cache_bf16)
    context_lens = torch.full(
        (batch, query_tokens), context_tokens, device=query.device, dtype=torch.int32
    )
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        deep_gemm.get_num_sms(),
        indices=None,
    )
    weights = kwargs["weights"].reshape(batch * query_tokens, index_heads).to(torch.float32)
    return {
        "q": q_fp8,
        "kv_cache": fused_kv_cache,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "schedule_meta": schedule_meta,
        "max_context_len": max_context_len,
        "clean_logits": False,
        "indices": None,
    }


def _flash_mla_prefill_attention_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    torch = _require_torch(kwargs)
    payload = _require_payload(kwargs)
    batch, query_heads, query_tokens, head_dim = payload["query_shape"]
    _, kv_heads, context_tokens, _ = payload["key_shape"]
    query = _bhtd_to_bthd_flat(kwargs["query"], batch, query_heads, query_tokens, head_dim)
    kv = _bhtd_to_bthd_flat(kwargs["values"], batch, kv_heads, context_tokens, head_dim)
    indices = kwargs["indices"].to(torch.int32)
    offsets = (
        torch.arange(batch, device=query.device, dtype=torch.int32)
        .view(batch, 1, 1)
        * context_tokens
    )
    indices = indices + offsets
    indices = indices.reshape(batch * query_tokens, 1, payload["indices_shape"][2])
    if kv_heads > 1:
        indices = indices.expand(batch * query_tokens, kv_heads, payload["indices_shape"][2])
    return {
        "q": query.to(torch.bfloat16),
        "kv": kv.to(torch.bfloat16),
        "indices": indices.contiguous(),
        "sm_scale": float(kwargs.get("softmax_scale", head_dim ** -0.5)),
        "d_v": head_dim,
    }


def _flash_mla_decode_attention_kwargs(flash_mla, kwargs: dict[str, Any]) -> dict[str, Any]:
    torch = _require_torch(kwargs)
    payload = _require_payload(kwargs)
    block_size = 64
    batch, query_heads, query_tokens, head_dim = payload["query_shape"]
    _, kv_heads, context_tokens, _ = payload["key_shape"]
    if query_tokens != 1:
        raise RuntimeError("FlashMLA decode attention requires query_tokens == 1")
    if kv_heads != 1:
        raise RuntimeError("FlashMLA sparse decode requires kv_heads == 1")
    query = _bhtd_to_bthd(kwargs["query"], batch, query_heads, query_tokens, head_dim)
    block_table, max_context_len = _sequential_block_table(
        torch,
        batch=batch,
        context_tokens=context_tokens,
        block_size=block_size,
        device=query.device,
    )
    kv_cache_bf16 = _blocked_kv_cache(
        torch,
        kwargs["values"],
        batch=batch,
        context_tokens=context_tokens,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_context_len=max_context_len,
    ).to(torch.bfloat16)
    k_cache = _flash_mla_model1_fp8_k_cache(torch, kv_cache_bf16)
    abs_indices = kwargs["indices"].to(torch.int32)
    indices_in_kvcache = _indices_to_kvcache_indices(
        torch,
        abs_indices,
        block_table,
        block_size=block_size,
    )
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()
    return {
        "q": query.to(torch.bfloat16),
        "k_cache": k_cache,
        "block_table": None,
        "cache_seqlens": None,
        "head_dim_v": head_dim,
        "tile_scheduler_metadata": tile_scheduler_metadata,
        "num_splits": num_splits,
        "softmax_scale": float(kwargs.get("softmax_scale", head_dim ** -0.5)),
        "causal": False,
        "is_fp8_kvcache": True,
        "indices": indices_in_kvcache.contiguous(),
    }


def _require_torch(kwargs: dict[str, Any]):
    torch = kwargs.get("torch")
    if torch is None:
        raise RuntimeError("CUDA DSA wrapper requires torch in adapter kwargs")
    return torch


def _require_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    payload = kwargs.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("CUDA DSA wrapper requires payload in adapter kwargs")
    return payload


def _bhtd_to_bthd(tensor, batch: int, heads: int, tokens: int, dim: int):
    return tensor.reshape(batch, heads, tokens, dim).permute(0, 2, 1, 3).contiguous()


def _bhtd_to_bthd_flat(tensor, batch: int, heads: int, tokens: int, dim: int):
    return _bhtd_to_bthd(tensor, batch, heads, tokens, dim).reshape(
        batch * tokens, heads, dim
    )


def _sequential_block_table(torch, *, batch: int, context_tokens: int, block_size: int, device):
    blocks_per_batch = max(math.ceil(context_tokens / block_size), 1)
    max_context_len = blocks_per_batch * block_size
    block_table = torch.arange(
        batch * blocks_per_batch, device=device, dtype=torch.int32
    ).reshape(batch, blocks_per_batch)
    return block_table, max_context_len


def _blocked_kv_cache(
    torch,
    values,
    *,
    batch: int,
    context_tokens: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    max_context_len: int,
):
    values = _bhtd_to_bthd(values, batch, kv_heads, context_tokens, head_dim)
    if max_context_len > context_tokens:
        padding = torch.zeros(
            batch,
            max_context_len - context_tokens,
            kv_heads,
            head_dim,
            device=values.device,
            dtype=values.dtype,
        )
        values = torch.cat((values, padding), dim=1)
    return values.reshape(batch * (max_context_len // block_size), block_size, kv_heads, head_dim)


def _indices_to_kvcache_indices(torch, indices, block_table, *, block_size: int):
    batch, query_tokens, top_k = indices.shape
    blocks_per_batch = block_table.shape[1]
    safe_indices = indices.clone()
    invalid = safe_indices < 0
    safe_indices[invalid] = 0
    block_offsets = safe_indices // block_size
    flat_block_indices = (
        block_offsets
        + torch.arange(batch, device=indices.device, dtype=torch.int32).view(batch, 1, 1)
        * blocks_per_batch
    ).reshape(-1)
    real_blocks = block_table.reshape(-1).index_select(0, flat_block_indices.to(torch.long))
    result = (real_blocks.reshape(batch, query_tokens, top_k) * block_size) + (
        safe_indices % block_size
    )
    result[invalid] = -1
    return result


def _per_custom_dims_cast_to_fp8(torch, tensor, *, dims: tuple[int, ...], use_ue8m0: bool):
    excluded_dims = tuple(index for index in range(tensor.dim()) if index not in set(dims))
    scale = tensor.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4) / 448.0
    if use_ue8m0:
        bits = scale.abs().float().view(torch.int32)
        exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
        scale = (exponent.clamp(1, 254) << 23).view(torch.float32)
    return (tensor * (1.0 / scale)).to(torch.float8_e4m3fn).contiguous(), scale.squeeze().contiguous()


def _deep_gemm_fp8_kv_cache(torch, tensor):
    num_blocks, block_size, num_heads, head_dim = tensor.shape
    if num_heads != 1:
        raise RuntimeError("DeepGEMM paged indexer requires kv_heads == 1")
    scale = tensor.abs().float().amax(dim=3, keepdim=True).clamp(1e-4) / 448.0
    scaled = (tensor * (1.0 / scale)).to(torch.float8_e4m3fn)
    result = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=tensor.device,
        dtype=torch.uint8,
    )
    result[:, : block_size * head_dim] = scaled.reshape(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)
    result[:, block_size * head_dim :] = scale.reshape(num_blocks, block_size).view(
        torch.uint8
    )
    return result.view(num_blocks, block_size, num_heads, head_dim + 4)


def _flash_mla_model1_fp8_k_cache(torch, tensor):
    num_blocks, block_size, num_heads, head_dim = tensor.shape
    if num_heads != 1 or head_dim != 512:
        raise RuntimeError("FlashMLA MODEL1 sparse decode requires shape [blocks, block, 1, 512]")
    d_nope = 448
    d_rope = 64
    tile_size = 64
    num_tiles = 7
    bytes_per_token = d_nope + 2 * d_rope + num_tiles + 1
    size_per_block_padded = math.ceil(block_size * bytes_per_token / 576) * 576
    result = torch.empty(
        (num_blocks, size_per_block_padded),
        dtype=torch.float8_e4m3fn,
        device=tensor.device,
    )[:, : block_size * bytes_per_token]
    nope_rope = result[:, : block_size * (d_nope + 2 * d_rope)].view(
        num_blocks, block_size, d_nope + 2 * d_rope
    )
    nope = nope_rope[:, :, :d_nope]
    rope = nope_rope[:, :, d_nope:].view(tensor.dtype)
    scales = result[:, block_size * (d_nope + 2 * d_rope) :].view(
        num_blocks, block_size, 8
    )[:, :, :num_tiles]
    if hasattr(torch, "float8_e8m0fnu"):
        scales = scales.view(torch.float8_e8m0fnu)
    source = tensor.squeeze(2)
    rope[:] = source[..., d_nope:]
    for tile_index in range(num_tiles):
        start = tile_index * tile_size
        end = start + tile_size
        cur_scale = source[..., start:end].abs().float().amax(dim=-1).clamp(1e-4) / 448.0
        bits = cur_scale.abs().float().view(torch.int32)
        exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
        cur_scale = (exponent.clamp(1, 254) << 23).view(torch.float32)
        scales[:, :, tile_index] = cur_scale.to(scales.dtype)
        nope[:, :, start:end] = (
            source[..., start:end].float() / cur_scale.unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
    return result.view(num_blocks, block_size, num_heads, -1)


def _unsupported_phase(phase: str, op_name: str) -> RuntimeError:
    return RuntimeError(
        f"cannbench_cuda_dsa_flashmla_deepgemm {op_name} does not support "
        f"phase {phase!r}; expected 'decode' or 'prefill'."
    )
