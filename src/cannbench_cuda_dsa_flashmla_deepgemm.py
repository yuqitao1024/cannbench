from __future__ import annotations

import importlib
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
            kwargs=kwargs,
        )
    if phase == "prefill":
        return _call_required(
            deep_gemm,
            "fp8_mqa_logits",
            op_name="lightning_indexer",
            kwargs=kwargs,
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
            kwargs=kwargs,
        )
    if phase == "prefill":
        return _call_required(
            flash_mla,
            "flash_mla_sparse_fwd",
            op_name="sparse_attention",
            kwargs=kwargs,
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
    return candidate(**_library_kwargs(kwargs))


def _library_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key not in {"torch", "request", "case", "device", "dtype", "phase"}
    }


def _unsupported_phase(phase: str, op_name: str) -> RuntimeError:
    return RuntimeError(
        f"cannbench_cuda_dsa_flashmla_deepgemm {op_name} does not support "
        f"phase {phase!r}; expected 'decode' or 'prefill'."
    )
