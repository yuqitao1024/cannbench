from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any


_LIGHTNING_INDEXER_ENV = "CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER"
_SPARSE_ATTENTION_ENV = "CANNBENCH_CUDA_DSA_SPARSE_ATTENTION"


def lightning_indexer(**kwargs: Any) -> Any:
    """Dispatch CannBench DSA index selection to a configured CUDA library wrapper."""
    return _resolve_configured_callable(
        env_var=_LIGHTNING_INDEXER_ENV,
        fallback_names=(
            "cannbench_cuda_dsa_flashmla_deepgemm:lightning_indexer",
            "flash_mla:lightning_indexer",
            "flashmla:lightning_indexer",
            "deep_gemm:lightning_indexer",
            "deepgemm:lightning_indexer",
        ),
        op_name="lightning_indexer",
    )(**kwargs)


def sparse_attention(**kwargs: Any) -> Any:
    """Dispatch CannBench sparse attention to a configured CUDA library wrapper."""
    return _resolve_configured_callable(
        env_var=_SPARSE_ATTENTION_ENV,
        fallback_names=(
            "cannbench_cuda_dsa_flashmla_deepgemm:sparse_attention",
            "flash_mla:sparse_attention",
            "flash_mla:sparse_mla_decode",
            "flash_mla:sparse_mla_prefill",
            "flashmla:sparse_attention",
            "flashmla:sparse_mla_decode",
            "flashmla:sparse_mla_prefill",
        ),
        op_name="sparse_attention",
    )(**kwargs)


def _resolve_configured_callable(
    *,
    env_var: str,
    fallback_names: tuple[str, ...],
    op_name: str,
) -> Callable[..., Any]:
    explicit = os.environ.get(env_var)
    candidates = (explicit,) if explicit else fallback_names
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return _import_symbol(candidate, default_symbol=op_name)
        except (ModuleNotFoundError, AttributeError, TypeError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        f"cannbench_cuda_dsa cannot resolve CUDA DSA callable {op_name}. "
        f"Set {env_var}=<module>:<callable> to a wrapper that calls the installed "
        "FlashMLA/DeepGEMM CUDA library. Tried: "
        + (", ".join(errors) if errors else ", ".join(fallback_names))
    )


def _import_symbol(spec: str, *, default_symbol: str | None = None) -> Callable[..., Any]:
    module_name, separator, symbol_name = spec.partition(":")
    if not module_name:
        raise TypeError("empty module name")
    module = importlib.import_module(module_name)
    if separator:
        if not symbol_name:
            raise TypeError("empty callable name")
        candidate = getattr(module, symbol_name)
    else:
        default_name = default_symbol or module_name.rsplit(".", 1)[-1]
        candidate = getattr(module, default_name, None)
        if candidate is None:
            raise AttributeError(
                f"module {module_name!r} has no module-level callable {default_name!r}"
            )
    if not callable(candidate):
        raise TypeError(f"{spec} is not callable")
    return candidate
