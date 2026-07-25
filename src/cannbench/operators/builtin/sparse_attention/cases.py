from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class SparseAttentionCase:
    case_id: str
    family: str
    batch: int
    query_heads: int
    kv_heads: int
    query_tokens: int
    context_tokens: int
    selected_tokens: int
    qk_head_dim: int
    value_head_dim: int
    causal: bool
    phase: str
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str
    shared_kv: bool = True
    softmax_scale: float | None = None
    topk_lengths: tuple[int, ...] | None = None
    query_lens: tuple[int, ...] | None = None
    context_lens: tuple[int, ...] | None = None
    query_start_positions: tuple[int, ...] | None = None
    page_block_size: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_heads",
            "kv_heads",
            "query_tokens",
            "context_tokens",
            "selected_tokens",
            "qk_head_dim",
            "value_head_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.selected_tokens > self.context_tokens:
            raise ValueError("selected_tokens must not exceed context_tokens")
        if self.shared_kv and self.value_head_dim > self.qk_head_dim:
            raise ValueError("shared-KV requires value_head_dim <= qk_head_dim")
        if self.phase not in {"decode", "prefill"}:
            raise ValueError("phase must be decode or prefill")
        softmax_scale = self.softmax_scale
        if softmax_scale is None:
            softmax_scale = self.qk_head_dim**-0.5
            object.__setattr__(self, "softmax_scale", softmax_scale)
        if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
            raise ValueError("softmax_scale must be finite and positive")
        if self.topk_lengths is not None:
            topk_lengths = tuple(self.topk_lengths)
            if len(topk_lengths) != self.batch * self.query_tokens:
                raise ValueError(
                    "topk_lengths must contain one value per query"
                )
            if any(
                length < 0 or length > self.selected_tokens
                for length in topk_lengths
            ):
                raise ValueError(
                    "topk_lengths values must be between 0 and selected_tokens"
                )
            object.__setattr__(self, "topk_lengths", topk_lengths)
        self._validate_sequence_metadata()
        if self.topk_lengths is not None and any(
            self.topk_lengths[batch_index * self.query_tokens + query_index] != 0
            for batch_index, query_len in enumerate(self.resolved_query_lens)
            for query_index in range(query_len, self.query_tokens)
        ):
            raise ValueError("padding query rows must have zero topk_lengths")

    def _validate_sequence_metadata(self) -> None:
        for name, values, maximum, allow_zero in (
            ("query_lens", self.query_lens, self.query_tokens, True),
            ("context_lens", self.context_lens, self.context_tokens, False),
        ):
            if values is None:
                continue
            values = tuple(values)
            if len(values) != self.batch:
                raise ValueError(f"{name} must contain one value per request")
            minimum = 0 if allow_zero else 1
            if any(value < minimum or value > maximum for value in values):
                raise ValueError(f"{name} values exceed the padded shape")
            object.__setattr__(self, name, values)
        if self.query_start_positions is not None:
            positions = tuple(self.query_start_positions)
            if len(positions) != self.batch:
                raise ValueError(
                    "query_start_positions must contain one value per request"
                )
            object.__setattr__(self, "query_start_positions", positions)
        if any(
            start < 0 or start + query_len > context_len
            for start, query_len, context_len in zip(
                self.resolved_query_start_positions,
                self.resolved_query_lens,
                self.resolved_context_lens,
                strict=True,
            )
        ):
            raise ValueError("query positions must lie within each context")
        if self.causal and any(
            start != context_len - query_len
            for start, query_len, context_len in zip(
                self.resolved_query_start_positions,
                self.resolved_query_lens,
                self.resolved_context_lens,
                strict=True,
            )
        ):
            raise ValueError("causal queries must be right-aligned within each context")
        if self.page_block_size is not None and self.page_block_size <= 0:
            raise ValueError("page_block_size must be positive")

    @property
    def resolved_query_lens(self) -> tuple[int, ...]:
        return self.query_lens or (self.query_tokens,) * self.batch

    @property
    def resolved_context_lens(self) -> tuple[int, ...]:
        return self.context_lens or (self.context_tokens,) * self.batch

    @property
    def resolved_query_start_positions(self) -> tuple[int, ...]:
        if self.query_start_positions is not None:
            return self.query_start_positions
        return tuple(
            context_len - query_len
            for query_len, context_len in zip(
                self.resolved_query_lens,
                self.resolved_context_lens,
                strict=True,
            )
        )

    @property
    def cu_seqlens_q(self) -> tuple[int, ...]:
        return _prefix_sums(self.resolved_query_lens)

    @property
    def cu_seqlens_kv(self) -> tuple[int, ...]:
        return _prefix_sums(self.resolved_context_lens)

    @property
    def resolved_page_block_size(self) -> int:
        if self.page_block_size is not None:
            return self.page_block_size
        for block_size in (64, 32, 16):
            if self.context_tokens >= block_size:
                return block_size
        return self.context_tokens

    @property
    def block_tables(self) -> tuple[tuple[int, ...], ...]:
        block_size = self.resolved_page_block_size
        max_blocks = (self.context_tokens + block_size - 1) // block_size
        tables = []
        for batch_index, context_len in enumerate(self.resolved_context_lens):
            valid_blocks = (context_len + block_size - 1) // block_size
            base = batch_index * max_blocks
            tables.append(
                tuple(base + index if index < valid_blocks else -1 for index in range(max_blocks))
            )
        return tuple(tables)

    @property
    def resolved_topk_lengths(self) -> tuple[int, ...]:
        if self.topk_lengths is not None:
            return self.topk_lengths
        return tuple(
            self.selected_tokens if query_index < query_len else 0
            for query_len in self.resolved_query_lens
            for query_index in range(self.query_tokens)
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "batch": self.batch,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "selected_tokens": self.selected_tokens,
            "qk_head_dim": self.qk_head_dim,
            "value_head_dim": self.value_head_dim,
            "shared_kv": self.shared_kv,
            "causal": self.causal,
            "phase": self.phase,
            "softmax_scale": self.softmax_scale,
            "topk_lengths": self.resolved_topk_lengths,
            "query_lens": self.resolved_query_lens,
            "context_lens": self.resolved_context_lens,
            "query_start_positions": self.resolved_query_start_positions,
            "cu_seqlens_q": self.cu_seqlens_q,
            "cu_seqlens_kv": self.cu_seqlens_kv,
            "page_block_size": self.resolved_page_block_size,
            "block_tables": self.block_tables,
        }


@dataclass(frozen=True)
class SparseAttentionDataset:
    name: str
    cases: tuple[SparseAttentionCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_sparse_attention_dataset(name: str) -> SparseAttentionDataset:
    resource = files(__package__).joinpath("data", f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown sparse_attention dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(SparseAttentionCase(**item) for item in payload["cases"])
    return SparseAttentionDataset(name=payload["name"], cases=cases)


def get_sparse_attention_case(dataset_name: str, case_id: str) -> SparseAttentionCase:
    dataset = get_sparse_attention_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown sparse_attention case: {case_id}")


def _prefix_sums(lengths: tuple[int, ...]) -> tuple[int, ...]:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return tuple(values)
