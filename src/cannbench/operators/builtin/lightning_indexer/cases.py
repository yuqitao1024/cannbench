from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class LightningIndexerCase:
    case_id: str
    family: str
    batch: int
    query_tokens: int
    context_tokens: int
    index_heads: int
    index_dim: int
    top_k: int
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str
    causal: bool = False
    score_scale: float = 1.0
    tie_policy: str = "equivalent_score_set"
    query_lens: tuple[int, ...] | None = None
    context_lens: tuple[int, ...] | None = None
    query_start_positions: tuple[int, ...] | None = None
    page_block_size: int | None = None
    shape_scope: str | None = None
    tp_size: int | None = None
    dp_size: int | None = None
    cp_size: int | None = None
    kv_shard: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_tokens",
            "context_tokens",
            "index_heads",
            "index_dim",
            "top_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.top_k > self.context_tokens:
            raise ValueError("top_k must not exceed context_tokens")
        if self.causal and self.query_tokens > self.context_tokens:
            raise ValueError("causal query_tokens must not exceed context_tokens")
        if not math.isfinite(self.score_scale) or self.score_scale <= 0.0:
            raise ValueError("score_scale must be finite and positive")
        if self.tie_policy != "equivalent_score_set":
            raise ValueError(f"unsupported tie_policy: {self.tie_policy}")
        self._validate_parallelism()
        self._validate_sequence_metadata()

    def _validate_parallelism(self) -> None:
        values = (
            self.shape_scope,
            self.tp_size,
            self.dp_size,
            self.cp_size,
            self.kv_shard,
        )
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("rank-local shape metadata must be complete")
        if self.shape_scope != "rank_local":
            raise ValueError("shape_scope must be rank_local")
        if any(size <= 0 for size in (self.tp_size, self.dp_size, self.cp_size)):
            raise ValueError("parallel sizes must be positive")
        if self.kv_shard not in {"replicated", "context_sharded"}:
            raise ValueError(f"unsupported kv_shard: {self.kv_shard}")

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
    def payload(self) -> dict[str, object]:
        payload = {
            "batch": self.batch,
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "index_heads": self.index_heads,
            "index_dim": self.index_dim,
            "top_k": self.top_k,
            "causal": self.causal,
            "score_scale": self.score_scale,
            "tie_policy": self.tie_policy,
            "query_lens": self.resolved_query_lens,
            "context_lens": self.resolved_context_lens,
            "query_start_positions": self.resolved_query_start_positions,
            "cu_seqlens_q": self.cu_seqlens_q,
            "cu_seqlens_kv": self.cu_seqlens_kv,
            "page_block_size": self.resolved_page_block_size,
            "block_tables": self.block_tables,
        }
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.shape_scope is not None:
            payload["parallelism"] = self.parallelism
        return payload

    @property
    def parallelism(self) -> dict[str, object] | None:
        if self.shape_scope is None:
            return None
        return {
            "shape_scope": self.shape_scope,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "cp_size": self.cp_size,
            "kv_shard": self.kv_shard,
        }

    @property
    def phase(self) -> str | None:
        if self.family.startswith("decode_") or "_decode_" in self.family:
            return "decode"
        if self.family.startswith("prefill_") or "_prefill_" in self.family:
            return "prefill"
        return None


@dataclass(frozen=True)
class LightningIndexerDataset:
    name: str
    cases: tuple[LightningIndexerCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_lightning_indexer_dataset(name: str) -> LightningIndexerDataset:
    resource = files(__package__).joinpath("data", f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown lightning_indexer dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(LightningIndexerCase(**item) for item in payload["cases"])
    return LightningIndexerDataset(name=payload["name"], cases=cases)


def get_lightning_indexer_case(
    dataset_name: str, case_id: str
) -> LightningIndexerCase:
    dataset = get_lightning_indexer_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown lightning_indexer case: {case_id}")


def _prefix_sums(lengths: tuple[int, ...]) -> tuple[int, ...]:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return tuple(values)
