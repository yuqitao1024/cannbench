from __future__ import annotations

import random
from math import ceil, floor

from .cases import AdaptiveMaxPool3DGradCase


def _numel(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def _pool_start(output_index: int, input_size: int, output_size: int) -> int:
    return floor(output_index * input_size / output_size)


def _pool_end(output_index: int, input_size: int, output_size: int) -> int:
    return ceil((output_index + 1) * input_size / output_size)


def _flatten_spatial_index(d: int, h: int, w: int, h_in: int, w_in: int) -> int:
    return d * h_in * w_in + h * w_in + w


def compute_adaptivemaxpool3d_indices(
    input_values: tuple[float, ...],
    *,
    input_shape: tuple[int, int, int, int, int],
    output_size: tuple[int, int, int],
) -> tuple[int, ...]:
    n_size, c_size, d_in, h_in, w_in = input_shape
    d_out, h_out, w_out = output_size
    indices: list[int] = []
    spatial_size = d_in * h_in * w_in

    for n_index in range(n_size):
        for c_index in range(c_size):
            channel_base = (n_index * c_size + c_index) * spatial_size
            for od in range(d_out):
                d_start = _pool_start(od, d_in, d_out)
                d_end = _pool_end(od, d_in, d_out)
                for oh in range(h_out):
                    h_start = _pool_start(oh, h_in, h_out)
                    h_end = _pool_end(oh, h_in, h_out)
                    for ow in range(w_out):
                        w_start = _pool_start(ow, w_in, w_out)
                        w_end = _pool_end(ow, w_in, w_out)
                        max_value = None
                        max_index = 0
                        for d_index in range(d_start, d_end):
                            for h_index in range(h_start, h_end):
                                for w_index in range(w_start, w_end):
                                    spatial_index = _flatten_spatial_index(
                                        d_index,
                                        h_index,
                                        w_index,
                                        h_in,
                                        w_in,
                                    )
                                    value = input_values[channel_base + spatial_index]
                                    if max_value is None or value > max_value:
                                        max_value = value
                                        max_index = spatial_index
                        indices.append(max_index)
    return tuple(indices)


def materialize_adaptivemaxpool3dgrad_inputs(
    case: AdaptiveMaxPool3DGradCase,
    *,
    dtype: str,
    seed: int,
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = _numel(case.input_shape)
    grad_output_size = _numel(case.output_shape)

    input_values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    grad_output_values = tuple(
        round(generator.uniform(-1.0, 1.0), 6) for _ in range(grad_output_size)
    )
    indices = compute_adaptivemaxpool3d_indices(
        input_values,
        input_shape=case.input_shape,
        output_size=case.output_size,
    )
    return {
        "input_shape": case.input_shape,
        "output_size": case.output_size,
        "output_shape": case.output_shape,
        "dtype": dtype,
        "input_values": input_values,
        "grad_output_values": grad_output_values,
        "indices": indices,
    }
