from __future__ import annotations


def bound_indices(ctx, shape, *, dtype=None):
    indices = ctx.bound_inputs.get("indices")
    if indices is None:
        return None
    if dtype is not None and getattr(indices, "dtype", dtype) != dtype:
        indices = indices.to(dtype=dtype)
    return indices.reshape(shape)
