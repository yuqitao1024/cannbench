from cannbench.backends.pytorch_backend import AscendBackend, NvidiaBackend


def get_backend(name: str):
    if name == "nvidia":
        return NvidiaBackend()
    if name == "ascend":
        return AscendBackend()
    raise ValueError(f"Unsupported backend: {name}")


__all__ = ["AscendBackend", "NvidiaBackend", "get_backend"]
