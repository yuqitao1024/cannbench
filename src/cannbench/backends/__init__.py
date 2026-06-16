from cannbench.backends.pytorch_backend import NvidiaBackend


def get_backend(name: str):
    if name == "nvidia":
        return NvidiaBackend()
    raise ValueError(f"Unsupported backend: {name}")


__all__ = ["get_backend", "NvidiaBackend"]
