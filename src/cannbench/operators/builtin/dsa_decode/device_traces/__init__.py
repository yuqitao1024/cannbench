from .v2 import build_v2_device_trace

DEVICE_TRACE_BUILDERS = {"v2": build_v2_device_trace}

__all__ = ["DEVICE_TRACE_BUILDERS", "build_v2_device_trace"]
