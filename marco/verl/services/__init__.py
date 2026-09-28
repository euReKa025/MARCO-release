from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictors import (
        DEFAULT_ADMET_API,
        DEFAULT_DRD2_API,
        DEFAULT_PROBE_SMILES,
        DEFAULT_PREDICTOR_CACHE_RELATIVE,
        build_predictor_exports,
        is_port_in_use,
        legacy_predictor_cache_path,
        probe_predictor_endpoint,
        predictor_cache_path,
        predictor_endpoints,
        wait_for_predictor_endpoint,
        wait_for_tcp_port,
    )

__all__ = [
    "DEFAULT_ADMET_API",
    "DEFAULT_DRD2_API",
    "DEFAULT_PROBE_SMILES",
    "DEFAULT_PREDICTOR_CACHE_RELATIVE",
    "build_predictor_exports",
    "is_port_in_use",
    "legacy_predictor_cache_path",
    "probe_predictor_endpoint",
    "predictor_cache_path",
    "predictor_endpoints",
    "wait_for_predictor_endpoint",
    "wait_for_tcp_port",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".predictors", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
