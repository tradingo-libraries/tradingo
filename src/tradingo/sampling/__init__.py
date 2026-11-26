"""Data samplers, one module per provider.

Submodules are not imported here, so a process loads only the providers (and
SDKs) it uses: import the one you need, e.g. ``tradingo.sampling.dukascopy``.
"""

__all__ = [
    "dukascopy",
    "instruments",
    "ig",
    "quality",
    "yf",
]
