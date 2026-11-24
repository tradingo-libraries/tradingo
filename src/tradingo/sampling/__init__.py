"""Data samplers, one module per provider.

Submodules are not imported here: importing ``ig`` loads trading_ig, whose
package import starts Lightstreamer's polling threads (~0.05 core per process,
for ever). Import the provider module you need, e.g. ``tradingo.sampling.dukascopy``.
"""

__all__ = [
    "dukascopy",
    "instruments",
    "ig",
    "quality",
    "yf",
]
