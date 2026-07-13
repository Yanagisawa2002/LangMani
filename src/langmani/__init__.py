"""LangMani package metadata.

The root import remains lightweight. Import :mod:`langmani.environments` to register
the M1 environment, :mod:`langmani.experts` for the M2 privileged expert and lazy
mplib adapter, or :mod:`langmani.collection` / :mod:`langmani.datasets` for the
active M3A raw-demonstration pipeline.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
