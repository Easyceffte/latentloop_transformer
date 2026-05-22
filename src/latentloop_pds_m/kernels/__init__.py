"""Kernel helpers for LatentLoop memory retrieval.

The package is intentionally optional-dependency friendly: CUDA/Triton specific
paths may be added later, but all public functions have a deterministic PyTorch
fallback used by tests and 4070-local runs.
"""

from .memory_retrieval import chunked_exact_topk_retrieval

__all__ = ["chunked_exact_topk_retrieval"]
