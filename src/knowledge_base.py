"""
Online self-retrieval knowledge base using FAISS.
Stores embeddings from the target series' own history.
"""

import faiss
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque


@dataclass
class KBEntry:
    """A single entry in the knowledge base."""
    window: np.ndarray          # Raw time series values
    future: np.ndarray          # What actually happened next (ground truth)
    embedding: np.ndarray       # Chronos encoder embedding ([REG] token)
    timestamp: int              # Position in the stream
    uncertainty: float          # Model's uncertainty at prediction time
    error: Optional[float] = None  # Prediction error (filled in later)


class SelfRetrievalKB:
    def __init__(
        self,
        embedding_dim: int,
        max_entries: int = 10000,
        uncertainty_window: int = 50,
    ):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries

        # FAISS flat index (exact search, supports incremental adds)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.entries: List[KBEntry] = []

        # Uncertainty tracking for adaptive gating
        self.recent_uncertainties = deque(maxlen=uncertainty_window)
        self.recent_errors = deque(maxlen=uncertainty_window)

    def add(self, entry: KBEntry):
        """Add an entry to the KB."""
        if len(self.entries) >= self.max_entries:
            self._evict_oldest()

        self.index.add(entry.embedding.reshape(1, -1).astype(np.float32))
        self.entries.append(entry)
        self.recent_uncertainties.append(entry.uncertainty)

    def update_error(self, idx: int, error: float):
        """Update the error for an entry once ground truth is available."""
        self.entries[idx].error = error
        self.recent_errors.append(error)

    def should_retrieve(self, current_uncertainty: float, z_threshold: float = 1.0) -> bool:
        """
        Uncertainty-gated retrieval decision.
        Retrieve only when current uncertainty is unusually high.
        """
        if len(self.recent_uncertainties) < 10:
            return True  # Always retrieve during cold start

        mean_u = np.mean(self.recent_uncertainties)
        std_u = np.std(self.recent_uncertainties) + 1e-8
        z_score = (current_uncertainty - mean_u) / std_u

        return z_score > z_threshold

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        use_error_weighting: bool = True,
        recency_decay: float = 0.999,
    ) -> List[Tuple[KBEntry, float]]:
        """
        Retrieve top-k entries by error-weighted similarity.
        Score = similarity x error_weight x recency_weight
        """
        if self.index.ntotal == 0:
            return []

        actual_k = min(k * 3, self.index.ntotal)

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            actual_k
        )

        # Convert L2 distance to similarity
        similarities = 1.0 / (1.0 + distances[0])

        scored_results = []
        current_time = len(self.entries)
        errors_list = [e for e in self.recent_errors if e is not None]
        mean_error = np.mean(errors_list) if errors_list else 1.0

        for sim, idx in zip(similarities, indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]

            # Error weight: high past error -> more informative pattern
            if use_error_weighting and entry.error is not None:
                error_weight = entry.error / (mean_error + 1e-8)
                error_weight = np.clip(error_weight, 0.5, 3.0)
            else:
                error_weight = 1.0

            # Recency decay based on entry index, not timestamp
            entry_idx = idx  # FAISS index = insertion order
            age = max(0, len(self.entries) - 1 - entry_idx)
            recency_weight = recency_decay ** age

            score = sim * error_weight * recency_weight
            scored_results.append((entry, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:k]

    def _evict_oldest(self):
        """Remove oldest entry. Rebuild FAISS index."""
        self.entries.pop(0)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if self.entries:
            embeddings = np.stack([e.embedding for e in self.entries]).astype(np.float32)
            self.index.add(embeddings)

    @property
    def size(self) -> int:
        return len(self.entries)
