"""
Extract embeddings from Chronos-Bolt's encoder for retrieval.
Uses the built-in embed() method which returns encoder hidden states.
The [REG] token (last position) serves as a summary vector (like BERT's [CLS]).
"""

import torch
import numpy as np
from chronos import ChronosBoltPipeline
from typing import List, Tuple, Optional


class ChronosEmbeddingExtractor:
    def __init__(self, model_name="amazon/chronos-bolt-small", device="cpu"):
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
        self.device = device
        self.model = self.pipeline.model
        # Determine embedding dim from model config
        self.embedding_dim = self.model.config.d_model

    def extract_embedding(self, window: np.ndarray) -> np.ndarray:
        """
        Extract encoder embedding for a single window.
        Uses the [REG] token (last position) as the summary representation.
        Returns: embedding vector of shape (embedding_dim,)
        """
        context = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # embed() returns (embeddings, (loc, scale))
            # embeddings shape: (1, num_patches+1, d_model)
            # Last token is the [REG] token
            embeddings, _ = self.pipeline.embed(context)
            reg_embedding = embeddings[0, -1, :]  # [REG] token
        return reg_embedding.cpu().float().numpy()

    def extract_embeddings_batch(
        self, windows: List[np.ndarray], batch_size: int = 32
    ) -> np.ndarray:
        """Extract embeddings for multiple windows. Returns (n, embedding_dim)."""
        all_embeddings = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            # Pad to same length if needed
            max_len = max(len(w) for w in batch)
            padded = []
            for w in batch:
                if len(w) < max_len:
                    pad = np.full(max_len - len(w), np.nan)
                    padded.append(np.concatenate([pad, w]))
                else:
                    padded.append(w)
            context = torch.tensor(np.stack(padded), dtype=torch.float32)
            with torch.no_grad():
                embeddings, _ = self.pipeline.embed(context)
                reg_embeddings = embeddings[:, -1, :]  # [REG] tokens
            all_embeddings.append(reg_embeddings.cpu().float().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def predict_with_uncertainty(
        self, context: np.ndarray, prediction_length: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Produce forecast and uncertainty estimate.
        Returns: (median_forecast, quantile_forecasts, uncertainty_score)
        """
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # predict_quantiles allows custom quantile levels
        quantile_levels = [0.1, 0.5, 0.9]
        quantiles, mean = self.pipeline.predict_quantiles(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )
        # quantiles shape: (1, prediction_length, num_quantiles)
        # mean shape: (1, prediction_length)
        q10 = quantiles[0, :, 0].numpy()
        median = quantiles[0, :, 1].numpy()
        q90 = quantiles[0, :, 2].numpy()

        # Uncertainty = mean width of 80% prediction interval
        uncertainty = float(np.mean(q90 - q10))

        # Return all quantiles as (num_quantiles, prediction_length) for compatibility
        all_quantiles = quantiles[0].numpy().T  # (3, prediction_length)

        return median, all_quantiles, uncertainty

    def predict_median(
        self, context: np.ndarray, prediction_length: int
    ) -> np.ndarray:
        """Quick median-only prediction."""
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        quantiles, mean = self.pipeline.predict_quantiles(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
        )
        return mean[0].numpy()
