"""
TimesFM adapter for RCAF cross-model evaluation.
Uses timesfm pip package with TimesFM 1.0-200m checkpoint.
Embedding: functional embedding (forecasts at multiple offsets).
"""

import numpy as np


class TimesFMAdapter:
    def __init__(self, device="cuda"):
        import timesfm
        backend = 'gpu' if device == 'cuda' else 'cpu'
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id='google/timesfm-1.0-200m-pytorch'
            ),
        )
        # Functional embedding: forecasts at 4 offsets, 64 steps each
        self._offsets = [0, 32, 64, 128]
        self._emb_forecast_len = 64
        self.embedding_dim = len(self._offsets) * self._emb_forecast_len  # 256

    def _forecast_single(self, context, horizon):
        """Run TimesFM on a single context, return mean forecast."""
        ctx = np.asarray(context, dtype=np.float32)
        result = self.tfm.forecast([ctx], freq=[0])
        # result is (point_forecast, quantile_forecast)
        # point_forecast: (1, horizon_len), quantile: (1, horizon_len, num_quantiles)
        pred = result[0][0]  # (horizon_len,)
        pred = pred.astype(np.float32)
        if len(pred) >= horizon:
            return pred[:horizon]
        # Chain for longer horizons
        full_pred = pred.tolist()
        ext = np.concatenate([ctx, pred])
        while len(full_pred) < horizon:
            rem = horizon - len(full_pred)
            ctx_len = min(len(ctx), len(ext))
            chunk_result = self.tfm.forecast([ext[-ctx_len:].astype(np.float32)], freq=[0])
            chunk = chunk_result[0][0].astype(np.float32)
            take = min(len(chunk), rem)
            full_pred.extend(chunk[:take].tolist())
            ext = np.concatenate([ext, chunk[:take]])
        return np.array(full_pred[:horizon], dtype=np.float32)

    def extract_embedding(self, window: np.ndarray) -> np.ndarray:
        """Functional embedding: z-normalized forecasts at multiple offsets."""
        parts = []
        for off in self._offsets:
            sub = window[off:]
            if len(sub) < 128:
                sub = window
            pred = self._forecast_single(sub, self._emb_forecast_len)
            pred_z = (pred - pred.mean()) / (pred.std() + 1e-8)
            parts.append(pred_z)
        return np.concatenate(parts).astype(np.float32)

    def predict_median(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        return self._forecast_single(context, prediction_length)

    def predict_with_uncertainty(self, context: np.ndarray, prediction_length: int):
        ctx = np.asarray(context, dtype=np.float32)
        result = self.tfm.forecast([ctx], freq=[0])
        pred = result[0][0][:prediction_length].astype(np.float32)
        # Quantile spread for uncertainty
        quantiles = result[1][0][:prediction_length]  # (pl, num_quantiles)
        uncertainty = float(np.mean(quantiles[:, -1] - quantiles[:, 0]))
        return pred, np.stack([pred]*3), uncertainty
