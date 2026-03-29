"""
RCAF: Retrieval-Calibrated Adaptive Forecasting.
Three components: CRC (conformal bias correction), CGPF (consensus-gated
per-timestep fusion), and uncertainty scaling for robustness.
"""

import numpy as np
from typing import List, Tuple
from knowledge_base import KBEntry


class RCAF:
    """
    Retrieval-Calibrated Adaptive Forecasting.

    Uses k retrieved neighbors' known forecast errors as a per-timestep
    calibration signal, gated by cross-neighbor consensus.

    Three interlocking mechanisms:
    1. CRC: Conformal Retrieval Calibration — correct systematic model biases
       using median residuals across neighbors
    2. CGPF: Consensus-Gated Per-Timestep Fusion — only apply directional
       corrections where neighbors agree
    3. Uncertainty scaling: reduce correction magnitude on easy (low uncertainty)
       windows to prevent degradation
    """

    def __init__(self, extractor, k: int = 7, alpha: float = 0.35,
                 beta: float = 0.25):
        self.extractor = extractor
        self.k = k
        self.alpha = alpha  # CRC damping
        self.beta = beta    # CGPF damping

    def fuse(self, base_forecast, base_uncertainty, retrieved,
             prediction_length, current_context=None,
             kb_windows=None, kb_futures=None, faiss_index=None,
             unc_mean=None, unc_std=None, **kwargs):
        """
        Apply RCAF: CRC bias correction + CGPF consensus correction.

        Requires access to KB windows, futures, and FAISS index directly
        (bypasses the KBEntry-based retrieval for efficiency).
        """
        if current_context is None or faiss_index is None:
            return base_forecast
        if kb_windows is None or kb_futures is None:
            return base_forecast

        ctx_mean = current_context.mean()
        ctx_std = current_context.std() + 1e-8
        pl = prediction_length

        emb = self.extractor.extract_embedding(current_context)
        emb = emb.reshape(1, -1).astype(np.float32)
        D, I = faiss_index.search(emb, self.k)

        nn_forecasts = []
        residuals = []

        for j in range(min(self.k, len(I[0]))):
            idx = I[0, j]
            if idx < 0 or idx >= len(kb_windows):
                continue
            rw = kb_windows[idx]
            rf = kb_futures[idx][:pl]
            if len(rf) < pl:
                continue

            rw_mean, rw_std = rw.mean(), rw.std() + 1e-8

            # Normalize window and future to current context distribution
            rw_norm = (rw - rw_mean) / rw_std * ctx_std + ctx_mean
            rf_adj = (rf - rw_mean) / rw_std * ctx_std + ctx_mean

            # Re-forecast from normalized window
            rp = self.extractor.predict_median(rw_norm, pl)
            nn_forecasts.append(rp)

            # Calibration residual: model prediction - adjusted truth
            residuals.append(rp - rf_adj)

        if len(nn_forecasts) < 3:
            return base_forecast

        nn_forecasts = np.array(nn_forecasts)
        residuals = np.array(residuals)

        # --- CRC: Conformal bias correction ---
        bias = np.median(residuals, axis=0)
        mad = np.median(np.abs(residuals - bias), axis=0)
        crc_conf = 1.0 / (1.0 + mad / (np.abs(bias) + 1e-8))

        # --- CGPF: Consensus-gated per-timestep correction ---
        corrections = nn_forecasts - base_forecast
        agreement = np.abs(np.mean(np.sign(corrections), axis=0))
        median_corr = np.median(corrections, axis=0)
        cgpf_gate = np.clip((agreement - 0.4) / 0.4, 0, 1)  # soft gate

        # --- Uncertainty scaling ---
        unc_scale = 1.0
        if unc_mean is not None and unc_std is not None:
            unc_z = (base_uncertainty - unc_mean) / (unc_std + 1e-8)
            # High uncertainty -> scale ~1.0; low uncertainty -> scale ~0.3
            unc_scale = 0.3 + 0.7 / (1.0 + np.exp(-unc_z))

        # --- Integration ---
        final = base_forecast.copy()
        final -= self.alpha * unc_scale * crc_conf * bias
        final += self.beta * unc_scale * cgpf_gate * median_corr

        return final


class ForecastEnsembleFusion:
    """Distribution-normalized k=1 ensemble (previous best method)."""

    def __init__(self, extractor, base_weight: float = 2.0):
        self.extractor = extractor
        self.base_weight = base_weight

    def fuse(self, base_forecast, base_uncertainty, retrieved,
             prediction_length, current_context=None, **kwargs):
        if not retrieved or current_context is None:
            return base_forecast
        ctx_mean = current_context.mean()
        ctx_std = current_context.std() + 1e-8
        entry, score = retrieved[0]
        if entry.window is None:
            return base_forecast
        rw = entry.window
        rw_norm = (rw - rw.mean()) / (rw.std() + 1e-8) * ctx_std + ctx_mean
        nn_pred = self.extractor.predict_median(rw_norm, prediction_length)
        sim = score
        return (self.base_weight * base_forecast + sim * nn_pred) / (self.base_weight + sim)


class ShapeBlendFusion:
    """Z-score shape blending."""

    def __init__(self, extractor, alpha: float = 0.5):
        self.extractor = extractor
        self.alpha = alpha

    def fuse(self, base_forecast, base_uncertainty, retrieved,
             prediction_length, **kwargs):
        if not retrieved:
            return base_forecast
        base_mean = np.mean(base_forecast)
        base_std = np.std(base_forecast) + 1e-8
        base_shape = (base_forecast - base_mean) / base_std
        weighted_shapes, total_weight = [], 0.0
        for entry, score in retrieved:
            if entry.future is not None and len(entry.future) >= prediction_length:
                future = entry.future[:prediction_length]
                f_shape = (future - future.mean()) / (future.std() + 1e-8)
                weighted_shapes.append(score * f_shape)
                total_weight += score
        if total_weight == 0:
            return base_forecast
        avg_shape = np.sum(weighted_shapes, axis=0) / total_weight
        blended = ((1 - self.alpha) * base_shape + self.alpha * avg_shape) * base_std + base_mean
        return blended
