"""
CGRC: Consensus-Gated Retrieval Correction.
Simplified method: CGPF consensus gate + uncertainty scaling. No CRC.
"""

import numpy as np
import faiss


class CGRC:
    """
    Consensus-Gated Retrieval Correction.

    Retrieves k neighbors, re-forecasts from each (distribution-normalized),
    and applies per-timestep median correction ONLY where a supermajority
    of neighbors agree on the correction direction.

    No learned parameters. No model internals needed (black-box compatible).
    """

    def __init__(self, extractor, k: int = 5, beta: float = 0.3, gate_lo: float = 0.4, gate_hi: float = 0.8):
        self.extractor = extractor
        self.k = k
        self.beta = beta
        self.gate_lo = gate_lo
        self.gate_hi = gate_hi

    def correct(self, base_forecast, uncertainty, context,
                kb_windows, kb_futures, faiss_index,
                unc_mean=None, unc_std=None):
        """
        Apply consensus-gated correction to base_forecast.

        Returns: corrected forecast, metadata dict
        """
        pl = len(base_forecast)
        ctx_m, ctx_s = context.mean(), context.std() + 1e-8

        if faiss_index.ntotal < self.k:
            return base_forecast, {'gate_rate': 0.0, 'mean_agreement': 0.0, 'corrected': False}

        emb = self.extractor.extract_embedding(context).reshape(1, -1).astype(np.float32)
        D, I = faiss_index.search(emb, self.k)

        nn_forecasts = []
        for j in range(min(self.k, len(I[0]))):
            idx = I[0, j]
            if idx < 0 or idx >= len(kb_windows):
                continue
            rw = kb_windows[idx]
            rw_m, rw_s = rw.mean(), rw.std() + 1e-8
            rw_norm = (rw - rw_m) / rw_s * ctx_s + ctx_m
            nn_pred = self.extractor.predict_median(rw_norm, pl)
            nn_forecasts.append(nn_pred)

        if len(nn_forecasts) < 3:
            return base_forecast, {'gate_rate': 0.0, 'mean_agreement': 0.0, 'corrected': False}

        nn_forecasts = np.array(nn_forecasts)

        # Per-timestep corrections and consensus
        corrections = nn_forecasts - base_forecast
        signs = np.sign(corrections)
        agreement = np.abs(np.mean(signs, axis=0))
        median_correction = np.median(corrections, axis=0)

        # Soft consensus gate
        gate = np.clip((agreement - self.gate_lo) / (self.gate_hi - self.gate_lo), 0, 1)

        # Uncertainty scaling (optional)
        unc_scale = 1.0
        if unc_mean is not None and unc_std is not None:
            unc_z = (uncertainty - unc_mean) / (unc_std + 1e-8)
            unc_scale = 0.3 + 0.7 / (1.0 + np.exp(-unc_z))

        # Apply gated correction
        corrected = base_forecast + self.beta * unc_scale * gate * median_correction

        meta = {
            'gate_rate': float(np.mean(gate > 0)),
            'mean_agreement': float(np.mean(agreement)),
            'mean_correction': float(np.mean(np.abs(self.beta * unc_scale * gate * median_correction))),
            'corrected': True,
        }
        return corrected, meta
