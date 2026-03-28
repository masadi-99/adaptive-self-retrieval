"""
Fusion strategies for combining base forecasts with retrieved patterns.
"""

import numpy as np
from typing import List, Tuple
from knowledge_base import KBEntry


class ForecastEnsembleFusion:
    """
    Distribution-normalized forecast ensemble (k=1 nearest neighbor).
    Normalize the retrieved window to match current context distribution,
    re-forecast, and ensemble with the base forecast.
    """

    def __init__(self, extractor, base_weight: float = 2.0):
        self.extractor = extractor
        self.base_weight = base_weight

    def fuse(
        self,
        base_forecast: np.ndarray,
        base_uncertainty: float,
        retrieved: List[Tuple[KBEntry, float]],
        prediction_length: int,
        current_context: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        if not retrieved or current_context is None:
            return base_forecast

        ctx_mean = current_context.mean()
        ctx_std = current_context.std() + 1e-8

        # Use only the nearest neighbor (k=1)
        entry, score = retrieved[0]
        if entry.window is None:
            return base_forecast

        rw = entry.window
        rw_norm = (rw - rw.mean()) / (rw.std() + 1e-8) * ctx_std + ctx_mean
        nn_pred = self.extractor.predict_median(rw_norm, prediction_length)

        sim = score  # already 1/(1+dist)
        ensemble = (self.base_weight * base_forecast + sim * nn_pred) / (self.base_weight + sim)
        return ensemble


class ErrorCorrectionFusion:
    """
    Forecast correction transfer: use the error patterns from similar past windows
    to correct the current forecast. If Chronos systematically errs for a pattern,
    apply the opposite correction.

    Requires KB entries to have precomputed predictions stored.
    """

    def __init__(self, extractor, correction_scale: float = 0.3, k_correct: int = 3):
        self.extractor = extractor
        self.correction_scale = correction_scale
        self.k_correct = k_correct

    def fuse(
        self,
        base_forecast: np.ndarray,
        base_uncertainty: float,
        retrieved: List[Tuple[KBEntry, float]],
        prediction_length: int,
        **kwargs,
    ) -> np.ndarray:
        if not retrieved:
            return base_forecast

        base_scale = base_forecast.std() + 1e-8

        err_shapes = []
        total_w = 0.0
        for entry, score in retrieved[:self.k_correct]:
            if entry.future is None or entry.error is None:
                continue
            # Error: what Chronos missed (actual - predicted)
            # We need the precomputed prediction for this entry
            # Use the stored error directly if available
            if hasattr(entry, 'prediction') and entry.prediction is not None:
                err = entry.future[:prediction_length] - entry.prediction[:prediction_length]
            else:
                # Fallback: compute error from uncertainty as proxy
                continue

            err_shape = (err - err.mean()) / (err.std() + 1e-8)
            err_shapes.append(score * err_shape)
            total_w += score

        if total_w == 0:
            return base_forecast

        avg_err_shape = sum(err_shapes) / total_w
        correction = avg_err_shape * base_scale * self.correction_scale
        return base_forecast + correction


class CombinedFusion:
    """
    Best approach: Ensemble + Error Correction.
    1. Ensemble: normalize NN window, re-forecast, blend with base (k=1)
    2. Error correction: apply normalized error pattern from top-k NNs
    """

    def __init__(self, extractor, base_weight: float = 2.0,
                 correction_scale: float = 0.3, correction_weight: float = 0.5,
                 k_correct: int = 3):
        self.extractor = extractor
        self.ensemble = ForecastEnsembleFusion(extractor, base_weight=base_weight)
        self.correction_scale = correction_scale
        self.correction_weight = correction_weight
        self.k_correct = k_correct

    def fuse(
        self,
        base_forecast: np.ndarray,
        base_uncertainty: float,
        retrieved: List[Tuple[KBEntry, float]],
        prediction_length: int,
        current_context: np.ndarray = None,
        kb_predictions: dict = None,
        **kwargs,
    ) -> np.ndarray:
        # Step 1: Ensemble forecast
        ens_forecast = self.ensemble.fuse(
            base_forecast, base_uncertainty, retrieved, prediction_length,
            current_context=current_context,
        )

        # Step 2: Error correction from top-k
        if kb_predictions is None:
            return ens_forecast

        base_scale = base_forecast.std() + 1e-8
        err_shapes = []
        total_w = 0.0

        for entry, score in retrieved[:self.k_correct]:
            if entry.future is None:
                continue
            # Look up precomputed prediction for this KB entry
            entry_id = id(entry)
            if entry_id not in kb_predictions:
                continue
            kb_pred = kb_predictions[entry_id]
            future = entry.future[:prediction_length]
            if len(future) < prediction_length or len(kb_pred) < prediction_length:
                continue

            err = future - kb_pred[:prediction_length]
            err_shape = (err - err.mean()) / (err.std() + 1e-8)
            err_shapes.append(score * err_shape)
            total_w += score

        if total_w > 0:
            avg_err_shape = sum(err_shapes) / total_w
            correction = avg_err_shape * base_scale * self.correction_scale * self.correction_weight
            ens_forecast = ens_forecast + correction

        return ens_forecast


class ShapeBlendFusion:
    """Simple shape blending (z-score normalized futures)."""

    def __init__(self, extractor, alpha: float = 0.5):
        self.extractor = extractor
        self.alpha = alpha

    def fuse(self, base_forecast, base_uncertainty, retrieved, prediction_length, **kwargs):
        if not retrieved:
            return base_forecast

        base_mean = np.mean(base_forecast)
        base_std = np.std(base_forecast) + 1e-8
        base_shape = (base_forecast - base_mean) / base_std

        weighted_shapes = []
        total_weight = 0.0
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
