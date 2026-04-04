"""
RERC: Retrieve Errors, not Examples — Risk-Calibrated Forecast Intervention.

Memory bank stores (context, forecast, residual, embedding, forecast_descriptor).
Retrieval uses dual keys: context embedding + forecast descriptor.
Intervention score combines sign agreement, magnitude agreement, density, calibration.
Threshold calibrated on validation to control harmful intervention rate.
"""

import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class ResidualEntry:
    """A single entry in the residual memory bank."""
    context: np.ndarray          # Raw context window (L,)
    forecast: np.ndarray         # Model's own forecast ŷ (H,)
    actual: np.ndarray           # Ground truth future f (H,)
    residual: np.ndarray         # r = f - ŷ (H,)
    context_embedding: np.ndarray  # Embedding of context
    forecast_descriptor: np.ndarray  # Descriptor of forecast behavior
    context_scale: float         # std of context (for residual alignment)
    context_level: float         # mean of context


class ResidualMemoryBank:
    """
    Memory bank that stores model errors, not examples.
    Dual-key retrieval: context embedding + forecast descriptor.
    """

    def __init__(self, context_emb_dim: int, forecast_desc_dim: int,
                 alpha_context: float = 0.7):
        self.context_emb_dim = context_emb_dim
        self.forecast_desc_dim = forecast_desc_dim
        self.combined_dim = context_emb_dim + forecast_desc_dim
        self.alpha = alpha_context  # weight for context vs forecast in retrieval

        self.entries: List[ResidualEntry] = []
        self.index = faiss.IndexFlatL2(self.combined_dim)

    def compute_forecast_descriptor(self, forecast: np.ndarray, n_segments: int = 8) -> np.ndarray:
        """
        Extract behavioral descriptor from a forecast.
        Captures shape, trend, and variability — not raw values.
        """
        H = len(forecast)
        seg_len = max(1, H // n_segments)
        desc = []
        # Z-normalize forecast first (scale-invariant)
        f_z = (forecast - forecast.mean()) / (forecast.std() + 1e-8)

        for i in range(n_segments):
            seg = f_z[i*seg_len:min((i+1)*seg_len, H)]
            if len(seg) == 0:
                desc.extend([0, 0, 0])
                continue
            desc.append(float(seg.mean()))      # level
            desc.append(float(seg[-1] - seg[0]))  # trend
            desc.append(float(seg.std()))        # variability

        # Global descriptors
        desc.append(float(np.diff(f_z).mean()))    # avg change
        desc.append(float(np.diff(f_z).std()))     # change volatility
        desc.append(float(f_z.max() - f_z.min()))  # range

        arr = np.array(desc, dtype=np.float32)
        # Pad or truncate to forecast_desc_dim
        if len(arr) < self.forecast_desc_dim:
            arr = np.pad(arr, (0, self.forecast_desc_dim - len(arr)))
        else:
            arr = arr[:self.forecast_desc_dim]
        return arr

    def _make_key(self, context_emb: np.ndarray, forecast_desc: np.ndarray) -> np.ndarray:
        """Combine context embedding and forecast descriptor into retrieval key."""
        # Scale to balance contributions
        c = context_emb / (np.linalg.norm(context_emb) + 1e-8) * self.alpha
        f = forecast_desc / (np.linalg.norm(forecast_desc) + 1e-8) * (1 - self.alpha)
        return np.concatenate([c, f]).astype(np.float32)

    def add(self, entry: ResidualEntry):
        """Add an entry to the memory bank."""
        key = self._make_key(entry.context_embedding, entry.forecast_descriptor)
        self.index.add(key.reshape(1, -1))
        self.entries.append(entry)

    def retrieve(self, context_emb: np.ndarray, forecast_desc: np.ndarray,
                 k: int = 7) -> List[Tuple[ResidualEntry, float]]:
        """Retrieve k nearest entries by dual-key similarity."""
        if self.index.ntotal == 0:
            return []

        key = self._make_key(context_emb, forecast_desc)
        actual_k = min(k, self.index.ntotal)
        D, I = self.index.search(key.reshape(1, -1), actual_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            sim = 1.0 / (1.0 + dist)
            results.append((self.entries[idx], sim))
        return results

    @property
    def size(self) -> int:
        return len(self.entries)


class InterventionScorer:
    """
    Computes intervention score from retrieved residuals.
    High score = strong evidence that correction will help.
    """

    def compute_score(self, retrieved_residuals: np.ndarray,
                      distances: np.ndarray,
                      prediction_length: int) -> Tuple[np.ndarray, Dict]:
        """
        Compute per-timestep intervention scores.

        The score is primarily driven by SIGN AGREEMENT among retrieved
        residuals. When residuals consistently point in the same direction,
        there's strong evidence of a systematic model error at that timestep.

        Args:
            retrieved_residuals: (k, H) array of aligned residuals
            distances: (k,) array of retrieval distances
            prediction_length: H

        Returns:
            scores: (H,) per-timestep intervention scores in [0, 1]
            meta: dict with diagnostic info
        """
        k = len(retrieved_residuals)
        if k < 2:
            return np.zeros(prediction_length), {'intervene': False}

        # 1. Sign agreement: primary signal
        signs = np.sign(retrieved_residuals)
        sign_agreement = np.abs(np.mean(signs, axis=0))  # (H,)

        # 2. Magnitude consistency: secondary signal (penalize high variance)
        magnitudes = np.abs(retrieved_residuals)
        mag_cv = np.std(magnitudes, axis=0) / (np.mean(magnitudes, axis=0) + 1e-8)
        mag_consistency = np.clip(1.0 - 0.5 * mag_cv, 0, 1)  # (H,)

        # 3. Score = sign agreement * magnitude consistency
        # Sign agreement is the gate; magnitude consistency modulates strength
        score = sign_agreement * mag_consistency  # (H,)

        meta = {
            'mean_sign_agreement': float(np.mean(sign_agreement)),
            'mean_mag_consistency': float(np.mean(mag_consistency)),
            'mean_score': float(np.mean(score)),
        }
        return score, meta


class RERC:
    """
    Retrieve Errors, not Examples — Risk-Calibrated Forecast Intervention.

    Full pipeline: build residual memory → retrieve → align → score → intervene.
    """

    def __init__(self, extractor, k: int = 7, beta: float = 0.4,
                 forecast_desc_dim: int = 27, alpha_context: float = 0.7,
                 intervention_threshold: float = 0.15):
        self.extractor = extractor
        self.k = k
        self.beta = beta  # correction strength
        self.forecast_desc_dim = forecast_desc_dim
        self.scorer = InterventionScorer()
        self.intervention_threshold = intervention_threshold

        self.memory = ResidualMemoryBank(
            context_emb_dim=extractor.embedding_dim,
            forecast_desc_dim=forecast_desc_dim,
            alpha_context=alpha_context,
        )

    def build_memory(self, train_data: np.ndarray, window_size: int,
                     prediction_length: int, stride: int = 64):
        """Build residual memory bank from training data."""
        count = 0
        for start in range(0, len(train_data) - window_size - prediction_length, stride):
            ctx = train_data[start:start + window_size]
            actual = train_data[start + window_size:start + window_size + prediction_length]
            if len(actual) < prediction_length:
                continue

            # Get model's forecast for this context
            forecast = self.extractor.predict_median(ctx, prediction_length)
            residual = actual - forecast  # r = f - ŷ

            # Embeddings
            ctx_emb = self.extractor.extract_embedding(ctx)
            fc_desc = self.memory.compute_forecast_descriptor(forecast)

            entry = ResidualEntry(
                context=ctx,
                forecast=forecast,
                actual=actual,
                residual=residual,
                context_embedding=ctx_emb,
                forecast_descriptor=fc_desc,
                context_scale=float(ctx.std()),
                context_level=float(ctx.mean()),
            )
            self.memory.add(entry)
            count += 1
            if count % 50 == 0:
                print(f"    Memory: {count} entries...")

        print(f"  Memory bank: {count} entries")

    def intervene(self, base_forecast: np.ndarray, context: np.ndarray,
                  uncertainty: float = 0.0,
                  unc_mean: float = 0.0, unc_std: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Decide whether and how to intervene on the base forecast.

        Uses a TWO-SIGNAL approach:
        1. STORED RESIDUALS → intervention score (should we correct?)
        2. RE-FORECASTS → correction magnitude (how much to correct?)

        The stored residuals tell us WHETHER the model historically errs in
        this pattern. The re-forecasts tell us WHAT the correction should be.

        Returns:
            corrected_forecast: (H,) — may equal base_forecast if no intervention
            meta: dict with intervention diagnostics
        """
        pl = len(base_forecast)
        ctx_scale = context.std() + 1e-8
        ctx_level = context.mean()

        if self.memory.size < 3:
            return base_forecast, {'intervened': False, 'reason': 'insufficient_memory',
                                   'intervention_rate': 0, 'mean_score': 0,
                                   'mean_sign_agreement': 0, 'mean_mag_consistency': 0,
                                   'n_neighbors': 0}

        # Get retrieval keys
        ctx_emb = self.extractor.extract_embedding(context)
        fc_desc = self.memory.compute_forecast_descriptor(base_forecast)

        # Retrieve
        retrieved = self.memory.retrieve(ctx_emb, fc_desc, k=self.k)
        if len(retrieved) < 3:
            return base_forecast, {'intervened': False, 'reason': 'too_few_neighbors',
                                   'intervention_rate': 0, 'mean_score': 0,
                                   'mean_sign_agreement': 0, 'mean_mag_consistency': 0,
                                   'n_neighbors': len(retrieved)}

        # SIGNAL 1: Stored residuals → intervention score
        aligned_residuals = []
        distances = []
        for entry, sim in retrieved:
            scale_ratio = ctx_scale / (entry.context_scale + 1e-8)
            aligned_r = entry.residual[:pl] * scale_ratio
            if len(aligned_r) < pl:
                aligned_r = np.pad(aligned_r, (0, pl - len(aligned_r)))
            aligned_residuals.append(aligned_r)
            distances.append(1.0 / (sim + 1e-8) - 1.0)

        aligned_residuals = np.array(aligned_residuals)
        distances = np.array(distances)

        scores, score_meta = self.scorer.compute_score(aligned_residuals, distances, pl)

        # SIGNAL 2: Re-forecasts → correction direction and magnitude
        nn_forecasts = []
        for entry, sim in retrieved:
            rw = entry.context
            rw_m, rw_s = rw.mean(), rw.std() + 1e-8
            rw_norm = (rw - rw_m) / rw_s * ctx_scale + ctx_level
            nn_pred = self.extractor.predict_median(rw_norm, pl)
            nn_forecasts.append(nn_pred)

        nn_forecasts = np.array(nn_forecasts)
        corrections = nn_forecasts - base_forecast
        median_correction = np.median(corrections, axis=0)

        # Re-forecast consensus (reinforces stored-residual signal)
        reforecast_agreement = np.abs(np.mean(np.sign(corrections), axis=0))

        # Combined score: stored residual agreement * re-forecast agreement
        combined_score = scores * reforecast_agreement

        # Uncertainty scaling
        unc_z = (uncertainty - unc_mean) / (unc_std + 1e-8)
        unc_scale = 0.3 + 0.7 / (1.0 + np.exp(-unc_z))

        # Soft intervention gate
        soft_mask = np.clip((combined_score - self.intervention_threshold) /
                           (1.0 - self.intervention_threshold + 1e-8), 0, 1)

        correction = self.beta * unc_scale * soft_mask * median_correction
        corrected = base_forecast + correction

        intervention_rate = float(np.mean(combined_score >= self.intervention_threshold))

        meta = {
            'intervened': intervention_rate > 0,
            'intervention_rate': intervention_rate,
            'mean_score': float(np.mean(combined_score)),
            'n_neighbors': len(retrieved),
            'unc_scale': float(unc_scale),
            'mean_correction': float(np.mean(np.abs(correction))),
            **score_meta,
        }
        return corrected, meta

    def calibrate_threshold(self, val_data: np.ndarray, train_data: np.ndarray,
                           window_size: int, prediction_length: int,
                           target_hir: float = 0.10) -> float:
        """
        Calibrate intervention threshold on validation data to achieve
        a target harmful intervention rate (HIR).

        HIR = fraction of intervened windows where MSE increases.
        """
        # Build memory from train only
        self.memory = ResidualMemoryBank(
            self.memory.context_emb_dim, self.memory.forecast_desc_dim,
            self.memory.alpha
        )
        self.build_memory(train_data, window_size, prediction_length, stride=64)

        # Evaluate on val at different thresholds
        positions = list(range(0, len(val_data) - window_size - prediction_length + 1,
                              prediction_length))
        if not positions:
            return self.intervention_threshold

        all_uncs = [self.extractor.predict_with_uncertainty(
            val_data[s:s+window_size], prediction_length)[2] for s in positions]
        um, us = np.mean(all_uncs), np.std(all_uncs) + 1e-8

        # Collect per-window data at different thresholds
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        best_threshold = 0.15
        best_gain = -999

        for thresh in thresholds:
            self.intervention_threshold = thresh
            n_intervened = 0
            n_harmful = 0
            total_gain = 0

            for s in positions:
                ctx = val_data[s:s+window_size]
                gt = val_data[s+window_size:s+window_size+prediction_length]
                base, _, unc = self.extractor.predict_with_uncertainty(ctx, prediction_length)
                base_mse = float(np.mean((base - gt)**2))

                corrected, meta = self.intervene(base, ctx, unc, um, us)
                if meta['intervention_rate'] > 0:
                    n_intervened += 1
                    corr_mse = float(np.mean((corrected - gt)**2))
                    if corr_mse > base_mse:
                        n_harmful += 1
                    total_gain += (base_mse - corr_mse) / base_mse * 100

            hir = n_harmful / max(n_intervened, 1)
            avg_gain = total_gain / max(n_intervened, 1)

            if hir <= target_hir and avg_gain > best_gain:
                best_gain = avg_gain
                best_threshold = thresh

        self.intervention_threshold = best_threshold
        print(f"  Calibrated threshold: {best_threshold} (target HIR={target_hir})")
        return best_threshold
