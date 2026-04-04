"""
REBI: Retrieval of Errors for Black-Box Intervention.

Core method: store model errors in memory, retrieve them at test time,
apply risk-calibrated corrections WITHOUT re-forecasting.

Test-time cost: 1 predict + 1 embed (NO extra model calls).
"""

import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional


def forecast_descriptor(y_hat: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Compact behavioral descriptor of a forecast (6-dim, scale-invariant)."""
    ctx_std = context.std() + 1e-8
    ctx_mean = context[-len(y_hat):].mean() if len(context) >= len(y_hat) else context.mean()
    y_norm = (y_hat - ctx_mean) / ctx_std
    H = len(y_hat)
    half = H // 2

    slope = float(np.polyfit(np.arange(H), y_norm, 1)[0])
    mean_ratio = float(y_hat.mean() / (ctx_mean + 1e-8))
    std_ratio = float(y_hat.std() / ctx_std)
    asymmetry = float(y_norm[:half].mean() - y_norm[half:].mean()) if half > 0 else 0.0
    roughness = float(np.max(np.abs(np.diff(y_norm)))) if H > 1 else 0.0
    ac1 = float(np.corrcoef(y_norm[:-1], y_norm[1:])[0, 1]) if H > 2 else 0.0
    if np.isnan(ac1): ac1 = 0.0

    return np.array([slope, mean_ratio, std_ratio, asymmetry, roughness, ac1],
                    dtype=np.float32)


class REBIMemory:
    """Dual-key residual memory bank."""

    def __init__(self, ctx_emb_dim: int, fc_desc_dim: int = 6,
                 w_ctx: float = 0.6, w_fcd: float = 0.4):
        self.ctx_emb_dim = ctx_emb_dim
        self.fc_desc_dim = fc_desc_dim
        self.w_ctx = w_ctx
        self.w_fcd = w_fcd

        self.ctx_index = faiss.IndexFlatL2(ctx_emb_dim)
        self.fcd_index = faiss.IndexFlatL2(fc_desc_dim)

        self.residuals: List[np.ndarray] = []   # (H,) each
        self.ctx_stds: List[float] = []
        self.forecasts: List[np.ndarray] = []   # (H,) each

    def add(self, ctx_emb: np.ndarray, fc_desc: np.ndarray,
            residual: np.ndarray, ctx_std: float, forecast: np.ndarray):
        self.ctx_index.add(ctx_emb.reshape(1, -1).astype(np.float32))
        self.fcd_index.add(fc_desc.reshape(1, -1).astype(np.float32))
        self.residuals.append(residual)
        self.ctx_stds.append(ctx_std)
        self.forecasts.append(forecast)

    def retrieve(self, ctx_emb: np.ndarray, fc_desc: np.ndarray,
                 k: int = 7) -> List[Tuple[int, float]]:
        """Dual-key retrieval. Returns list of (index, combined_score)."""
        if self.ctx_index.ntotal == 0:
            return []

        k_search = min(2 * k, self.ctx_index.ntotal)

        D_ctx, I_ctx = self.ctx_index.search(
            ctx_emb.reshape(1, -1).astype(np.float32), k_search)
        D_fcd, I_fcd = self.fcd_index.search(
            fc_desc.reshape(1, -1).astype(np.float32), k_search)

        sim_ctx = 1.0 / (1.0 + D_ctx[0])
        sim_fcd = 1.0 / (1.0 + D_fcd[0])

        # Merge candidates
        scores = {}
        for j, idx in enumerate(I_ctx[0]):
            if idx >= 0:
                scores[int(idx)] = scores.get(int(idx), 0) + self.w_ctx * sim_ctx[j]
        for j, idx in enumerate(I_fcd[0]):
            if idx >= 0:
                scores[int(idx)] = scores.get(int(idx), 0) + self.w_fcd * sim_fcd[j]

        top_k = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return top_k

    @property
    def size(self) -> int:
        return self.ctx_index.ntotal


class REBI:
    """
    Retrieval of Errors for Black-Box Intervention.

    Offline: build residual memory (1 predict per training window).
    Online: 1 predict + 1 embed per test window. NO re-forecasting.
    """

    def __init__(self, extractor, k: int = 7, target_risk: float = 0.10,
                 w_ctx: float = 0.6, w_fcd: float = 0.4):
        self.extractor = extractor
        self.k = k
        self.target_risk = target_risk
        self.tau = 0.3  # default threshold, calibrated on val
        self.cal_accuracy = 0.5

        self.memory = REBIMemory(
            ctx_emb_dim=extractor.embedding_dim,
            fc_desc_dim=6, w_ctx=w_ctx, w_fcd=w_fcd,
        )

    def build_memory(self, train_data: np.ndarray, ws: int, pl: int,
                     stride: int = 64):
        """Build residual memory bank. Requires 1 predict per window."""
        count = 0
        for s in range(0, len(train_data) - ws - pl, stride):
            ctx = train_data[s:s+ws]
            future = train_data[s+ws:s+ws+pl]
            if len(future) < pl:
                continue

            forecast = self.extractor.predict_median(ctx, pl)
            residual = future - forecast
            ctx_emb = self.extractor.extract_embedding(ctx)
            fc_desc = forecast_descriptor(forecast, ctx)

            self.memory.add(ctx_emb, fc_desc, residual,
                          float(ctx.std()), forecast)
            count += 1
            if count % 50 == 0:
                print(f"    Memory: {count}...")

        print(f"  Memory: {count} entries")

    def calibrate(self, val_data: np.ndarray, train_data: np.ndarray,
                  ws: int, pl: int):
        """Calibrate intervention threshold on validation data."""
        # Build memory from train only
        temp_memory = REBIMemory(
            self.extractor.embedding_dim, 6,
            self.memory.w_ctx, self.memory.w_fcd,
        )
        orig_memory = self.memory
        self.memory = temp_memory
        self.build_memory(train_data, ws, pl, stride=128)

        # Evaluate on val
        positions = list(range(0, len(val_data) - ws - pl + 1, pl))
        if not positions:
            self.memory = orig_memory
            return

        scores_and_helped = []
        for s in positions:
            ctx = val_data[s:s+ws]
            gt = val_data[s+ws:s+ws+pl]
            base = self.extractor.predict_median(ctx, pl)
            base_mse = float(np.mean((base - gt)**2))

            score, correction = self._compute_correction(base, ctx)
            if correction is not None:
                corrected = base + 0.5 * correction  # damped
                corr_mse = float(np.mean((corrected - gt)**2))
                helped = corr_mse < base_mse
                scores_and_helped.append((score, helped))

        # Find threshold
        if scores_and_helped:
            sorted_pairs = sorted(scores_and_helped, key=lambda x: -x[0])
            best_tau = 1.0
            n_int, n_hurt = 0, 0
            for score, helped in sorted_pairs:
                n_int += 1
                if not helped:
                    n_hurt += 1
                if n_hurt / n_int <= self.target_risk:
                    best_tau = score
            self.tau = best_tau
            self.cal_accuracy = sum(h for _, h in scores_and_helped) / len(scores_and_helped)
            print(f"  Calibrated: tau={self.tau:.3f}, cal_acc={self.cal_accuracy:.2f}")

        # Restore original memory (or rebuild from train+val)
        self.memory = orig_memory

    def _compute_correction(self, base_forecast: np.ndarray,
                           context: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """Compute intervention score and candidate correction."""
        pl = len(base_forecast)
        ctx_std = context.std() + 1e-8

        if self.memory.size < 3:
            return 0.0, None

        ctx_emb = self.extractor.extract_embedding(context)
        fc_desc = forecast_descriptor(base_forecast, context)
        retrieved = self.memory.retrieve(ctx_emb, fc_desc, k=self.k)

        if len(retrieved) < 3:
            return 0.0, None

        # Align residuals
        aligned = []
        for idx, sim in retrieved:
            if idx >= len(self.memory.residuals):
                continue
            r = self.memory.residuals[idx][:pl]
            entry_std = self.memory.ctx_stds[idx]
            scale = ctx_std / (entry_std + 1e-8)
            aligned_r = r * scale
            if len(aligned_r) < pl:
                aligned_r = np.pad(aligned_r, (0, pl - len(aligned_r)))
            aligned.append(aligned_r)

        if len(aligned) < 3:
            return 0.0, None

        aligned = np.array(aligned)

        # Intervention score: sign agreement * magnitude consistency
        signs = np.sign(aligned)
        sign_agr = np.abs(np.mean(signs, axis=0))  # (H,)
        abs_r = np.abs(aligned)
        mag_cv = np.std(abs_r, axis=0) / (np.mean(abs_r, axis=0) + 1e-8)
        mag_con = np.clip(1.0 - 0.5 * mag_cv, 0, 1)

        per_ts_score = sign_agr * mag_con
        score = float(np.mean(per_ts_score))

        # Candidate correction: median of aligned residuals
        correction = np.median(aligned, axis=0)

        return score, correction

    def correct(self, base_forecast: np.ndarray, context: np.ndarray,
                uncertainty: float = 0.0,
                unc_mean: float = 0.0, unc_std: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Apply risk-calibrated correction. NO re-forecasting.

        Returns: (corrected_forecast, metadata)
        """
        score, correction = self._compute_correction(base_forecast, context)

        if correction is None or score < self.tau:
            return base_forecast, {
                'intervened': False, 'score': score, 'threshold': self.tau,
                'intervention_rate': 0.0,
            }

        # Uncertainty scaling
        unc_z = (uncertainty - unc_mean) / (unc_std + 1e-8)
        unc_scale = 0.3 + 0.7 / (1.0 + np.exp(-unc_z))

        # Adaptive damping: higher score → stronger correction
        damping = min(1.0, score / (self.tau + 1e-8) * 0.5)

        corrected = base_forecast + damping * unc_scale * correction

        return corrected, {
            'intervened': True, 'score': score, 'threshold': self.tau,
            'damping': float(damping), 'unc_scale': float(unc_scale),
            'intervention_rate': 1.0,
            'mean_correction': float(np.mean(np.abs(damping * unc_scale * correction))),
        }
