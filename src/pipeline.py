"""
Main adaptive self-retrieval pipeline.
"""

import numpy as np
from typing import Dict, Optional
from embeddings import ChronosEmbeddingExtractor
from knowledge_base import SelfRetrievalKB, KBEntry
from fusion import ForecastEnsembleFusion, ShapeBlendFusion, HybridFusion


class AdaptiveSelfRetrievalPipeline:
    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-small",
        device: str = "cpu",
        window_size: int = 512,
        k: int = 5,
        z_threshold: float = 1.0,
        use_error_weighting: bool = True,
        fusion_strategy: str = "ensemble",
        extractor: Optional[ChronosEmbeddingExtractor] = None,
        ensemble_k: int = 3,
        base_weight: float = 2.0,
        shape_alpha: float = 0.5,
    ):
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = ChronosEmbeddingExtractor(model_name, device)
        self.window_size = window_size
        self.k = k
        self.z_threshold = z_threshold
        self.use_error_weighting = use_error_weighting
        self.fusion_strategy = fusion_strategy

        self.embedding_dim = self.extractor.embedding_dim
        self.kb = SelfRetrievalKB(self.embedding_dim)

        if fusion_strategy == "ensemble":
            self.fusion = ForecastEnsembleFusion(self.extractor)
        elif fusion_strategy == "shape_blend":
            self.fusion = ShapeBlendFusion(self.extractor, alpha=shape_alpha)
        elif fusion_strategy == "hybrid":
            self.fusion = HybridFusion(self.extractor, ensemble_k=ensemble_k,
                                        base_weight=base_weight, shape_alpha=shape_alpha)
        else:
            self.fusion = ForecastEnsembleFusion(self.extractor)

        self.ensemble_k = ensemble_k
        self.base_weight = base_weight

        self.stats = {
            'total_predictions': 0,
            'retrievals_triggered': 0,
            'errors': [],
            'uncertainties': [],
        }

    def build_initial_kb(
        self, train_data: np.ndarray, prediction_length: int = 96,
        stride: int = None, precomputed_kb: list = None,
    ):
        if precomputed_kb is not None:
            for entry in precomputed_kb:
                self.kb.add(entry)
            return

        if stride is None:
            stride = self.window_size

        count = 0
        for start in range(0, len(train_data) - self.window_size - prediction_length, stride):
            window = train_data[start:start + self.window_size]
            future = train_data[start + self.window_size:start + self.window_size + prediction_length]
            if len(future) < prediction_length:
                continue

            embedding = self.extractor.extract_embedding(window)
            _, _, uncertainty = self.extractor.predict_with_uncertainty(window, len(future))

            entry = KBEntry(
                window=window, future=future, embedding=embedding,
                timestamp=start, uncertainty=uncertainty,
            )
            self.kb.add(entry)
            count += 1

    def predict(
        self,
        context: np.ndarray,
        prediction_length: int,
    ) -> Dict:
        self.stats['total_predictions'] += 1

        median, quantiles, uncertainty = self.extractor.predict_with_uncertainty(
            context, prediction_length
        )
        self.stats['uncertainties'].append(uncertainty)

        result = {
            'forecast': median,
            'quantiles': quantiles,
            'uncertainty': uncertainty,
            'retrieved': False,
            'num_retrieved': 0,
        }

        should_retrieve = self.kb.should_retrieve(uncertainty, self.z_threshold)

        if should_retrieve and self.kb.size > 0:
            self.stats['retrievals_triggered'] += 1
            result['retrieved'] = True

            embedding = self.extractor.extract_embedding(context)
            retrieved = self.kb.retrieve(
                embedding, k=self.k,
                use_error_weighting=self.use_error_weighting,
            )

            if retrieved:
                result['forecast'] = self.fusion.fuse(
                    median, uncertainty, retrieved, prediction_length,
                    current_context=context,
                )
                result['num_retrieved'] = len(retrieved)

        return result

    def update(self, context, ground_truth, prediction):
        error = float(np.mean((prediction - ground_truth) ** 2))
        embedding = self.extractor.extract_embedding(context)
        _, _, uncertainty = self.extractor.predict_with_uncertainty(context, len(ground_truth))
        entry = KBEntry(
            window=context.copy(), future=ground_truth.copy(),
            embedding=embedding, timestamp=self.stats['total_predictions'],
            uncertainty=uncertainty, error=error,
        )
        self.kb.add(entry)
        self.stats['errors'].append(error)

    def get_retrieval_efficiency(self):
        if self.stats['total_predictions'] == 0:
            return 0.0
        return self.stats['retrievals_triggered'] / self.stats['total_predictions']

    def reset_stats(self):
        self.stats = {'total_predictions': 0, 'retrievals_triggered': 0, 'errors': [], 'uncertainties': []}

    def reset_kb(self):
        self.kb = SelfRetrievalKB(self.embedding_dim)
        self.reset_stats()
