import numpy as np
from python_polar_coding.polar_codes.base.decoding_path import DecodingPathMixin
from python_polar_coding.polar_codes.sc.decoder import SCDecoder


class AIPath(DecodingPathMixin, SCDecoder):
    """Path object used by AISCL decoders.

    Stores `ai_model` and exposes `score_ai()` that returns an AI-provided
    score when available, or falls back to the path metric.
    """
    def __init__(self, ai_model=None, **kwargs):
        super().__init__(**kwargs)
        self.ai_model = ai_model

    def score_ai(self):
        """Return score from ai_model when possible, else path metric."""
        if self.ai_model is None:
            return float(self._path_metric)
        try:
            llr_vec = self.intermediate_llr[0]
            bits_vec = self.intermediate_bits[-1]
            if llr_vec is not None and bits_vec is not None:
                # ai_model may implement `score` or `score_batch`; prefer `score` here
                if hasattr(self.ai_model, 'score'):
                    return float(self.ai_model.score(llr_vec, bits_vec))
                # fallback: try score_batch with single-row numpy
                if hasattr(self.ai_model, 'score_batch'):
                    X = np.concatenate([np.asarray(llr_vec, dtype=np.float32),
                                        np.pad(np.asarray(bits_vec, dtype=np.float32),
                                               (0, len(llr_vec) - len(bits_vec)), 'constant')])
                    scores = self.ai_model.score_batch(np.expand_dims(X, 0))
                    if hasattr(scores, 'cpu'):
                        scores = scores.cpu().numpy()
                    return float(np.asarray(scores).ravel()[0])
        except Exception:
            pass
        return float(self._path_metric)
