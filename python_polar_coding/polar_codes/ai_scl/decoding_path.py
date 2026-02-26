import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            if llr_vec is None or bits_vec is None:
                return float(self._path_metric)

            # Preallocate bits padded array
            bits_pad = np.zeros(len(llr_vec), dtype=np.float32)
            bits_pad[:len(bits_vec)] = bits_vec

            X = np.concatenate([llr_vec.astype(np.float32), bits_pad])

            if hasattr(self.ai_model, 'score'):
                return float(self.ai_model.score(llr_vec, bits_vec))

            if hasattr(self.ai_model, 'score_batch'):
                scores = self.ai_model.score_batch(np.expand_dims(X, 0))
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                return float(np.asarray(scores).ravel()[0])

        except Exception:
            return float(self._path_metric)


class PathPruningNet(nn.Module):
    """Small MLP for AI-SCL path scoring (optimized)."""
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.fc1 = nn.Linear(2 * N, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)).squeeze(-1)
        return x

    def score_batch(self, X):
        """Score multiple paths at once. X: (batch, 2*N)"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        device = next(self.parameters()).device
        X = X.to(device)
        self.eval()
        with torch.no_grad():
            out = self.forward(X)
        return out.cpu().numpy()


# ---------------- Example of faster _ai_prune_paths usage -----------------
# Instead of calling score_ai() per path, collect features of all paths and
# call score_batch once. Do this inside AISCLDecoder._ai_prune_paths:
#
# X_list = []
# for path in self.paths:
#     llr = path.intermediate_llr[0] if path.intermediate_llr[0] is not None else np.zeros(N, dtype=np.float32)
#     bits = path.intermediate_bits[-1] if path.intermediate_bits[-1] is not None else np.zeros(0, dtype=np.float32)
#     bits_pad = np.zeros(N, dtype=np.float32)
#     bits_pad[:len(bits)] = bits
#     X_list.append(np.concatenate([llr, bits_pad]))
# X_batch = np.stack(X_list, axis=0)
# scores = self.ai_model.score_batch(X_batch)
# keep_idx = np.argpartition(-scores, keep_k-1)[:keep_k]
# self.paths = [self.paths[int(i)] for i in keep_idx]