import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoding_path import AIPath
from python_polar_coding.polar_codes.sc_list.decoder import SCListDecoder


class _PathPruningNet(nn.Module):
    """Small MLP used by AISCLDecoder as a default scorer.

    Input: concatenation of LLR vector and padded bits (length 2*N).
    Output: scalar score per path in [0,1].
    """
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
        """Accepts numpy array (batch, 2*N) or torch tensor and returns numpy scores."""
        was_numpy = isinstance(X, np.ndarray)
        if was_numpy:
            X = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            out = self.forward(X)
        return out.cpu().numpy()


class AISCLDecoder(SCListDecoder):
    """AI-guided SCL decoder with optional NN-based path pruning."""
    path_class = AIPath

    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 L: int = 1,
                 ai_model=None, deterministic: bool = True):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic, L=L)
        self.L = L
        # AI model can be provided; if None, create a lightweight NN scorer
        if ai_model is None:
            try:
                self.ai_model = _PathPruningNet(self.N)
            except Exception:
                self.ai_model = None
        else:
            self.ai_model = ai_model
        if isinstance(self.ai_model, nn.Module):
            self.ai_model.eval()

        # Deterministic mode (seed RNGs) to improve repeatability
        self.deterministic = bool(deterministic)
        if self.deterministic:
            np.random.seed(0)
            try:
                torch.manual_seed(0)
            except Exception:
                pass

        self.paths = [self.path_class(n=n, mask=mask, is_systematic=is_systematic, ai_model=self.ai_model)]

    @property
    def result(self):
        return [path.result for path in self.paths]

    @property
    def best_result(self):
        return self.result[0]

    def decode_internal(self, received_llr: np.array) -> np.array:
        self._reset_counters()
        self._set_initial_state(received_llr)
        for pos in range(self.N):
            self._decode_position(pos)
        return self.best_result

    def _set_initial_state(self, received_llr):
        for path in self.paths:
            path._set_initial_state(received_llr)

    def _decode_position(self, position):
        self.set_decoder_state(position)
        self._compute_intermediate_alpha(position)
        if self.mask[position] == 1:
            self._populate_paths()
            if position % 2 == 0 and len(self.paths) > 2 * self.L:
                k = max(self.L + 1, int(1.25 * self.L)) if self.L > 1 else 2
                self._prune_to_k_paths(k)
        if self.mask[position] == 0:
            self.set_frozen_value()
        self._update_paths_metrics()
        self._select_best_paths()
        self._compute_bits(position)

    def set_decoder_state(self, position):
        for path in self.paths:
            path._set_decoder_state(position)

    def _compute_intermediate_alpha(self, position):
        for path in self.paths:
            path._compute_intermediate_alpha(position)

    def set_frozen_value(self):
        for path in self.paths:
            path._current_decision = 0

    def _populate_paths(self):
        new_paths = list()
        for path in self.paths:
            split_result = path.split_path()
            new_paths += split_result
        self.paths = new_paths
    
    def _prune_by_metric(self):
        """Fast O(n) metric-based pruning to reduce memory and computation."""
        target = min(self.L, len(self.paths))
        if len(self.paths) <= target:
            return
        try:
            metrics = np.array([path._path_metric for path in self.paths], dtype=np.float64)
            partition_idx = np.argpartition(-metrics, target - 1)
            keep_idx = partition_idx[:target]
            self.paths = [self.paths[int(i)] for i in keep_idx]
        except:
            pass

    def _ai_prune_paths(self, keep_k: int):
        """Use batch NN scoring to prune to top `keep_k` paths.

        Builds a lightweight feature matrix from each path (LLR vector
        + padded bits) and calls `ai_model.score_batch` to score them.
        """
        if self.ai_model is None or len(self.paths) <= keep_k:
            return
        try:
            X_list = []
            for path in self.paths:
                try:
                    llr = path.intermediate_llr[0]
                    bits = path.intermediate_bits[-1]
                    if llr is None:
                        llr = np.zeros(self.N, dtype=np.float32)
                    if bits is None:
                        bits = np.zeros(0, dtype=np.float32)
                    llr = np.asarray(llr, dtype=np.float32)
                    bits = np.asarray(bits, dtype=np.float32)
                    N = len(llr)
                    bits_pad = np.zeros(N, dtype=np.float32)
                    bits_pad[:len(bits)] = bits
                    X_list.append(np.concatenate([llr, bits_pad]))
                except Exception:
                    X_list.append(np.zeros(2 * self.N, dtype=np.float32))

            X_batch = np.stack(X_list, axis=0)
            # ai_model may accept numpy or torch; many implementations provide
            # score_batch that accepts numpy, so try that first
            if hasattr(self.ai_model, 'score_batch'):
                try:
                    scores = self.ai_model.score_batch(X_batch)
                except Exception:
                    # try calling via torch if it's a nn.Module with score_batch
                    try:
                        self.ai_model.eval()
                        with torch.no_grad():
                            tX = torch.from_numpy(X_batch.astype(np.float32))
                            scores = self.ai_model(tX).cpu().numpy()
                    except Exception:
                        scores = None
            else:
                # fallback: call score() per path
                scores = np.array([self.ai_model.score(x[:self.N], x[self.N:]) for x in X_batch], dtype=np.float32)
            if scores is None:
                scores = np.array([p._path_metric for p in self.paths], dtype=np.float64)
            scores = np.asarray(scores, dtype=np.float64)
            # Keep top keep_k by ai score
            keep_idx = np.argpartition(-scores, keep_k - 1)[:keep_k]
            self.paths = [self.paths[int(i)] for i in keep_idx]
        except Exception:
            # fallback: keep top by existing metric
            if len(self.paths) > keep_k:
                metrics = np.array([p._path_metric for p in self.paths], dtype=np.float64)
                idx = np.argpartition(-metrics, keep_k - 1)[:keep_k]
                self.paths = [self.paths[int(i)] for i in idx]

    def _select_best_paths(self):
        """Select best L paths. Use AI pruning as a fast pre-filter when available."""
        if len(self.paths) <= self.L:
            return

        # If we have a neural scorer, do a cheap AI prune to keep 1.5*L candidates,
        # then finalize by metric to exactly L. This minimizes expensive metric
        # updates and sorting while protecting BER.
        if self.ai_model is not None and len(self.paths) > 2 * self.L:
            # Use an aggressive-but-safe candidate window (1.25*L)
            candidates = max(self.L + 1, int(1.25 * self.L))
            self._ai_prune_paths(candidates)

        # Final selection by path metric (deterministic)
        if len(self.paths) <= self.L:
            return
        metrics = np.array([p._path_metric for p in self.paths], dtype=np.float64)
        idx = np.argpartition(-metrics, self.L - 1)[:self.L]
        self.paths = [self.paths[int(i)] for i in idx]
    
    def _prune_to_k_paths(self, k):
        """Fast O(n) pruning to keep only top k paths."""
        if len(self.paths) <= k:
            return
        try:
            metrics = np.array([path._path_metric for path in self.paths], dtype=np.float64)
            partition_idx = np.argpartition(-metrics, k - 1)
            keep_idx = partition_idx[:k]
            self.paths = [self.paths[int(i)] for i in keep_idx]
        except:
            self.paths = self.paths[:k]
    
    def _update_paths_metrics(self):
        for path in self.paths:
            path.update_path_metric()

    

    def _reset_counters(self):
        pass

    def _finalize_counters(self):
        pass

    def _compute_bits(self, position):
        for path in self.paths:
            path._compute_intermediate_beta(position)
            path._update_decoder_state()
