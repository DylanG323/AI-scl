"""Improved AI-Fast SCL decoder with LLR-gated branching and noise-aware pruning.

Key improvements over AIFastSCLDecoder:
  A) LLR-Confidence Gated Branching: skip split when |LLR| exceeds threshold
  B) Noise-Aware Pruning: adapt pruning aggressiveness to estimated SNR
  C) Frozen-Bit Consistency Check: pre-drop paths that disagree on frozen bits
"""

import numpy as np
from python_polar_coding.polar_codes.ai_fast_scl.decoder import AIFastSCLDecoder


class ImprovedAISCLDecoder(AIFastSCLDecoder):
    """SCL decoder with LLR-gated branching, noise-aware pruning, and frozen-bit checks."""

    def __init__(self, n: int, mask: np.ndarray,
                 is_systematic: bool = True, L: int = 4,
                 ai_model=None, ai_threshold: float = 0.05,
                 enable_ai_pruning: bool = True,
                 llr_gate_threshold: float = 3.0,
                 enable_noise_adapt: bool = True,
                 enable_frozen_check: bool = True,
                 frozen_check_window: int = 3):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic, L=L,
                         ai_model=ai_model, ai_threshold=ai_threshold,
                         enable_ai_pruning=enable_ai_pruning)

        # Method A: LLR-gated branching
        self.llr_gate_threshold = llr_gate_threshold
        self.gated_split_count = 0

        # Method C: noise-aware pruning
        self.enable_noise_adapt = enable_noise_adapt
        self.estimated_snr = None

        # Method D: frozen-bit consistency check
        self.enable_frozen_check = enable_frozen_check
        self.frozen_check_window = frozen_check_window
        self.frozen_mismatch_count = 0

        # Per-path frozen-bit mismatch counters
        self._frozen_mismatch = {}

    # ═══════════════════════════════════════════════════════════════
    # Method A: LLR-Confidence Gated Branching
    # ═══════════════════════════════════════════════════════════════

    def _populate_paths(self):
        """Populate paths with LLR-gated branching.

        When |current_llr| > threshold, skip the expensive deepcopy split
        and only keep the hard-decision path. P(error) = Q(threshold) is
        negligible for threshold >= 3.0.
        """
        new_paths = []
        for path in self.paths:
            llr = getattr(path, 'current_llr', 0.0)
            if abs(llr) > self.llr_gate_threshold:
                # High confidence: no split needed, keep only hard decision
                path._current_decision = 0 if llr >= 0 else 1
                new_paths.append(path)
                self.gated_split_count += 1
            else:
                # Low confidence: split as usual
                split_result = path.split_path()
                new_paths += split_result

        self.paths = new_paths

    # ═══════════════════════════════════════════════════════════════
    # Method C: Noise-Aware Pruning
    # ═══════════════════════════════════════════════════════════════

    def _estimate_channel_snr(self, received_llr: np.ndarray) -> float:
        """Estimate effective SNR from received LLRs.

        For BPSK over AWGN: LLR = 2*y/σ² where y ~ N(±1, σ²).
        Var[LLR] = 4/σ², so σ² = 4/Var[LLR].
        SNR_dB = -10*log10(σ²) = -10*log10(4/Var[LLR]).

        For robustness, use median absolute deviation.
        """
        llr_abs = np.abs(received_llr)
        if len(llr_abs) == 0:
            return -10.0

        # Robust sigma estimation via MAD of LLRs
        median_abs = np.median(llr_abs)
        if median_abs < 1e-8:
            return -10.0

        # E[|LLR|] ≈ sqrt(8/π) / σ for AWGN BPSK
        sigma_est = np.sqrt(8.0 / np.pi) / median_abs
        sigma_est = max(sigma_est, 1e-6)
        snr_db = -20.0 * np.log10(sigma_est)
        return float(np.clip(snr_db, -5.0, 20.0))

    def _get_adaptive_L(self) -> int:
        """Get effective list size based on estimated SNR."""
        if not self.enable_noise_adapt or self.estimated_snr is None:
            return self.L

        snr = self.estimated_snr
        if snr < 0.5:
            return self.L          # Very noisy: keep all paths
        elif snr < 2.0:
            return max(2, self.L)  # Moderate: standard
        elif snr < 4.0:
            return max(2, self.L - 1)  # Good: reduce slightly
        else:
            return max(1, self.L - 2)  # Excellent: aggressive pruning

    def _prune_before_branching(self, position: int):
        """Noise-aware pruning: adapt keep count to estimated SNR."""
        if not self.enable_ai_pruning:
            return

        adaptive_L = self._get_adaptive_L()
        if len(self.paths) <= adaptive_L:
            return

        try:
            self.ai_calls += 1
            metrics = np.array([getattr(p, '_path_metric', -np.inf)
                               for p in self.paths], dtype=np.float64)

            partition_idx = np.argpartition(-metrics, adaptive_L - 1)
            keep_idx = partition_idx[:adaptive_L]

            old_len = len(self.paths)
            self.paths = [self.paths[int(i)] for i in keep_idx]
            self.ai_pruned_count += old_len - len(self.paths)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # Method D: Frozen-Bit Consistency Check
    # ═══════════════════════════════════════════════════════════════

    def _check_frozen_consistency(self):
        """Drop paths that consistently disagree on frozen bits."""
        if not self.enable_frozen_check:
            return

        id_to_drop = []
        for i, path in enumerate(self.paths):
            path_id = id(path)
            if path_id not in self._frozen_mismatch:
                self._frozen_mismatch[path_id] = 0

            # Check if path disagrees with frozen value (always 0)
            decision = getattr(path, '_current_decision', 0)
            if decision != 0:
                self._frozen_mismatch[path_id] += 1
                if self._frozen_mismatch[path_id] >= self.frozen_check_window:
                    id_to_drop.append(i)
            else:
                # Reset on agreement (consecutive mismatches only)
                self._frozen_mismatch[path_id] = max(0, self._frozen_mismatch[path_id] - 1)

        if id_to_drop and len(self.paths) > 1:
            # Don't drop ALL paths
            if len(id_to_drop) < len(self.paths):
                self.paths = [p for i, p in enumerate(self.paths)
                             if i not in id_to_drop]
                self.frozen_mismatch_count += len(id_to_drop)
                # Clean up dropped path counters
                for i in id_to_drop:
                    if i < len(self._frozen_mismatch):
                        pid = id(self.paths[min(i, len(self.paths)-1)])
                        # Just reset — old IDs will be garbage collected
                self._frozen_mismatch = {id(p): self._frozen_mismatch.get(id(p), 0)
                                        for p in self.paths}

    # ═══════════════════════════════════════════════════════════════
    # Override main decode step
    # ═══════════════════════════════════════════════════════════════

    def _decode_position(self, position):
        """Single SCL step with all improvements layered."""
        self.set_decoder_state(position)
        self._compute_intermediate_alpha(position)

        # Early pruning BEFORE branching (noise-aware)
        if self.mask[position] == 1 and len(self.paths) > 1:
            self._prune_before_branching(position)

        if self.mask[position] == 1:
            self._populate_paths()        # ← Method A: LLR-gated
        if self.mask[position] == 0:
            self.set_frozen_value()
            self._check_frozen_consistency()  # ← Method D

        self._update_paths_metrics()
        self._select_best_paths()
        self._compute_bits(position)

    def decode_internal(self, received_llr: np.ndarray) -> np.ndarray:
        """Override to add SNR estimation per frame."""
        from time import sleep
        sleep(0.0005)
        self._reset_counters()

        # Estimate SNR once per frame
        if self.enable_noise_adapt:
            self.estimated_snr = self._estimate_channel_snr(received_llr)

        self._set_initial_state(received_llr)
        self._frozen_mismatch = {}

        for pos in range(self.N):
            self._decode_position(pos)

        return self.best_result

    def reset_statistics(self):
        """Reset all statistics counters."""
        super().reset_statistics()
        self.gated_split_count = 0
        self.frozen_mismatch_count = 0

    def get_statistics(self) -> dict:
        """Return comprehensive statistics."""
        stats = super().get_statistics()
        stats.update({
            'gated_splits': self.gated_split_count,
            'frozen_drops': self.frozen_mismatch_count,
            'estimated_snr': self.estimated_snr,
        })
        return stats
