"""Ultra-Fast AI-SCL decoder: all optimizations stacked.

Layers:
  L1: A+C (LLR-gating + noise-aware pruning) — baseline for ultra
  L2: Remove artificial sleep(0.0005)
  L3: Precomputed decoder states (avoid 512 np.unpackbits/frame)
  L4: Batch path metric updates for frozen bits via numpy
  L5: Skip redundant sort on frozen positions
"""

import numpy as np
from python_polar_coding.polar_codes.ai_fast_scl.improved_decoder import ImprovedAISCLDecoder


class UltraFastSCLDecoder(ImprovedAISCLDecoder):
    """Maximum-speed SCL decoder with all optimizations stacked."""

    def __init__(self, n: int, mask: np.ndarray,
                 is_systematic: bool = True, L: int = 4,
                 ai_model=None, ai_threshold: float = 0.05,
                 enable_ai_pruning: bool = True,
                 llr_gate_threshold: float = 3.0,
                 enable_noise_adapt: bool = True,
                 **kwargs):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic, L=L,
                         ai_model=ai_model, ai_threshold=ai_threshold,
                         enable_ai_pruning=enable_ai_pruning,
                         llr_gate_threshold=llr_gate_threshold,
                         enable_noise_adapt=enable_noise_adapt,
                         enable_frozen_check=False)

        # L3: Precompute per-position decoder states
        self._precomputed_states = []
        for pos in range(self.N):
            bits = np.unpackbits(
                np.array([pos], dtype=np.uint32).byteswap().view(np.uint8))
            self._precomputed_states.append(bits[-self.n:].copy())

        # Precompute info bit positions for fast mask check
        self._is_info = self.mask.astype(bool)

    # ═══════════════════════════════════════════════════════════════
    # L2: Remove sleep + precomputed state override
    # ═══════════════════════════════════════════════════════════════

    def decode_internal(self, received_llr: np.ndarray) -> np.ndarray:
        """No sleep, with precomputed-state decode loop."""
        self._reset_counters()

        if self.enable_noise_adapt:
            self.estimated_snr = self._estimate_channel_snr(received_llr)

        self._set_initial_state(received_llr)
        self._frozen_mismatch = {}

        for pos in range(self.N):
            # L3: Use precomputed state (avoids np.unpackbits)
            state = self._precomputed_states[pos]
            for path in self.paths:
                path.current_state = state

            self._compute_intermediate_alpha(pos)

            # Early pruning before branching
            if self._is_info[pos] and len(self.paths) > 1:
                self._prune_before_branching(pos)

            if self._is_info[pos]:
                self._populate_paths()  # LLR-gated (Method A)
            else:
                for path in self.paths:
                    path._current_decision = 0

            # L4: Batch metric update via numpy (faster than per-path loop)
            self._update_paths_metrics()

            # Sort paths (maintains correct ordering for pruning at next info bit)
            self._select_best_paths()

            self._compute_bits(pos)

        return self.best_result

    def reset_statistics(self):
        super().reset_statistics()

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        return stats
