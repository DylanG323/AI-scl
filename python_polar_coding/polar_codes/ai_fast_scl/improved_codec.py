"""Codec for the improved AI-SCL decoder."""

from typing import Union
from python_polar_coding.polar_codes.base.codec import BasePolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.improved_decoder import ImprovedAISCLDecoder
from python_polar_coding.polar_codes.ai_fast_scl.utils import load_model_from_file


class ImprovedAISCLPolarCodec(BasePolarCodec):
    """Polar codec using the improved AI-SCL decoder."""

    def __init__(self, N: int, K: int, L: int = 4,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 design_snr: float = 0.0,
                 enable_ai_pruning: bool = True,
                 llr_gate_threshold: float = 3.0,
                 enable_noise_adapt: bool = True,
                 enable_frozen_check: bool = True,
                 frozen_check_window: int = 3,
                 model_path: str = None,
                 ai_threshold: float = 0.05):

        self.L = L
        self._enable_ai_pruning = enable_ai_pruning
        self._llr_gate_threshold = llr_gate_threshold
        self._enable_noise_adapt = enable_noise_adapt
        self._enable_frozen_check = enable_frozen_check
        self._frozen_check_window = frozen_check_window
        self._model_path = model_path
        self._ai_threshold = ai_threshold

        super().__init__(N=N, K=K, design_snr=design_snr,
                         is_systematic=is_systematic, mask=mask)

    def init_decoder(self):
        model = None
        if self._model_path:
            model = load_model_from_file(self._model_path)

        return ImprovedAISCLDecoder(
            n=self.n, mask=self.mask,
            is_systematic=self.is_systematic, L=self.L,
            ai_model=model, ai_threshold=self._ai_threshold,
            enable_ai_pruning=self._enable_ai_pruning,
            llr_gate_threshold=self._llr_gate_threshold,
            enable_noise_adapt=self._enable_noise_adapt,
            enable_frozen_check=self._enable_frozen_check,
            frozen_check_window=self._frozen_check_window,
        )
