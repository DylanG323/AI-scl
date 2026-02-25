"""
Debug test: Understand path counts and pruning at different SNRs.
"""
import time
import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.simulation.functions import (
    generate_binary_message,
)


def run_debug_test(N=128, K=64, messages=10, L=4):
    """Debug: understand what's happening at different SNRs."""
    
    snr_range = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    ai_model = PathPruningNN(input_dim=7, hidden_dim=32)
    try:
        ai_model.load_weights(f'trained_model_N{N}_K{K}.pt')
        ai_model.eval()
    except Exception as e:
        print(f"Model load failed: {e}")
        ai_model.eval()

    ai_codec = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L,
        ai_model=ai_model,
        ai_threshold=0.05,
        enable_ai_pruning=True
    )
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print(f'\nDebug: Path counts and pruning statistics')
    print(f'Parameters: N={N} K={K} L={L} messages={messages}')
    print('\nSNR(dB) | SCL paths | AI pruned | AI calls | time_scl(ms) | time_ai(ms) | Speedup')
    print('=' * 90)

    for snr in snr_range:
        scl_times = []
        ai_times = []
        
        # Create fresh decoders for each SNR to reset counters
        scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
        ai_codec = AIFastSCLPolarCodec(N=N, K=K, design_snr=0.0, L=L,
                                       ai_model=ai_model, ai_threshold=0.05, enable_ai_pruning=True)
        
        for i in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr)

            # SCL decode
            t0 = time.perf_counter()
            scl_codec.decode(received)
            t1 = time.perf_counter()

            # AI decode
            t2 = time.perf_counter()
            ai_codec.decode(received)
            t3 = time.perf_counter()

            scl_times.append((t1 - t0) * 1000.0)
            ai_times.append((t3 - t2) * 1000.0)
        
        avg_scl_time = np.mean(scl_times)
        avg_ai_time = np.mean(ai_times)
        speedup = avg_scl_time / avg_ai_time if avg_ai_time > 0 else 1.0
        
        # Get stats from AI decoder
        ai_decoder = ai_codec.decoder
        stats = ai_decoder.get_statistics()
        
        # Rough estimate of final path count (after all decoding)
        final_paths = len(ai_decoder.paths)
        
        print(f"{snr:5.1f}  | {final_paths:9d} | {stats['ai_pruned_count']:9d} | {stats['ai_calls']:8d} | {avg_scl_time:12.3f} | {avg_ai_time:11.3f} | {speedup:6.2f}x")

    print("=" * 90)
    print("Done!")


if __name__ == "__main__":
    run_debug_test(N=128, K=64, messages=10, L=4)
