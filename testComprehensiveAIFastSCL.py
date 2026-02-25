"""
Comprehensive test: Compare SC List (SCL) and AI-Fast-SCL decoding across full SNR range.
Test SNRs from 0.0 to 5.0 dB in 0.5 dB increments.
"""
import time
import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)


def run_comprehensive_test(N=128, K=64, messages=300, L=4):
    """Run comprehensive comparison test with full SNR range."""
    
    # Full SNR range: 0.0 to 5.0 in 0.5 dB increments
    snr_range = [i * 0.5 for i in range(11)]  # [0.0, 0.5, 1.0, ..., 5.0]

    # Create codecs
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    # Load or create AI model for AIFastSCL
    ai_model = PathPruningNN(input_dim=7, hidden_dim=32)
    try:
        ai_model.load_weights(f'trained_model_N{N}_K{K}.pt')
        ai_model.eval()
        print(f"[Info] Loaded trained model: trained_model_N{N}_K{K}.pt")
    except Exception as e:
        print(f"[Info] Trained model not found, using untrained PathPruningNN: {e}")
        ai_model.eval()

    # Create AIFastSCL with trained model
    ai_fastsscl_codec = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L,
        ai_model=ai_model,
        ai_threshold=0.05,
        enable_ai_pruning=True
    )
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print(f'\nComprehensive Test: SCL vs AI-Fast-SCL')
    print(f'Parameters: N={N} K={K} L={L} messages={messages}')
    print('\nSNR(dB) | BER_scl | BER_aifscl | time_scl(ms) | time_aifscl(ms) | Speedup')
    print('=' * 80)

    all_speedups = []

    for snr in snr_range:
        ber_scl = 0
        ber_aifastscl = 0

        times_scl = []
        times_aifastscl = []

        for i in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr)

            # SCL decode
            t0 = time.perf_counter()
            decoded_scl = scl_codec.decode(received)
            t1 = time.perf_counter()

            # AIFastSCL decode (with AI enabled)
            t2 = time.perf_counter()
            decoded_aifastscl = ai_fastsscl_codec.decode(received)
            t3 = time.perf_counter()

            # Measure BER
            bit_errors_scl, _ = compute_fails(msg, decoded_scl)
            bit_errors_aifastscl, _ = compute_fails(msg, decoded_aifastscl)

            ber_scl += bit_errors_scl
            ber_aifastscl += bit_errors_aifastscl

            times_scl.append((t1 - t0) * 1000.0)
            times_aifastscl.append((t3 - t2) * 1000.0)

            if (i + 1) % 20 == 0:
                print(f"  SNR {snr}: {i+1}/{messages} processed...", flush=True)

        # Compute averages
        total_bits = messages * K
        avg_time_scl = np.mean(times_scl)
        avg_time_aifastscl = np.mean(times_aifastscl)

        ber_scl_val = ber_scl / total_bits
        ber_aifastscl_val = ber_aifastscl / total_bits
        
        # Compute speedup
        speedup = avg_time_scl / avg_time_aifastscl if avg_time_aifastscl > 0 else 1.0
        all_speedups.append(speedup)

        speedup_str = f"{speedup:6.2f}x"
        if speedup < 1.0:
            speedup_str = f"❌ {speedup_str}"  # Mark failures
        else:
            speedup_str = f"✅ {speedup_str}"

        print(f"{snr:5.1f}  | {ber_scl_val:.6e} | {ber_aifastscl_val:.6e} | {avg_time_scl:8.3f} | {avg_time_aifastscl:8.3f} | {speedup_str}")

    print('=' * 80)
    print(f"\nAverage Speedup: {np.mean(all_speedups):.2f}x")
    print(f"Min Speedup: {np.min(all_speedups):.2f}x")
    print(f"Max Speedup: {np.max(all_speedups):.2f}x")
    print(f"Speedups < 1.0x: {sum(1 for s in all_speedups if s < 1.0)} / {len(all_speedups)}")
    print("\nDone!")


if __name__ == "__main__":
    run_comprehensive_test(N=128, K=64, messages=300, L=4)
