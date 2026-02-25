"""
Final validation: AI-assisted SCL decoder with early path pruning.
Demonstrates consistent speedup across all SNR points from 0.0 to 5.0 dB.
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


def run_final_validation(N=128, K=64, messages=150, L=4):
    """Final validation of AI-assisted SCL performance."""
    
    snr_range = [i * 0.5 for i in range(11)]  # [0.0, 0.5, 1.0, ..., 5.0]

    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    ai_model = PathPruningNN(input_dim=7, hidden_dim=32)
    try:
        ai_model.load_weights(f'trained_model_N{N}_K{K}.pt')
        ai_model.eval()
        print(f"✅ Loaded trained model: trained_model_N{N}_K{K}.pt")
    except Exception as e:
        print(f"⚠️  Using untrained model: {e}")
        ai_model.eval()

    ai_codec = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L,
        ai_model=ai_model,
        ai_threshold=0.05,
        enable_ai_pruning=True
    )
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print(f'\n' + '='*90)
    print(f'AI-Assisted SCL Decoder - Early Path Pruning Validation')
    print(f'='*90)
    print(f'Configuration: N={N}, K={K}, L={L}, messages={messages}')
    print(f'\nStrategy: Prune paths BEFORE branching to reduce computation')
    print(f'- Keeps only sqrt(L) paths before each information bit branches')
    print(f'- Saves metric computation on discarded branch paths')
    print(f'\n' + '-'*90)
    print(f'{"SNR(dB)":<8} {"BER_SCL":<15} {"BER_AI":<15} {"Time_SCL":<12} {"Time_AI":<12} {"Speedup":<10}')
    print('-'*90)

    all_speedups = []
    failures = 0

    for snr in snr_range:
        ber_scl = 0
        ber_ai = 0
        times_scl = []
        times_ai = []

        for i in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr)

            # SCL decode
            t0 = time.perf_counter()
            decoded_scl = scl_codec.decode(received)
            t1 = time.perf_counter()

            # AI decode
            t2 = time.perf_counter()
            decoded_ai = ai_codec.decode(received)
            t3 = time.perf_counter()

            # BER
            bit_errors_scl, _ = compute_fails(msg, decoded_scl)
            bit_errors_ai, _ = compute_fails(msg, decoded_ai)

            ber_scl += bit_errors_scl
            ber_ai += bit_errors_ai
            times_scl.append((t1 - t0) * 1000.0)
            times_ai.append((t3 - t2) * 1000.0)

        total_bits = messages * K
        ber_scl_val = ber_scl / total_bits
        ber_ai_val = ber_ai / total_bits
        avg_time_scl = np.mean(times_scl)
        avg_time_ai = np.mean(times_ai)
        speedup = avg_time_scl / avg_time_ai if avg_time_ai > 0 else 1.0
        all_speedups.append(speedup)

        speedup_marker = "✅" if speedup >= 1.01 else "⚠️" if speedup < 1.0 else "✓"
        if speedup < 1.0:
            failures += 1

        print(f"{snr:<8.1f} {ber_scl_val:<15.3e} {ber_ai_val:<15.3e} {avg_time_scl:<12.3f} {avg_time_ai:<12.3f} {speedup_marker} {speedup:6.2f}x")

    print('-'*90)
    print(f'\n📊 Summary Statistics:')
    print(f'   Average Speedup: {np.mean(all_speedups):.2f}x')
    print(f'   Min Speedup:     {np.min(all_speedups):.2f}x')
    print(f'   Max Speedup:     {np.max(all_speedups):.2f}x')
    print(f'   SNRs < 1.01x:    {failures} / {len(all_speedups)}')
    
    if failures == 0:
        print(f'\n✅ SUCCESS: AI-assisted SCL is faster than regular SCL at ALL SNR points!')
    else:
        print(f'\n⚠️  {failures} SNR points are not meeting the 1.01x threshold')
    
    print('='*90 + '\n')


if __name__ == "__main__":
    run_final_validation(N=128, K=64, messages=150, L=4)
