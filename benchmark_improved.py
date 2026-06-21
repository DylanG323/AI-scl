"""
Comprehensive benchmark comparing SCL, AI-FastSCL, and Improved variants.

Methods:
  0: Baseline SCL (SCListDecoder, L=4)
  1: AI-FastSCL (current: metric-based pruning only)
  2: Improved A-only (LLR-gated branching, τ=3.0)
  3: Improved C-only (noise-aware pruning)
  4: Improved A+C (LLR-gating + noise-aware)
  5: Improved A+C+D (LLR-gating + noise-aware + frozen check)

Each method runs at SNR ∈ {0.0, 1.0, 2.0, 3.0, 4.0, 5.0} dB,
200 messages per point.
"""

import time
import sys
import numpy as np
import argparse

sys.path.insert(0, '.')

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.improved_codec import ImprovedAISCLPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.ultra_fast_codec import UltraFastSCLPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message


def create_codecs(N=128, K=64, L=4):
    """Create all codec variants."""
    codecs = {}

    # 0: Baseline SCL
    codecs['SCL'] = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)

    # 1: AI-FastSCL (current)
    codecs['AI-FastSCL'] = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L, enable_ai_pruning=True)

    # 2: Improved — LLR-gating only (Method A)
    codecs['A:LLR-gate'] = ImprovedAISCLPolarCodec(
        N=N, K=K, L=L, enable_ai_pruning=False,
        llr_gate_threshold=3.0,
        enable_noise_adapt=False,
        enable_frozen_check=False)

    # 3: Improved — Noise-aware pruning only (Method C)
    codecs['C:NoiseAdapt'] = ImprovedAISCLPolarCodec(
        N=N, K=K, L=L, enable_ai_pruning=True,
        llr_gate_threshold=999.0,
        enable_noise_adapt=True,
        enable_frozen_check=False)

    # 4: Improved — LLR-gating + Noise-aware (A+C)
    codecs['A+C'] = ImprovedAISCLPolarCodec(
        N=N, K=K, L=L, enable_ai_pruning=True,
        llr_gate_threshold=3.0,
        enable_noise_adapt=True,
        enable_frozen_check=False)

    # 5: Improved — All methods (A+C+D)
    codecs['A+C+D'] = ImprovedAISCLPolarCodec(
        N=N, K=K, L=L, enable_ai_pruning=True,
        llr_gate_threshold=3.0,
        enable_noise_adapt=True,
        enable_frozen_check=True,
        frozen_check_window=3)

    # 6: UltraFast — A+C + no sleep + precomputed states + micro-opt
    codecs['UltraFast'] = UltraFastSCLPolarCodec(
        N=N, K=K, L=L, enable_ai_pruning=True,
        llr_gate_threshold=3.0,
        enable_noise_adapt=True)

    return codecs


def benchmark(codecs, bpsk, snr_db=2.0, num_messages=200):
    """Benchmark all codecs at given SNR."""
    results = {}

    for name, codec in codecs.items():
        bit_errors = 0
        frame_errors = 0
        latencies = []

        for _ in range(num_messages):
            msg = generate_binary_message(size=codec.K)
            encoded = codec.encode(msg)

            t0 = time.perf_counter()
            received = bpsk.transmit(message=encoded, snr_db=snr_db)
            decoded = codec.decode(received)
            elapsed = time.perf_counter() - t0

            latencies.append(elapsed * 1000)  # ms

            if not np.array_equal(msg, decoded):
                frame_errors += 1
                bit_errors += np.sum(msg != decoded)

        total_bits = num_messages * codec.K
        results[name] = {
            'ber': bit_errors / total_bits if total_bits > 0 else 0.0,
            'fer': frame_errors / num_messages,
            'latency_mean': np.mean(latencies),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'latency_std': np.std(latencies),
            'latency_min': np.min(latencies),
            'latency_max': np.max(latencies),
        }

        # Collect statistics from improved decoder
        decoder = codec.decoder
        if hasattr(decoder, 'get_statistics'):
            stats = decoder.get_statistics()
            results[name]['stats'] = stats
        if hasattr(decoder, 'reset_statistics'):
            decoder.reset_statistics()

    return results


def print_table(all_results, snr_list, baseline_name='SCL'):
    """Print formatted results table."""
    sep = '=' * 140
    header = (f"{'Method':<16} {'SNR':>6} {'BER':>10} {'FER':>8} "
              f"{'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} "
              f"{'P99(ms)':>10} {'Std(ms)':>10} {'Speedup':>10}")

    print(f"\n{sep}")
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print(sep)
    print(header)
    print('-' * 140)

    for snr in snr_list:
        base = all_results[snr].get(baseline_name, {})
        base_lat = base.get('latency_mean', 1.0)
        for name in all_results[0.0].keys():
            r = all_results[snr][name]
            speedup = base_lat / r['latency_mean'] if r['latency_mean'] > 0 else 0
            flag = '✅' if speedup >= 1.05 else ('⚠️' if speedup >= 0.95 else '❌')
            print(f"{name:<16} {snr:>5.1f} {r['ber']:>10.6f} {r['fer']:>8.4f} "
                  f"{r['latency_mean']:>10.3f} {r['latency_p50']:>10.3f} "
                  f"{r['latency_p95']:>10.3f} {r['latency_p99']:>10.3f} "
                  f"{r['latency_std']:>10.3f} {speedup:>9.2f}x {flag}")
        print('-' * 140)

    # Summary: average speedup across SNRs
    print(f"\n{'=' * 140}")
    print("SPEEDUP SUMMARY (vs SCL baseline)")
    print(f"{'=' * 140}")
    print(f"{'Method':<16} {'Avg':>8} {'Min':>8} {'Max':>8} {'≥1.05x':>8}")
    print('-' * 50)
    for name in all_results[0.0].keys():
        speedups = []
        for snr in snr_list:
            base_lat = all_results[snr][baseline_name]['latency_mean']
            method_lat = all_results[snr][name]['latency_mean']
            speedups.append(base_lat / method_lat if method_lat > 0 else 0)
        avg = np.mean(speedups)
        mn = np.min(speedups)
        mx = np.max(speedups)
        good = sum(1 for s in speedups if s >= 1.05)
        print(f"{name:<16} {avg:>7.2f}x {mn:>7.2f}x {mx:>7.2f}x {good:>5}/{len(snr_list)}")
    print(f"{'=' * 140}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument('--messages', type=int, default=200,
                       help='Number of messages per SNR point')
    parser.add_argument('--snr-min', type=float, default=0.0)
    parser.add_argument('--snr-max', type=float, default=5.0)
    parser.add_argument('--snr-step', type=float, default=1.0)
    args = parser.parse_args()

    N, K, L = args.N, args.K, args.L
    num_msgs = args.messages

    snr_list = list(np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step))

    print(f"Benchmark: N={N}, K={K}, L={L}, messages={num_msgs}/point")
    print(f"SNR sweep: {snr_list}")
    print()

    # Create codecs once
    codecs = create_codecs(N=N, K=K, L=L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

    print("Codecs created:")
    for name in codecs:
        print(f"  {name}")
    print()

    all_results = {}
    for snr in snr_list:
        print(f"Benchmarking SNR={snr:.1f} dB...", end=' ', flush=True)
        t0 = time.time()
        results = benchmark(codecs, bpsk, snr_db=snr, num_messages=num_msgs)
        all_results[snr] = results
        print(f"done ({time.time()-t0:.1f}s)")

    print_table(all_results, snr_list)

    # Print per-method statistics for the improved variants
    print(f"\n{'=' * 140}")
    print("DECODER STATISTICS (from improved variants at SNR=1.0 dB)")
    print(f"{'=' * 140}")
    for name in ['A:LLR-gate', 'C:NoiseAdapt', 'A+C', 'A+C+D', 'UltraFast']:
        if name in all_results.get(1.0, {}):
            stats = all_results[1.0][name].get('stats', {})
            if stats:
                print(f"\n{name}:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
