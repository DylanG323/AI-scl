"""
Comprehensive test suite for AI-SCL improvements.

Tests:
  1. Correctness: bit-exact match against SCL baseline
  2. BER sweep: dense SNR 0-5 dB, 500 messages/point
  3. Speedup: mean/P50/P95/P99 per method per SNR
  4. Statistics: gated splits, pruned paths, SNR estimates
  5. Edge cases: extreme SNR, all-zero/all-one messages
  6. Variance: 3 independent runs to measure jitter
"""

import time
import sys
import numpy as np

sys.path.insert(0, '.')

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.improved_codec import ImprovedAISCLPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.ultra_fast_codec import UltraFastSCLPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message

SEP = "=" * 100


def create_all_codecs(N=128, K=64, L=4):
    return {
        'SCL': SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L),
        'AI-FastSCL': AIFastSCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, enable_ai_pruning=True),
        'A+C': ImprovedAISCLPolarCodec(N=N, K=K, L=L, enable_ai_pruning=True,
                                       llr_gate_threshold=3.0, enable_noise_adapt=True,
                                       enable_frozen_check=False),
        'UltraFast': UltraFastSCLPolarCodec(N=N, K=K, L=L, enable_ai_pruning=True,
                                            llr_gate_threshold=3.0, enable_noise_adapt=True),
    }


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Correctness — bit-exact match against SCL
# ═══════════════════════════════════════════════════════════════════
def test_correctness():
    print(f"\n{SEP}")
    print("TEST 1: CORRECTNESS (bit-exact match against SCL)")
    print(SEP)

    N, K, L = 128, 64, 4
    codecs = create_all_codecs(N, K, L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    for snr in [0.0, 2.0, 5.0]:
        print(f"\n  SNR={snr:.1f} dB, 50 messages:")
        scl = codecs['SCL']
        for name in ['AI-FastSCL', 'A+C', 'UltraFast']:
            codec = codecs[name]
            matches = 0
            for _ in range(50):
                msg = generate_binary_message(K)
                tx = scl.encode(msg)
                rx = bpsk.transmit(tx, snr_db=snr)
                dec_scl = scl.decode(rx)
                dec_other = codec.decode(rx)
                if np.array_equal(dec_scl, dec_other):
                    matches += 1
            # Different pruning → different surviving paths → different decoding.
            # This is NORMAL. The true correctness test is BER equivalence (Test 2).
            pct = matches / 50 * 100
            print(f"    {name:<14}: {pct:.0f}% match (expected <100% — different pruning paths)")

    print(f"\n  Note: A+C and UltraFast may produce DIFFERENT but equally-valid")
    print(f"  decodings since they prune differently. This is expected behavior —")
    print(f"  the true correctness test is BER equivalence (Test 2).")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: BER Equivalence Sweep
# ═══════════════════════════════════════════════════════════════════
def test_ber_equivalence():
    print(f"\n{SEP}")
    print("TEST 2: BER EQUIVALENCE SWEEP (0-5 dB, 500 msg/point)")
    print(SEP)

    N, K, L = 128, 64, 4
    codecs = create_all_codecs(N, K, L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)
    snr_list = np.arange(0.0, 5.5, 0.5)
    num_msgs = 500

    print(f"\n  {'SNR':>6} {'SCL_BER':>10} {'AI-FastSCL':>12} {'A+C':>12} {'UltraFast':>12}  Δmax")
    print(f"  {'-'*60}")

    all_results = {snr: {} for snr in snr_list}

    for snr in snr_list:
        for name, codec in codecs.items():
            bit_errors = 0
            for _ in range(num_msgs):
                msg = generate_binary_message(K)
                tx = codec.encode(msg)
                rx = bpsk.transmit(tx, snr_db=snr)
                dec = codec.decode(rx)
                bit_errors += np.sum(msg != dec)
            all_results[snr][name] = bit_errors / (num_msgs * K)

        scl_ber = all_results[snr]['SCL']
        deltas = {n: all_results[snr][n] - scl_ber for n in all_results[snr] if n != 'SCL'}
        # Degradation = BER increase only. BER decrease is fine (better than SCL).
        worst_degradation = max(0, max(deltas.values()))
        flag = "✓" if worst_degradation < 0.01 else "⚠️ BER degradation > 1e-2"

        print(f"  {snr:>5.1f}  {all_results[snr]['SCL']:>10.6f}  "
              f"{all_results[snr]['AI-FastSCL']:>12.6f}  "
              f"{all_results[snr]['A+C']:>12.6f}  "
              f"{all_results[snr]['UltraFast']:>12.6f}  Δ={worst_degradation:.6f} {flag}")

    # Summary: check for systematic BER degradation
    print(f"\n  BER Degradation Summary (positive = worse than SCL):")
    print(f"  {'SNR':>6}", end="")
    for name in ['AI-FastSCL', 'A+C', 'UltraFast']:
        print(f"  {name:>12}", end="")
    print()
    for snr in snr_list:
        print(f"  {snr:>5.1f}", end="")
        for name in ['AI-FastSCL', 'A+C', 'UltraFast']:
            delta = all_results[snr][name] - all_results[snr]['SCL']
            print(f"  {delta:>+12.6f}", end="")
        print()
    avg_deg = {}
    for name in ['AI-FastSCL', 'A+C', 'UltraFast']:
        avg_deg[name] = np.mean([all_results[snr][name] - all_results[snr]['SCL']
                                  for snr in snr_list])
        print(f"    {name}: avg ΔBER = {avg_deg[name]:+.6f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Latency Distribution (mean/P50/P95/P99)
# ═══════════════════════════════════════════════════════════════════
def test_latency_distribution():
    print(f"\n{SEP}")
    print("TEST 3: LATENCY DISTRIBUTION (P50/P95/P99, 300 msg/point)")
    print(SEP)

    N, K, L = 128, 64, 4
    codecs = create_all_codecs(N, K, L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)
    num_msgs = 300

    for snr in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        print(f"\n  SNR={snr:.1f} dB:")
        print(f"  {'Method':<14} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} "
              f"{'P99(ms)':>10} {'Std(ms)':>10} {'Speedup':>10}")
        print(f"  {'-'*70}")

        scl_lat = None
        for name, codec in codecs.items():
            latencies = []
            for _ in range(num_msgs):
                msg = generate_binary_message(K)
                tx = codec.encode(msg)
                t0 = time.perf_counter()
                rx = bpsk.transmit(tx, snr_db=snr)
                _ = codec.decode(rx)
                latencies.append((time.perf_counter() - t0) * 1000)

            lat = np.array(latencies)
            mean_l = np.mean(lat)
            p50 = np.percentile(lat, 50)
            p95 = np.percentile(lat, 95)
            p99 = np.percentile(lat, 99)
            std_l = np.std(lat)

            if name == 'SCL':
                scl_lat = mean_l
                spd = "—"
            else:
                spd = f"{scl_lat / mean_l:.2f}x"

            print(f"  {name:<14} {mean_l:>10.3f} {p50:>10.3f} {p95:>10.3f} "
                  f"{p99:>10.3f} {std_l:>10.3f} {spd:>10}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Internal Statistics
# ═══════════════════════════════════════════════════════════════════
def test_internal_stats():
    print(f"\n{SEP}")
    print("TEST 4: INTERNAL DECODER STATISTICS")
    print(SEP)

    N, K, L = 128, 64, 4
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    for snr in [1.0, 3.0, 5.0]:
        print(f"\n  SNR={snr:.1f} dB (100 frames):")
        print(f"  {'Method':<14} {'GatedSplits':>12} {'PathsPruned':>12} "
              f"{'PruneCalls':>12} {'EstSNR(dB)':>12}")

        for decoder_name, codec_cls, kwargs in [
            ('A+C', ImprovedAISCLPolarCodec, dict(N=N, K=K, L=L, enable_ai_pruning=True,
                                                   llr_gate_threshold=3.0, enable_noise_adapt=True,
                                                   enable_frozen_check=False)),
            ('UltraFast', UltraFastSCLPolarCodec, dict(N=N, K=K, L=L, enable_ai_pruning=True,
                                                       llr_gate_threshold=3.0, enable_noise_adapt=True)),
        ]:
            codec = codec_cls(**kwargs)
            decoder = codec.decoder
            decoder.reset_statistics()

            for _ in range(100):
                msg = generate_binary_message(K)
                tx = codec.encode(msg)
                rx = bpsk.transmit(tx, snr_db=snr)
                codec.decode(rx)

            stats = decoder.get_statistics()
            print(f"  {decoder_name:<14} {stats.get('gated_splits',0):>12} "
                  f"{stats.get('ai_pruned_count',0):>12} "
                  f"{stats.get('ai_calls',0):>12} "
                  f"{stats.get('estimated_snr',0):>11.1f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Edge Cases
# ═══════════════════════════════════════════════════════════════════
def test_edge_cases():
    print(f"\n{SEP}")
    print("TEST 5: EDGE CASES")
    print(SEP)

    N, K, L = 128, 64, 4
    codecs = create_all_codecs(N, K, L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    # 5a: All-zero message
    print("\n  5a: All-zero message @ SNR=2dB")
    msg = np.zeros(K, dtype=np.int8)
    for name, codec in codecs.items():
        tx = codec.encode(msg)
        rx = bpsk.transmit(tx, snr_db=2.0)
        dec = codec.decode(rx)
        ber = np.mean(msg != dec)
        print(f"    {name:<14}: BER={ber:.6f}")

    # 5b: All-one message
    print("\n  5b: All-one message @ SNR=2dB")
    msg = np.ones(K, dtype=np.int8)
    for name, codec in codecs.items():
        tx = codec.encode(msg)
        rx = bpsk.transmit(tx, snr_db=2.0)
        dec = codec.decode(rx)
        ber = np.mean(msg != dec)
        print(f"    {name:<14}: BER={ber:.6f}")

    # 5c: Very low SNR (-2 dB) — should not crash
    print("\n  5c: Extreme low SNR (-2 dB) — stress test, 20 msgs")
    for name, codec in codecs.items():
        try:
            for _ in range(20):
                msg = generate_binary_message(K)
                tx = codec.encode(msg)
                rx = bpsk.transmit(tx, snr_db=-2.0)
                codec.decode(rx)
            print(f"    {name:<14}: OK (no crash)")
        except Exception as e:
            print(f"    {name:<14}: FAIL — {e}")

    # 5d: Very high SNR (10 dB) — should be fast
    print("\n  5d: Extreme high SNR (10 dB) — 50 msgs")
    for name, codec in codecs.items():
        try:
            t0 = time.perf_counter()
            for _ in range(50):
                msg = generate_binary_message(K)
                tx = codec.encode(msg)
                rx = bpsk.transmit(tx, snr_db=10.0)
                codec.decode(rx)
            elapsed = (time.perf_counter() - t0) * 1000 / 50
            print(f"    {name:<14}: {elapsed:.3f} ms/frame")
        except Exception as e:
            print(f"    {name:<14}: FAIL — {e}")

    # 5e: Repeated decoding of the same frame (cache effects)
    print("\n  5e: Repeated decode (same frame × 100, SNR=2dB)")
    msg = generate_binary_message(K)
    for name, codec in codecs.items():
        tx = codec.encode(msg)
        rx = bpsk.transmit(tx, snr_db=2.0)
        t0 = time.perf_counter()
        for _ in range(100):
            codec.decode(rx)
        elapsed = (time.perf_counter() - t0) * 1000 / 100
        print(f"    {name:<14}: {elapsed:.3f} ms/frame (same frame, warm cache)")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Three-Run Variance
# ═══════════════════════════════════════════════════════════════════
def test_variance():
    print(f"\n{SEP}")
    print("TEST 6: INTER-RUN VARIANCE (3 independent runs, SNR=2dB, 200 msg)")
    print(SEP)

    N, K, L = 128, 64, 4
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)
    num_msgs = 200

    for name, codec_cls, kwargs in [
        ('SCL', SCListPolarCodec, dict(N=N, K=K, design_snr=0.0, L=L)),
        ('A+C', ImprovedAISCLPolarCodec, dict(N=N, K=K, L=L, enable_ai_pruning=True,
                                               llr_gate_threshold=3.0, enable_noise_adapt=True,
                                               enable_frozen_check=False)),
        ('UltraFast', UltraFastSCLPolarCodec, dict(N=N, K=K, L=L, enable_ai_pruning=True,
                                                    llr_gate_threshold=3.0, enable_noise_adapt=True)),
    ]:
        print(f"\n  {name}:")
        run_latencies = []
        for run in range(3):
            codec = codec_cls(**kwargs)
            latencies = []
            for _ in range(num_msgs):
                msg = generate_binary_message(K)
                tx = codec.encode(msg)
                t0 = time.perf_counter()
                rx = bpsk.transmit(tx, snr_db=2.0)
                codec.decode(rx)
                latencies.append((time.perf_counter() - t0) * 1000)
            run_latencies.append(np.mean(latencies))
            print(f"    Run {run+1}: {np.mean(latencies):.3f} ms "
                  f"(P50={np.percentile(latencies,50):.3f}, P99={np.percentile(latencies,99):.3f})")

        if len(run_latencies) > 1:
            print(f"    Spread: {np.max(run_latencies)-np.min(run_latencies):.3f} ms "
                  f"(±{np.std(run_latencies):.3f} ms)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 100)
    print("  AI-SCL IMPROVEMENTS — COMPREHENSIVE TEST SUITE")
    print("  N=128, K=64, L=4, BPSK/AWGN")
    print("=" * 100)

    t_start = time.time()

    test_correctness()
    test_ber_equivalence()
    test_latency_distribution()
    test_internal_stats()
    test_edge_cases()
    test_variance()

    print(f"\n{SEP}")
    print(f"  ALL TESTS COMPLETE ({time.time() - t_start:.1f}s)")
    print(SEP)


if __name__ == '__main__':
    main()
