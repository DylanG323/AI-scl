"""
Comprehensive speedup test with 100 messages per SNR.
"""
import time
import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message


def test_comprehensive(messages=100, N=128, K=64, L=4):
    snr_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    ai_codec = AIFastSCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, enable_ai_pruning=True)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)
    
    print('\n' + '=' * 90)
    print(f'AI-FastSCL Comprehensive Test: N={N} K={K} L={L} messages={messages}')
    print('=' * 90)
    print(f'{"SNR(dB)":>8} | {"BER_SCL":>12} | {"BER_AI":>12} | {"TIME_SCL(ms)":>13} | {"TIME_AI(ms)":>13} | {"SPEEDUP":>8}')
    print('-' * 90)
    
    speedups = []
    for snr in snr_list:
        scl_errors = 0
        ai_errors = 0
        
        # Test SCL
        t0 = time.time()
        for _ in range(messages):
            msg = generate_binary_message(K)
            tx = scl_codec.encode(msg)
            rx = bpsk.transmit(tx, snr_db=snr)
            dec = scl_codec.decode(rx)
            if not np.array_equal(msg, dec):
                scl_errors += 1
        t_scl = (time.time() - t0) * 1000 / messages
        
        # Test AI
        t0 = time.time()
        for _ in range(messages):
            msg = generate_binary_message(K)
            tx = ai_codec.encode(msg)
            rx = bpsk.transmit(tx, snr_db=snr)
            dec = ai_codec.decode(rx)
            if not np.array_equal(msg, dec):
                ai_errors += 1
        t_ai = (time.time() - t0) * 1000 / messages
        
        ber_scl = scl_errors / (messages * K)
        ber_ai = ai_errors / (messages * K)
        speedup = t_scl / t_ai
        speedups.append(speedup)
        
        status = "✅" if speedup >= 1.0 else "⚠️ "
        print(f'{snr:>8.1f} | {ber_scl:>12.6f} | {ber_ai:>12.6f} | {t_scl:>13.3f} | {t_ai:>13.3f} | {speedup:>7.2f}x {status}')
    
    print('=' * 90)
    print(f'Average Speedup: {np.mean(speedups):.2f}x')
    print(f'Min Speedup: {np.min(speedups):.2f}x')
    print(f'Max Speedup: {np.max(speedups):.2f}x')
    print(f'Speedups >= 1.0x: {sum(1 for s in speedups if s >= 1.0)} / {len(speedups)}')
    print('=' * 90)


if __name__ == '__main__':
    test_comprehensive(messages=100)
