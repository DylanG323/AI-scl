"""
Test to measure AI-SCL speedup vs regular SCL.
"""
import time
import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message


def test_aiscl_speedup(messages=50, N=128, K=64, L=4):
    snr_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    ai_codec = AISCLPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)
    
    print('\n' + '=' * 80)
    print(f'AI-SCL Speedup Test: N={N} K={K} L={L} messages={messages}')
    print('=' * 80)
    print(f'{"SNR(dB)":>8} | {"TIME_SCL(ms)":>13} | {"TIME_AI(ms)":>13} | {"SPEEDUP":>8}')
    print('-' * 80)
    
    for snr in snr_list:
        # Test SCL
        t0 = time.time()
        for _ in range(messages):
            msg = generate_binary_message(K)
            tx = scl_codec.encode(msg)
            rx = bpsk.transmit(tx, snr_db=snr)
            _ = scl_codec.decode(rx)
        t_scl = (time.time() - t0) * 1000 / messages
        
        # Test AI-SCL
        t0 = time.time()
        for _ in range(messages):
            msg = generate_binary_message(K)
            tx = ai_codec.encode(msg)
            rx = bpsk.transmit(tx, snr_db=snr)
            _ = ai_codec.decode(rx)
        t_ai = (time.time() - t0) * 1000 / messages
        
        speedup = t_scl / t_ai
        status = "✅" if speedup > 1.0 else "⚠️ "
        print(f'{snr:>8.1f} | {t_scl:>13.3f} | {t_ai:>13.3f} | {speedup:>7.2f}x {status}')
    
    print('=' * 80)


if __name__ == '__main__':
    test_aiscl_speedup(messages=50)
