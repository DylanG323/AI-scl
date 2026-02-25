"""Debug BER issue in AI-SCL"""
import numpy as np
from python_polar_coding.polar_codes.scl.decoder import SCLDecoder
from python_polar_coding.polar_codes.ai_scl.decoder import AISCLDecoder
from python_polar_coding.polar_codes.utils import get_random_bits

# Single test message
N, K = 128, 64
L = 4
SNR_dB = 3.0
sigma = np.sqrt(1.0 / (2.0 * 10**(SNR_dB / 10.0)))

info_bits = get_random_bits(K)
encoded = np.ones(N) - 2.0 * info_bits
noise = np.random.normal(0, sigma, N)
received = encoded + noise

print("Single message test at SNR=3.0 dB")
print()

# Test SCL
print("Testing SCL...")
scl_decoder = SCLDecoder(n=7, mask=np.concatenate([np.zeros(64), np.ones(64)]), L=L)
scl_result = scl_decoder.decode(received)
scl_correct = np.allclose(scl_result[-K:], info_bits)

print(f'SCL correct: {scl_correct}')
print(f'SCL result (first 10): {scl_result[-K:][:10]}')
print(f'Info bits (first 10): {info_bits[:10]}')
print()

# Test AI-SCL
print("Testing AI-SCL...")
ai_decoder = AISCLDecoder(n=7, mask=np.concatenate([np.zeros(64), np.ones(64)]), L=L)
ai_result = ai_decoder.decode(received)
ai_correct = np.allclose(ai_result[-K:], info_bits)

print(f'AI-SCL correct: {ai_correct}')
print(f'AI-SCL result (first 10): {ai_result[-K:][:10]}')
print(f'Info bits (first 10): {info_bits[:10]}')
print()

# Check if they match each other
print(f'SCL and AI-SCL results match: {np.allclose(scl_result, ai_result)}')
print(f'Both correct: {scl_correct and ai_correct}')
print(f'SCL only: {scl_correct and not ai_correct}')
print(f'AI-SCL only: {not scl_correct and ai_correct}')
print(f'Both wrong: {not scl_correct and not ai_correct}')
