"""
Compare pruning vs no pruning to understand speedup source.
"""
import time
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message

scl = SCListPolarCodec(N=128, K=64, design_snr=0.0, L=4)
ai_pruning = AIFastSCLPolarCodec(N=128, K=64, design_snr=0.0, L=4, enable_ai_pruning=True)
ai_no_pruning = AIFastSCLPolarCodec(N=128, K=64, design_snr=0.0, L=4, enable_ai_pruning=False)
bpsk = SimpleBPSKModulationAWGN(fec_rate=64/128)

print('SNR 0.0, 30 messages')
print()

# SCL
t0 = time.time()
for i in range(30):
    msg = generate_binary_message(64)
    tx = scl.encode(msg)
    rx = bpsk.transmit(tx, snr_db=0.0)
    dec = scl.decode(rx)
t_scl = (time.time() - t0) * 1000 / 30

# AI with pruning
t0 = time.time()
for i in range(30):
    msg = generate_binary_message(64)
    tx = ai_pruning.encode(msg)
    rx = bpsk.transmit(tx, snr_db=0.0)
    dec = ai_pruning.decode(rx)
t_ai_pruning = (time.time() - t0) * 1000 / 30

# AI without pruning
t0 = time.time()
for i in range(30):
    msg = generate_binary_message(64)
    tx = ai_no_pruning.encode(msg)
    rx = bpsk.transmit(tx, snr_db=0.0)
    dec = ai_no_pruning.decode(rx)
t_ai_no_pruning = (time.time() - t0) * 1000 / 30

print(f'SCL:                {t_scl:.3f}ms')
print(f'AI (with pruning):  {t_ai_pruning:.3f}ms -> {t_scl/t_ai_pruning:.2f}x')
print(f'AI (no pruning):    {t_ai_no_pruning:.3f}ms -> {t_scl/t_ai_no_pruning:.2f}x')
