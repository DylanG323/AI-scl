"""Test AI-SCL decoding"""
import numpy as np
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message
from python_polar_coding.channels import SimpleBPSKModulationAWGN

print("Creating codecs...")
scl_codec = SCListPolarCodec(N=128, K=64, design_snr=0.0, L=4)
ai_codec = AISCLPolarCodec(N=128, K=64, design_snr=0.0, L=4)
bpsk = SimpleBPSKModulationAWGN(fec_rate=64/128)
print("Codecs created OK")

print("\nGenerating test message...")
message = generate_binary_message(64)
tx = scl_codec.encode(message)
received = bpsk.transmit(tx, snr_db=3.0)
print("Test message generated OK")

print("\nTesting SCL codec...")
scl_result = scl_codec.decode(received)
scl_correct = np.array_equal(scl_result, message)
print(f"SCL decoded: correct={scl_correct}")

print("\nTesting AI-SCL codec...")
ai_result = ai_codec.decode(received)
ai_correct = np.array_equal(ai_result, message)
print(f"AI-SCL decoded: correct={ai_correct}")

print(f"\nResults match: {np.array_equal(scl_result, ai_result)}")
print("Test completed successfully!")
