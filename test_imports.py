"""Minimal test of AI-SCL"""
import sys
print("Starting import...")

try:
    import numpy as np
    print("numpy imported OK")
except Exception as e:
    print(f"numpy import failed: {e}")

try:
    from python_polar_coding.polar_codes.base.decoder import BaseDecoder
    print("BaseDecoder imported OK")
except Exception as e:
    print(f"BaseDecoder import failed: {e}")
    sys.exit(1)

try:
    from python_polar_coding.polar_codes.ai_scl.decoding_path import AIPath
    print("AIPath imported OK")
except Exception as e:
    print(f"AIPath import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from python_polar_coding.polar_codes.ai_scl.decoder import AISCLDecoder
    print("AISCLDecoder imported OK")
except Exception as e:
    print(f"AISCLDecoder import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful!")
print("Testing decoder creation...")

try:
    mask = np.concatenate([np.zeros(64), np.ones(64)])
    decoder = AISCLDecoder(n=7, mask=mask, L=4)
    print("Decoder created OK")
except Exception as e:
    print(f"Decoder creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All tests passed!")
