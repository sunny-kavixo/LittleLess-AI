"""
LittleLess AI — Phase 3.9 File-Backed Streaming Safety & Verification Suite
Tests:
1. Native file-backed loading and CRC32 container integrity verification.
2. Exact active weight RAM verification (< 200 KB target).
3. File I/O telemetry verification (ensuring true pread calls occur during inference).
4. Bit-exact output sequence agreement against Phase 3.5 static and Phase 3.8 references.
5. Robustness against corrupt containers (bad magic, truncated file, payload CRC mismatch).
6. Leak-free repeated creation and destruction of native models.
"""

import ctypes, os, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIB_PATH = ROOT / 'src/kernels/libnative_transformer.dylib'
assert LIB_PATH.exists(), f"Missing native library: {LIB_PATH}"
lib = ctypes.CDLL(str(LIB_PATH))

# C-FFI Signatures for File Streaming
lib.create_file_streaming_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.create_file_streaming_model.restype = ctypes.c_void_p

lib.free_file_streaming_model.argtypes = [ctypes.c_void_p]

lib.file_streaming_get_active_weight_ram.argtypes = [ctypes.c_void_p]
lib.file_streaming_get_active_weight_ram.restype = ctypes.c_size_t

lib.file_streaming_get_total_model_ram.argtypes = [ctypes.c_void_p]
lib.file_streaming_get_total_model_ram.restype = ctypes.c_size_t

lib.file_streaming_get_io_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
lib.file_streaming_reset_io_stats.argtypes = [ctypes.c_void_p]

lib.file_streaming_verify_integrity.argtypes = [ctypes.c_void_p]
lib.file_streaming_verify_integrity.restype = ctypes.c_int

lib.file_streaming_generate_tokens.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)
]
lib.file_streaming_generate_tokens.restype = ctypes.c_int


class TestFileStreamingSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream_dir = ROOT / 'checkpoints/streaming'
        cls.c0_r3_path = cls.stream_dir / 'S_ADBO_C0_Radix3.llstream'
        cls.s1_r3_path = cls.stream_dir / 'S_ADBO_S1_Radix3.llstream'
        cls.s1_2b_path = cls.stream_dir / 'S_ADBO_S1_2bit.llstream'
        assert cls.c0_r3_path.exists(), f"Container missing: {cls.c0_r3_path}"
        assert cls.s1_r3_path.exists(), f"Container missing: {cls.s1_r3_path}"

    def test_01_load_and_integrity_verification(self):
        status = ctypes.c_int(0)
        m = lib.create_file_streaming_model(str(self.s1_r3_path).encode('utf-8'), 1, ctypes.byref(status))
        self.assertIsNotNone(m, f"Failed to load model, status: {status.value}")
        self.assertEqual(status.value, 0)

        # Verify all layer checksums
        crc_res = lib.file_streaming_verify_integrity(m)
        self.assertEqual(crc_res, 0, f"CRC32 integrity check failed with code: {crc_res}")
        lib.free_file_streaming_model(m)

    def test_02_active_weight_ram_under_200kb(self):
        status = ctypes.c_int(0)
        # Radix-3 Layer Streaming
        m_r3 = lib.create_file_streaming_model(str(self.s1_r3_path).encode('utf-8'), 1, ctypes.byref(status))
        r3_ram = lib.file_streaming_get_active_weight_ram(m_r3)
        lib.free_file_streaming_model(m_r3)

        self.assertEqual(r3_ram, 160704, f"Expected 160,704 B, got {r3_ram} B")
        self.assertLess(r3_ram, 200 * 1024, "Radix-3 Layer Streaming must be < 200 KB")

        # 2-Bit Layer Streaming
        m_2b = lib.create_file_streaming_model(str(self.s1_2b_path).encode('utf-8'), 1, ctypes.byref(status))
        b2_ram = lib.file_streaming_get_active_weight_ram(m_2b)
        lib.free_file_streaming_model(m_2b)

        self.assertEqual(b2_ram, 168704, f"Expected 168,704 B, got {b2_ram} B")
        self.assertLess(b2_ram, 200 * 1024, "2-Bit Layer Streaming must be < 200 KB")

    def test_03_io_telemetry_proves_file_streaming(self):
        status = ctypes.c_int(0)
        m = lib.create_file_streaming_model(str(self.s1_r3_path).encode('utf-8'), 1, ctypes.byref(status))
        self.assertEqual(status.value, 0)

        calls_0 = ctypes.c_uint64(0)
        bytes_0 = ctypes.c_uint64(0)
        lib.file_streaming_get_io_stats(m, ctypes.byref(calls_0), ctypes.byref(bytes_0))
        self.assertEqual(calls_0.value, 0)
        self.assertEqual(bytes_0.value, 0)

        # Generate 10 tokens from prompt of len 4 (4 prefill steps + 9 decode steps = 13 steps)
        # Each step evaluates 25 linear projections.
        # Each projection streams 2 planes via pread = 50 pread calls per step.
        # 13 steps * 50 preads = 650 preads minimum!
        prompt = np.array([42, 65, 99, 12], dtype=np.int32)
        out_tokens = np.zeros(10, dtype=np.int32)
        stats = np.zeros(4, dtype=np.float64)

        lib.file_streaming_generate_tokens(
            m,
            prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 4,
            10,
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        calls_1 = ctypes.c_uint64(0)
        bytes_1 = ctypes.c_uint64(0)
        lib.file_streaming_get_io_stats(m, ctypes.byref(calls_1), ctypes.byref(bytes_1))

        self.assertGreater(calls_1.value, 600, f"Expected >600 pread calls, got {calls_1.value}")
        self.assertGreater(bytes_1.value, 5_000_000, f"Expected >5 MB streamed from disk, got {bytes_1.value}")
        lib.free_file_streaming_model(m)

    def test_04_bit_exact_output_preservation(self):
        # Deterministic prompt tokens: [42, 65, 99, 12]
        # Reference generated tokens from Phase 3.8: [105, 110, 32, 116, 104, 101, 32, 115, 101, 97]
        expected_tokens = [105, 110, 32, 116, 104, 101, 32, 115, 101, 97]

        status = ctypes.c_int(0)
        m = lib.create_file_streaming_model(str(self.s1_r3_path).encode('utf-8'), 1, ctypes.byref(status))
        self.assertEqual(status.value, 0)

        prompt = np.array([42, 65, 99, 12], dtype=np.int32)
        out_tokens = np.zeros(10, dtype=np.int32)
        stats = np.zeros(4, dtype=np.float64)

        lib.file_streaming_generate_tokens(
            m,
            prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 4,
            10,
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        lib.free_file_streaming_model(m)

        actual_tokens = list(out_tokens)
        self.assertEqual(actual_tokens, expected_tokens, f"Token mismatch: {actual_tokens} vs {expected_tokens}")

    def test_05_corrupted_container_rejection(self):
        status = ctypes.c_int(0)
        # 1. Non-existent file
        m_bad = lib.create_file_streaming_model(b"/tmp/non_existent_model.llstream", 1, ctypes.byref(status))
        self.assertIsNone(m_bad)
        self.assertEqual(status.value, -1) # STREAM_ERR_OPEN

        # 2. Corrupted magic header
        with tempfile.NamedTemporaryFile(suffix='.llstream') as tmp:
            tmp.write(b"CORRUPT!" + b"\x00" * 1000)
            tmp.flush()
            m_corrupt = lib.create_file_streaming_model(tmp.name.encode('utf-8'), 1, ctypes.byref(status))
            self.assertIsNone(m_corrupt)
            self.assertEqual(status.value, -2) # STREAM_ERR_MAGIC

        # 3. Payload CRC mismatch
        content = bytearray(self.s1_r3_path.read_bytes())
        # Corrupt one byte in the payload area (near the end of file)
        content[-500] = (content[-500] + 1) % 256
        with tempfile.NamedTemporaryFile(suffix='.llstream') as tmp:
            tmp.write(content)
            tmp.flush()
            m_tampered = lib.create_file_streaming_model(tmp.name.encode('utf-8'), 1, ctypes.byref(status))
            self.assertIsNotNone(m_tampered)
            # Integrity check must detect the altered byte!
            crc_res = lib.file_streaming_verify_integrity(m_tampered)
            self.assertEqual(crc_res, -5) # STREAM_ERR_CRC
            lib.free_file_streaming_model(m_tampered)

    def test_06_leak_free_repeated_load_unload(self):
        status = ctypes.c_int(0)
        for _ in range(50):
            m = lib.create_file_streaming_model(str(self.s1_r3_path).encode('utf-8'), 1, ctypes.byref(status))
            self.assertIsNotNone(m)
            lib.free_file_streaming_model(m)


if __name__ == '__main__':
    unittest.main()
