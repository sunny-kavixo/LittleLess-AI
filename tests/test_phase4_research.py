"""
LittleLess AI — Phase 4.0 Research Test Suite
Tests:
1. Chunk-compressed .llchunk container loading, decompression, and CRC32 verification.
2. Dynamic Int8 Quantized KV Cache verification (71.9% KV RAM reduction).
3. Bit-exact token generation across KV modes and container formats.
4. Telemetry tracking: streamed bytes reduction vs decompression overhead.
5. Tamper and corruption rejection on chunk containers.
"""

import ctypes, os, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIB_PATH = ROOT / 'src/kernels/libnative_transformer.dylib'
assert LIB_PATH.exists()
lib = ctypes.CDLL(str(LIB_PATH))

# C-FFI Signatures for Phase 4
lib.create_phase4_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.create_phase4_model.restype = ctypes.c_void_p

lib.free_phase4_model.argtypes = [ctypes.c_void_p]

lib.phase4_get_active_weight_ram.argtypes = [ctypes.c_void_p]
lib.phase4_get_active_weight_ram.restype = ctypes.c_size_t

lib.phase4_get_kv_cache_ram.argtypes = [ctypes.c_void_p]
lib.phase4_get_kv_cache_ram.restype = ctypes.c_size_t

lib.phase4_get_total_model_ram.argtypes = [ctypes.c_void_p]
lib.phase4_get_total_model_ram.restype = ctypes.c_size_t

lib.phase4_get_telemetry_stats.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)
]
lib.phase4_reset_telemetry_stats.argtypes = [ctypes.c_void_p]

lib.phase4_verify_integrity.argtypes = [ctypes.c_void_p]
lib.phase4_verify_integrity.restype = ctypes.c_int

lib.phase4_generate_tokens.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)
]
lib.phase4_generate_tokens.restype = ctypes.c_int


class TestPhase4Research(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream_dir = ROOT / 'checkpoints/streaming'
        cls.s1_r3_stream = cls.stream_dir / 'S_ADBO_S1_Radix3.llstream'
        cls.s1_r3_chunk = cls.stream_dir / 'S_ADBO_S1_Radix3.llchunk'
        assert cls.s1_r3_stream.exists(), f"Missing: {cls.s1_r3_stream}"
        assert cls.s1_r3_chunk.exists(), f"Missing: {cls.s1_r3_chunk}"

    def test_01_chunk_container_integrity(self):
        status = ctypes.c_int(0)
        m = lib.create_phase4_model(str(self.s1_r3_chunk).encode('utf-8'), 1, 0, ctypes.byref(status))
        self.assertIsNotNone(m)
        self.assertEqual(status.value, 0)

        # Integrity check: decompresses every chunk and validates CRC32
        crc_res = lib.phase4_verify_integrity(m)
        self.assertEqual(crc_res, 0, f"CRC integrity failed: {crc_res}")
        lib.free_phase4_model(m)

    def test_02_kv_cache_ram_reduction(self):
        status = ctypes.c_int(0)
        # 1. FP32 KV Cache Mode
        m_fp32 = lib.create_phase4_model(str(self.s1_r3_stream).encode('utf-8'), 1, 0, ctypes.byref(status))
        kv_fp32 = lib.phase4_get_kv_cache_ram(m_fp32)
        tot_fp32 = lib.phase4_get_total_model_ram(m_fp32)
        lib.free_phase4_model(m_fp32)

        self.assertEqual(kv_fp32, 655360, f"Expected 655,360 B, got {kv_fp32}")

        # 2. Dynamic Int8 Quantized KV Cache Mode
        m_int8 = lib.create_phase4_model(str(self.s1_r3_stream).encode('utf-8'), 1, 1, ctypes.byref(status))
        kv_int8 = lib.phase4_get_kv_cache_ram(m_int8)
        tot_int8 = lib.phase4_get_total_model_ram(m_int8)
        lib.free_phase4_model(m_int8)

        self.assertEqual(kv_int8, 184320, f"Expected 184,320 B, got {kv_int8}")
        reduction_pct = (1.0 - kv_int8 / kv_fp32) * 100.0
        self.assertAlmostEqual(reduction_pct, 71.875, places=2)
        
        # Total model RAM should drop by ~471 KB (from ~824 KB to ~353 KB)
        self.assertEqual(tot_fp32 - tot_int8, 471040)
        self.assertLess(tot_int8, 360 * 1024, "Total model heap with Int8 KV should be <360 KB")

    def test_03_token_generation_agreement(self):
        expected_tokens = [105, 110, 32, 116, 104, 101, 32, 115, 101, 97]
        prompt = np.array([42, 65, 99, 12], dtype=np.int32)
        out_tokens = np.zeros(10, dtype=np.int32)
        stats = np.zeros(4, dtype=np.float64)

        status = ctypes.c_int(0)
        # Test 1: Chunk-compressed container with FP32 KV
        m1 = lib.create_phase4_model(str(self.s1_r3_chunk).encode('utf-8'), 1, 0, ctypes.byref(status))
        lib.phase4_generate_tokens(
            m1, prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 4, 10,
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        lib.free_phase4_model(m1)
        self.assertEqual(list(out_tokens), expected_tokens)

        # Test 2: Chunk-compressed container with Int8 KV
        out_tokens.fill(0)
        m2 = lib.create_phase4_model(str(self.s1_r3_chunk).encode('utf-8'), 1, 1, ctypes.byref(status))
        lib.phase4_generate_tokens(
            m2, prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 4, 10,
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        lib.free_phase4_model(m2)
        self.assertEqual(list(out_tokens), expected_tokens)

    def test_04_chunk_streaming_telemetry(self):
        status = ctypes.c_int(0)
        m = lib.create_phase4_model(str(self.s1_r3_chunk).encode('utf-8'), 1, 1, ctypes.byref(status))
        
        prompt = np.array([42, 65, 99, 12], dtype=np.int32)
        out_tokens = np.zeros(10, dtype=np.int32)
        stats = np.zeros(4, dtype=np.float64)

        lib.phase4_generate_tokens(
            m, prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 4, 10,
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        p_calls = ctypes.c_uint64(0)
        b_streamed = ctypes.c_uint64(0)
        d_calls = ctypes.c_uint64(0)
        d_bytes = ctypes.c_uint64(0)
        lib.phase4_get_telemetry_stats(m, ctypes.byref(p_calls), ctypes.byref(b_streamed), ctypes.byref(d_calls), ctypes.byref(d_bytes))
        lib.free_phase4_model(m)

        # 13 forward steps * 25 coalesced pread calls = 325 pread calls (down from 650!)
        self.assertEqual(p_calls.value, 325)
        self.assertEqual(d_calls.value, 650)
        # Raw decompressed bytes: 13 steps * 443,136 B = 5,760,768 B
        self.assertEqual(d_bytes.value, 5760768)
        # Streamed bytes (compressed): 13 steps * 314,720 B = 4,091,360 B (29% bandwidth saving!)
        self.assertEqual(b_streamed.value, 4091360)

    def test_05_tampered_chunk_rejection(self):
        status = ctypes.c_int(0)
        raw_bytes = bytearray(self.s1_r3_chunk.read_bytes())
        # Corrupt one compressed byte
        raw_bytes[200000] ^= 0xFF
        with tempfile.NamedTemporaryFile(suffix='.llchunk') as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            m_corrupt = lib.create_phase4_model(tmp.name.encode('utf-8'), 1, 0, ctypes.byref(status))
            self.assertIsNotNone(m_corrupt)
            res = lib.phase4_verify_integrity(m_corrupt)
            # Decompression or CRC error
            self.assertIn(res, [-5, -7])
            lib.free_phase4_model(m_corrupt)

    def test_06_stre_container_integrity_and_generation(self):
        stre_path = self.stream_dir / 'S_ADBO_S1_Radix3.llstre'
        self.assertTrue(stre_path.exists())
        status = ctypes.c_int(0)
        m = lib.create_phase4_model(str(stre_path).encode('utf-8'), 1, 1, ctypes.byref(status))
        self.assertIsNotNone(m)
        self.assertEqual(status.value, 0)

        # Integrity check
        self.assertEqual(lib.phase4_verify_integrity(m), 0)

        # Bit-exact token generation
        prompt = np.array([42, 65, 99, 12], dtype=np.int32)
        out_tokens = np.zeros(10, dtype=np.int32)
        stats = np.zeros(4, dtype=np.float64)
        lib.phase4_generate_tokens(
            m, prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 4, 10,
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        self.assertEqual(list(out_tokens), [105, 110, 32, 116, 104, 101, 32, 115, 101, 97])
        lib.free_phase4_model(m)


if __name__ == '__main__':
    unittest.main()
