# Phase 4.0 Python crash — local diagnosis

**Status:** Root cause reported as identified by local investigation; independent reproduction in this public repository is pending.

A macOS crash report at `~/Library/Logs/DiagnosticReports/Python-2026-10-20-103108.ips` reportedly associated the crash with an ad-hoc Python `ctypes` diagnostic, not the main Phase 4.0 benchmark suite.

- **Fault:** `EXC_BAD_ACCESS (SIGSEGV)`, native destructor / `free_phase4_model`.
- **Mechanism:** The temporary script omitted `free_phase4_model.argtypes = [ctypes.c_void_p]`. Without an explicit pointer argument signature, the native pointer was passed using an incompatible default integer type, potentially truncating the address on a 64-bit process.
- **Recovery:** A minimal test with the correct `ctypes` pointer signature reportedly created and freed a model successfully.
- **Benchmark status:** Two benchmark tasks reportedly exited with code 0 and produced an intact 11-row CSV.
- **Boundary:** This diagnosis does **not** certify the entire C++ engine as memory-safe.

**Prevention:** Centralize native FFI bindings; explicitly set every function's `argtypes` and `restype`; test lifecycle and invalid pointers; use ASan/UBSan or equivalent native diagnostics when practical. Do not publish the private crash log itself without reviewing sensitive system information.
