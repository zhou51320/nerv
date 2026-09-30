#!/usr/bin/env python3
"""Exercise DLL startup and cross-CRT model file I/O without a GPU or model download."""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


def gguf_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    args = parser.parse_args()
    server = args.server.resolve(strict=True)
    arch = "nerv_win7_smoke"
    # A valid metadata-only GGUF must reach llama_file before model creation
    # rejects the unknown architecture. /MT across DLLs fails at that file I/O.
    data = struct.pack("<4sIQQ", b"GGUF", 3, 0, 1)
    data += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string(arch)
    with tempfile.TemporaryDirectory(prefix="kvmem-startup-") as temp:
        model = Path(temp) / "probe.gguf"
        model.write_bytes(data)
        result = subprocess.run(
            [str(server), "-m", str(model), "-ngl", "0", "--no-kvmem",
             "--load-mode", "none", "--no-ui"],
            cwd=server.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    print(result.stdout)
    expected = "unknown model architecture: '" + arch + "'"
    if result.returncode != 1 or expected not in result.stdout:
        raise SystemExit(
            f"Startup/model I/O regression: exit={result.returncode} "
            f"(0x{result.returncode & 0xffffffff:08X}), expected exit=1 and {expected!r}"
        )
    print("PASS: DLL startup and model file I/O; controlled architecture rejection (not inference).")


if __name__ == "__main__":
    main()
