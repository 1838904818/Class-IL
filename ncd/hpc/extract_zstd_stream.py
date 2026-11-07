"""Stream a zstd frame to stdout using the system libzstd (no Python package)."""

import ctypes
import sys


class _InBuffer(ctypes.Structure):
    _fields_ = [("src", ctypes.c_void_p), ("size", ctypes.c_size_t), ("pos", ctypes.c_size_t)]


class _OutBuffer(ctypes.Structure):
    _fields_ = [("dst", ctypes.c_void_p), ("size", ctypes.c_size_t), ("pos", ctypes.c_size_t)]


def _decoder():
    errors = []
    for name in ("libzstd.so.1", "libzstd.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    else:
        raise RuntimeError("unable to load system libzstd: " + "; ".join(errors))

    lib.ZSTD_createDStream.argtypes = []
    lib.ZSTD_createDStream.restype = ctypes.c_void_p
    lib.ZSTD_initDStream.argtypes = [ctypes.c_void_p]
    lib.ZSTD_initDStream.restype = ctypes.c_size_t
    lib.ZSTD_DStreamInSize.argtypes = []
    lib.ZSTD_DStreamInSize.restype = ctypes.c_size_t
    lib.ZSTD_DStreamOutSize.argtypes = []
    lib.ZSTD_DStreamOutSize.restype = ctypes.c_size_t
    lib.ZSTD_decompressStream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_OutBuffer),
        ctypes.POINTER(_InBuffer),
    ]
    lib.ZSTD_decompressStream.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
    lib.ZSTD_getErrorName.restype = ctypes.c_char_p
    lib.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
    lib.ZSTD_freeDStream.restype = ctypes.c_size_t
    return lib


def _check(lib, code):
    if lib.ZSTD_isError(code):
        name = lib.ZSTD_getErrorName(code).decode("utf-8", "replace")
        raise RuntimeError(f"libzstd streaming decode failed: {name}")
    return code


def _stream(source, sink):
    lib = _decoder()
    stream = lib.ZSTD_createDStream()
    if not stream:
        raise RuntimeError("libzstd could not allocate a decompression stream")
    try:
        _check(lib, lib.ZSTD_initDStream(stream))
        input_size = lib.ZSTD_DStreamInSize()
        output_size = lib.ZSTD_DStreamOutSize()
        output_storage = ctypes.create_string_buffer(output_size)
        remaining = 1

        while True:
            chunk = source.read(input_size)
            if not chunk:
                break
            input_storage = ctypes.create_string_buffer(chunk)
            in_buffer = _InBuffer(ctypes.cast(input_storage, ctypes.c_void_p), len(chunk), 0)
            while in_buffer.pos < in_buffer.size:
                out_buffer = _OutBuffer(ctypes.cast(output_storage, ctypes.c_void_p), output_size, 0)
                old_pos = in_buffer.pos
                remaining = _check(lib, lib.ZSTD_decompressStream(stream, ctypes.byref(out_buffer), ctypes.byref(in_buffer)))
                if out_buffer.pos:
                    sink.write(output_storage.raw[:out_buffer.pos])
                if in_buffer.pos == old_pos and out_buffer.pos == 0:
                    raise RuntimeError("libzstd decoder made no progress")

        if remaining != 0:
            raise RuntimeError("truncated zstd input: final frame is incomplete")
        sink.flush()
    finally:
        _check(lib, lib.ZSTD_freeDStream(stream))


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} ARCHIVE.zst|-", file=sys.stderr)
        return 2
    source = sys.stdin.buffer if argv[1] == "-" else open(argv[1], "rb")
    try:
        _stream(source, sys.stdout.buffer)
    except (BrokenPipeError, OSError, RuntimeError) as exc:
        print(f"extract_zstd_stream: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not sys.stdin.buffer:
            source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
