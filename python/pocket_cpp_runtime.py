"""Standalone cross-platform ctypes wrapper for pocket-tts-cpp-multilingual.

Supports Linux (.so), Windows (.dll), and macOS (.dylib), HuggingFace
``tokenizer.json`` (6-layer multilingual models) or SentencePiece
``tokenizer.model``, and precomputed ``.kv`` voice snapshots.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

_LIB_HANDLE: ctypes.CDLL | None = None


def _platform_lib_names() -> tuple[tuple[str, ...], tuple[str, ...], str]:
    arch = platform.machine().lower()
    if arch in ("amd64", "x86_64"):
        arch = "x64"
    elif arch in ("aarch64", "arm64"):
        arch = "arm64"

    if sys.platform == "win32":
        return (("pocket_tts.dll", "libpocket_tts.dll"), ("onnxruntime.dll",), f"win-{arch}")
    if sys.platform == "darwin":
        return (("libpocket_tts.dylib",), ("libonnxruntime.dylib",), f"macos-{arch}")
    return (("libpocket_tts.so",), ("libonnxruntime.so.1", "libonnxruntime.so"), f"linux-{arch}")


def _find_native_lib(lib_dir: str | Path | None = None) -> Path:
    lib_names, _, plat_dir = _platform_lib_names()

    candidates: list[Path] = []
    if lib_dir:
        p = Path(lib_dir)
        if p.is_file():
            return p
        candidates.extend([p / plat_dir, p])

    env_lib = os.getenv("POCKET_TTS_CPP_LIB")
    if env_lib:
        p = Path(env_lib)
        if p.is_file():
            return p
        candidates.extend([p / plat_dir, p])

    here = Path(__file__).resolve().parent
    candidates.extend([here / plat_dir, here, here.parent / plat_dir, here.parent])

    for d in candidates:
        for name in lib_names:
            if (d / name).is_file():
                return d / name
    raise FileNotFoundError(
        f"Native library ({' / '.join(lib_names)}) not found for {plat_dir}. "
        "Pass lib_dir= or set POCKET_TTS_CPP_LIB."
    )


def _load_library(lib_dir: str | Path | None = None) -> ctypes.CDLL:
    global _LIB_HANDLE
    if _LIB_HANDLE is not None:
        return _LIB_HANDLE

    _, ort_names, _ = _platform_lib_names()
    lib_path = _find_native_lib(lib_dir)
    native_dir = lib_path.parent

    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(native_dir))

    ort_candidates = [native_dir / name for name in ort_names]
    try:
        import onnxruntime as ort

        capi_dir = Path(ort.__file__).resolve().parent / "capi"
        if sys.platform == "win32" and hasattr(os, "add_dll_directory") and capi_dir.is_dir():
            os.add_dll_directory(str(capi_dir))
        for pattern in ("*onnxruntime*.dll", "libonnxruntime*.dylib", "libonnxruntime.so*"):
            ort_candidates.extend(sorted(capi_dir.glob(pattern)))
    except Exception:
        pass

    dl_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for ort_lib in ort_candidates:
        if ort_lib.is_file():
            try:
                ctypes.CDLL(str(ort_lib), mode=dl_mode) if dl_mode else ctypes.CDLL(str(ort_lib))
                break
            except OSError:
                continue

    lib = ctypes.CDLL(str(lib_path))
    lib.ptt_create.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ]
    lib.ptt_create.restype = ctypes.c_void_p
    lib.ptt_destroy.argtypes = [ctypes.c_void_p]
    lib.ptt_destroy.restype = None
    lib.ptt_free_audio.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ptt_free_audio.restype = None
    lib.ptt_stream_start_tokens.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_float,
    ]
    lib.ptt_stream_start_tokens.restype = ctypes.c_void_p
    lib.ptt_stream_read.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_int),
    ]
    lib.ptt_stream_read.restype = ctypes.c_int
    lib.ptt_stream_end.argtypes = [ctypes.c_void_p]
    lib.ptt_stream_end.restype = None

    _LIB_HANDLE = lib
    return lib


class _TextTokenizer:
    def __init__(self, bundle_dir: Path, tokenizer_file: str | None = None):
        candidates = []
        if tokenizer_file:
            candidates.append(bundle_dir / tokenizer_file)
        candidates.extend([bundle_dir / "tokenizer.json", bundle_dir / "tokenizer.model"])
        tok_path = next((p for p in candidates if p.is_file()), None)
        if tok_path is None:
            raise FileNotFoundError(f"No tokenizer.json or tokenizer.model in {bundle_dir}")
        if tok_path.suffix == ".json":
            import tokenizers

            self._kind = "json"
            self._tok = tokenizers.Tokenizer.from_file(str(tok_path))
        else:
            import sentencepiece as spm

            self._kind = "sp"
            self._tok = spm.SentencePieceProcessor(str(tok_path))

    def encode(self, text: str) -> list[int]:
        if self._kind == "json":
            return list(self._tok.encode(text).ids)
        return list(self._tok.encode(text, out_type=int))


class PocketTTSCpp:
    def __init__(
        self,
        bundle_dir: str | Path,
        precision: str = "int8",
        temperature: float | None = None,
        lsd_steps: int = 1,
        num_threads: int = 2,
        lib_dir: str | Path | None = None,
    ):
        self.bundle_dir = Path(bundle_dir)
        meta_path = self.bundle_dir / "bundle.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        self.sample_rate = int(meta.get("sample_rate", 24000))
        self.default_temperature = float(
            temperature if temperature is not None else meta.get("default_temperature", 0.55)
        )
        self._remove_semicolons = bool(meta.get("remove_semicolons", False))
        self._replace_chars = dict(meta.get("replace_characters") or {})
        self._tokenizer = _TextTokenizer(self.bundle_dir, meta.get("tokenizer_file"))
        self._lib = _load_library(lib_dir)

        handle = self._lib.ptt_create(
            str(self.bundle_dir).encode("utf-8"),
            str(self.bundle_dir / "voices").encode("utf-8"),
            b"",
            precision.encode("utf-8"),
            ctypes.c_float(self.default_temperature),
            ctypes.c_int(lsd_steps),
            ctypes.c_int(num_threads),
        )
        if not handle:
            raise RuntimeError(f"ptt_create failed for {self.bundle_dir}")
        self._handle: int | None = handle

    def _normalize_text(self, text: str) -> str:
        t = text.strip()
        if not t:
            return ""
        if self._remove_semicolons:
            t = t.replace(";", ",")
        for src, dst in self._replace_chars.items():
            if src in t:
                t = t.replace(src, dst)
        t = " ".join(t.split())
        if not t:
            return ""
        if t[-1].isalnum():
            t += "."
        if t[0].islower():
            t = t[0].upper() + t[1:]
        return t

    def stream(
        self,
        text: str,
        voice: str | Path,
        frames_after_eos: int = 3,
        temperature: float | None = None,
    ) -> Iterator[np.ndarray]:
        if not self._handle:
            raise RuntimeError("PocketTTSCpp is closed")
        normalized = self._normalize_text(text)
        if not normalized:
            return
        token_ids = self._tokenizer.encode(normalized)
        if not token_ids:
            return

        ids_arr = (ctypes.c_int64 * len(token_ids))(*token_ids)
        temp_arg = float(temperature) if temperature is not None else -1.0
        stream_ctx = self._lib.ptt_stream_start_tokens(
            self._handle,
            ids_arr,
            len(token_ids),
            str(voice).encode("utf-8"),
            ctypes.c_int(frames_after_eos),
            ctypes.c_float(temp_arg),
        )
        if not stream_ctx:
            raise RuntimeError(f"ptt_stream_start_tokens failed for {voice}")

        try:
            out_ptr = ctypes.POINTER(ctypes.c_float)()
            out_len = ctypes.c_int(0)
            while True:
                rc = self._lib.ptt_stream_read(
                    stream_ctx, ctypes.byref(out_ptr), ctypes.byref(out_len)
                )
                if rc <= 0:
                    break
                n = out_len.value
                if n > 0 and out_ptr:
                    chunk = np.ctypeslib.as_array(out_ptr, shape=(n,)).copy()
                    self._lib.ptt_free_audio(out_ptr)
                    yield chunk
                elif out_ptr:
                    self._lib.ptt_free_audio(out_ptr)
        finally:
            self._lib.ptt_stream_end(stream_ctx)

    def generate(
        self,
        text: str,
        voice: str | Path,
        frames_after_eos: int = 3,
        temperature: float | None = None,
    ) -> np.ndarray:
        chunks = list(self.stream(text, voice, frames_after_eos=frames_after_eos, temperature=temperature))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        if self._handle:
            self._lib.ptt_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
