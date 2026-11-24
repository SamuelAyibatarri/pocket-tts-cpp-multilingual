# pocket-tts-cpp-multilingual

Cross-platform C++ ONNX Runtime FFI library, multilingual 6-layer exporter, and Python `ctypes` wrapper for **Kyutai Pocket TTS** — running **3x–4x faster than realtime** on a 4-core CPU with **~85–115 ms** time-to-first-audio across English, French, German, Italian, Portuguese, and Spanish.

## Attribution

- Base C++ runtime: [VolgaGerm/PocketTTS.cpp](https://github.com/VolgaGerm/PocketTTS.cpp) (MIT License)
- Models & architecture: [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) (`kyutai/pocket-tts-without-voice-cloning`)

## What Was Changed / Added

1. **Fixed 250-slot Mimi circular-buffer wrap artifact (`max_chunk_frames`: `15` → `3`)**
   - In the upstream runtime, decoding 15 latent frames at once (240 Mimi transformer steps) overwrote 240 of 250 circular cache slots before attention ran on the first frame of the chunk, causing a sharp audio glitch around ~1.5–2 s.
   - Setting `max_chunk_frames = 3` eliminates the artifact completely and tightens AR/decoder pipelining (~3.1x–3.9x realtime on 4-core CPU).
2. **Token-ID streaming C API (`ptt_stream_start_tokens`) + in-memory `.kv` voice cache**
   - Allows Python (or any host) to tokenize via Hugging Face `tokenizer.json` (used by Kyutai's newer 6-layer multilingual models) and stream directly from C++ using precomputed `.kv` voice snapshots — no SentencePiece `tokenizer.model` or dummy `.wav` files required.
3. **Optional `tokenizer.model` and `mimi_encoder.onnx`**
   - Skips loading the 72 MB voice-cloning encoder when not present on disk, keeping each language bundle at **~80 MB zipped**.
4. **All-in-one 6-layer no-clone exporter (`export_6l_cpp.py`)**
   - Forces `weights_path = weights_path_without_voice_cloning` (prevents gated/no-clone voice mismatches), exports split-fp16 KV `flow_lm_main_int8.onnx`, `flow_lm_flow_int8.onnx`, 250-slot `mimi_decoder_int8.onnx`, `text_conditioner.onnx`, validates numerically against PyTorch, copies `tokenizer.json`, and converts predefined voices (including native speakers per language) directly to `.kv`.
5. **Cross-platform CMake + GitHub Actions CI + Python `ctypes` wrapper (`python/pocket_cpp_runtime.py`)**
   - Builds `libpocket_tts.so` (Linux x64), `pocket_tts.dll` (Windows x64), and `libpocket_tts.dylib` (macOS arm64).

## Benchmarks (4-core CPU, INT8, `num_threads=2`, `lsd_steps=1`)

| Language | Native Voice | RTF | Speed | First Audio | Bundle Zip |
|---|---|---|---|---|---|
| `english_2026-10` | `alba` | **0.269** | **3.71x RT** | **104 ms** | 78.9 MB |
| `french` | `estelle` | **0.308** | **3.25x RT** | **115 ms** | 80.2 MB |
| `german` | `juergen` | **0.277** | **3.61x RT** | **98 ms** | 79.8 MB |
| `italian` | `giovanni` | **0.322** | **3.10x RT** | **106 ms** | 79.4 MB |
| `portuguese` | `rafael` | **0.257** | **3.90x RT** | **85 ms** | 81.4 MB |
| `spanish` | `lola` | **0.327** | **3.05x RT** | **96 ms** | 80.2 MB |

## Build

```bash
cmake -B .build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIB=ON
cmake --build .build --config Release --target pocket-tts-lib -j4
```

Prebuilt binaries for Linux, Windows, and macOS are built automatically by `.github/workflows/build-libs.yml`.

## Export a 6-Layer Language Bundle

```bash
python export_6l_cpp.py --language french --output-dir bundles/french
```

## Python Usage

```python
from python.pocket_cpp_runtime import PocketTTSCpp

eng = PocketTTSCpp("bundles/french", precision="int8", lsd_steps=1, num_threads=2)
for chunk in eng.stream("Bonjour, ceci fonctionne en local.", "bundles/french/voices/estelle.kv"):
    # chunk is float32 mono PCM at 24 kHz
    ...
eng.close()
```
