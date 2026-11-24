#!/usr/bin/env python3
"""All-in-one 6-layer no-clone exporter for PocketTTS.cpp (libpocket_tts.so).

For a given Pocket TTS language config (e.g. french, german, italian,
portuguese, spanish, english_2026-10), this script:
  1. Forces `weights_path = weights_path_without_voice_cloning` so gated
     weights are NEVER accidentally mixed with public no-clone voices.
  2. Exports:
       - text_conditioner.onnx
       - flow_lm_main_int8.onnx  (split fp16 KV, max_seq=1000, tanh GELU)
       - flow_lm_flow_int8.onnx  (flattened SimpleMLPAdaLN, rank-1 c=[1024])
       - mimi_decoder_int8.onnx  (250-slot circular buffer, split fp16 KV)
  3. Copies `tokenizer.json` from the config's pinned tokenizer_path.
  4. Downloads predefined no-clone voice `.safetensors` and converts them
     directly to PocketTTS.cpp `.kv` snapshots in `<out>/voices/<voice>.kv`.
  5. Writes `<out>/bundle.json` and validates `flow_lm_main` against PyTorch.

Run with Soundmind's venv Python (which has latest pocket_tts, onnx, onnxruntime):
  /home/sam/Desktop/projects/soundmind/.venv/bin/python scripts/export_6l_cpp.py \
      --language french --output-dir pocket-tts-onnx-cpp/french
"""

import argparse
import json
import logging
import os
import shutil
import struct
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("root").setLevel(logging.ERROR)

# Disable beartype BEFORE importing pocket_tts (if installed) so ONNX tracing works cleanly.
try:
    import beartype
    import beartype.claw

    beartype.beartype = lambda func: func
    beartype.claw.beartype_this_package = lambda *a, **kw: None
    beartype.claw.beartype_package = lambda *a, **kw: None
except ImportError:
    pass

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from safetensors import safe_open

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.attention import _LinearKVCacheBackend
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.rope import apply_rope
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.config import CONFIGS_DIR, load_config
from pocket_tts.utils.utils import download_if_necessary, get_predefined_voice

OPSET = 17
MIMI_CACHE_LEN = 250
KV_MAGIC = 0x3143564B
T_FLOAT16 = 10
T_INT64 = 7

DEFAULT_VOICES = ("alba", "bill_boerst")
NATIVE_VOICE_BY_LANG = {
    "french": "estelle",
    "german": "juergen",
    "italian": "giovanni",
    "portuguese": "rafael",
    "spanish": "lola",
    "dutch": "daan",
}


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class TextConditionerWrapper(nn.Module):
    def __init__(self, conditioner):
        super().__init__()
        self.embed = conditioner.embed

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(token_ids)


class FlowLMMainCppWrapper(nn.Module):
    """Split-fp16 KV cache backbone matching PocketTTS.cpp StatefulRunner."""

    def __init__(self, flow_lm):
        super().__init__()
        layers = flow_lm.transformer.layers
        self.layers = layers
        self.num_layers = len(layers)
        attn0 = layers[0].self_attn
        self.d_model = attn0.embed_dim
        self.num_heads = attn0.num_heads
        self.dim_per_head = self.d_model // self.num_heads
        cfg = getattr(getattr(flow_lm, "config", None), "transformer", None)
        self.rope_max_period = float(getattr(cfg, "max_period", 10000.0))
        self.bos_emb = flow_lm.bos_emb
        self.input_linear = flow_lm.input_linear
        self.out_norm = flow_lm.out_norm
        self.out_eos = flow_lm.out_eos

    def _attention(self, attn, x, cache_k, cache_v, step):
        B, L, _ = x.shape
        projected = attn.in_proj(x)
        packed = projected.view(B, L, 3, self.num_heads, self.dim_per_head)
        q, k, v = torch.unbind(packed, dim=2)

        offset = step[0]
        q, k = apply_rope(q, k, offset=offset, max_period=self.rope_max_period)

        k_fp16 = k.half()
        v_fp16 = v.half()

        positions = step[0] + torch.arange(L, device=x.device, dtype=torch.long)
        idx = positions.view(1, L, 1, 1).expand(B, L, self.num_heads, self.dim_per_head)
        updated_k = cache_k.scatter(1, idx, k_fp16)
        updated_v = cache_v.scatter(1, idx, v_fp16)

        valid_len = step[0] + L
        valid_k = updated_k[:, :valid_len].float()
        valid_v = updated_v[:, :valid_len].float()

        q_positions = step[0] + torch.arange(L, device=x.device)
        kv_positions = torch.arange(valid_len, device=x.device)
        mask = q_positions.unsqueeze(1) >= kv_positions.unsqueeze(0)
        attn_mask = torch.log(mask.float())

        q_t = q.transpose(1, 2)
        k_t = valid_k.transpose(1, 2)
        v_t = valid_v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        out = attn.out_proj(out)
        new_step = step + L
        return out, updated_k, updated_v, new_step

    def forward(self, sequence, text_embeddings, *states):
        n = self.num_layers
        k_caches = [states[i * 3] for i in range(n)]
        v_caches = [states[i * 3 + 1] for i in range(n)]
        steps = [states[i * 3 + 2] for i in range(n)]

        sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
        input_ = self.input_linear(sequence)
        x = torch.cat([text_embeddings, input_], dim=1)

        out_k, out_v, out_steps = [], [], []
        for i, layer in enumerate(self.layers):
            x_norm = layer.norm1(x)
            attn_out, new_k, new_v, new_step = self._attention(
                layer.self_attn, x_norm, k_caches[i], v_caches[i], steps[i]
            )
            x = x + attn_out
            x_ff = layer.norm2(x)
            x = x + layer.linear2(F.gelu(layer.linear1(x_ff), approximate="tanh"))
            out_k.append(new_k)
            out_v.append(new_v)
            out_steps.append(new_step)

        x = self.out_norm(x)
        conditioning = x[:, -1]
        eos_logit = self.out_eos(conditioning)

        result = [conditioning.squeeze(0), eos_logit.squeeze(0)]
        for i in range(n):
            result.append(out_k[i])
            result.append(out_v[i])
            result.append(out_steps[i])
        return tuple(result)


class FlowLMFlowWrapper(nn.Module):
    """Flattened SimpleMLPAdaLN matching PocketTTS.cpp input shapes."""

    def __init__(self, flow_net):
        super().__init__()
        self.te0_lin1_weight = nn.Parameter(flow_net.time_embed[0].mlp[0].weight.data, requires_grad=False)
        self.te0_lin1_bias = nn.Parameter(flow_net.time_embed[0].mlp[0].bias.data, requires_grad=False)
        self.te0_lin2_weight = nn.Parameter(flow_net.time_embed[0].mlp[2].weight.data, requires_grad=False)
        self.te0_lin2_bias = nn.Parameter(flow_net.time_embed[0].mlp[2].bias.data, requires_grad=False)
        self.te0_rms_alpha = nn.Parameter(flow_net.time_embed[0].mlp[3].alpha.data, requires_grad=False)
        self.te0_rms_eps = flow_net.time_embed[0].mlp[3].eps
        self.te0_freqs = nn.Parameter(flow_net.time_embed[0].freqs.data, requires_grad=False)

        self.te1_lin1_weight = nn.Parameter(flow_net.time_embed[1].mlp[0].weight.data, requires_grad=False)
        self.te1_lin1_bias = nn.Parameter(flow_net.time_embed[1].mlp[0].bias.data, requires_grad=False)
        self.te1_lin2_weight = nn.Parameter(flow_net.time_embed[1].mlp[2].weight.data, requires_grad=False)
        self.te1_lin2_bias = nn.Parameter(flow_net.time_embed[1].mlp[2].bias.data, requires_grad=False)
        self.te1_rms_alpha = nn.Parameter(flow_net.time_embed[1].mlp[3].alpha.data, requires_grad=False)
        self.te1_rms_eps = flow_net.time_embed[1].mlp[3].eps
        self.te1_freqs = nn.Parameter(flow_net.time_embed[1].freqs.data, requires_grad=False)

        self.cond_embed = flow_net.cond_embed
        self.input_proj = flow_net.input_proj
        self.res_blocks = nn.ModuleList(list(flow_net.res_blocks))
        self.final_layer = flow_net.final_layer

    def _timestep_embed(self, t, freqs, w1, b1, w2, b2, rms_alpha, rms_eps):
        args = t * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        x = F.linear(embedding, w1, b1)
        x = F.silu(x)
        x = F.linear(x, w2, b2)
        var = rms_eps + x.var(dim=-1, keepdim=True)
        x = x * (rms_alpha * torch.rsqrt(var))
        return x

    def forward(self, c, s, t, x):
        x = self.input_proj(x)
        t0 = self._timestep_embed(
            s, self.te0_freqs, self.te0_lin1_weight, self.te0_lin1_bias,
            self.te0_lin2_weight, self.te0_lin2_bias, self.te0_rms_alpha, self.te0_rms_eps,
        )
        t1 = self._timestep_embed(
            t, self.te1_freqs, self.te1_lin1_weight, self.te1_lin1_bias,
            self.te1_lin2_weight, self.te1_lin2_bias, self.te1_rms_alpha, self.te1_rms_eps,
        )
        t_combined = (t0 + t1) / 2.0
        c_emb = self.cond_embed(c)
        y = t_combined + c_emb
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


def fix_onnx_identity_dedup(onnx_path: Path):
    model = onnx.load(str(onnx_path))
    init_map = {i.name: i for i in model.graph.initializer}
    nodes_to_remove = []
    inits_to_add = []
    for node in model.graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            src, dst = node.input[0], node.output[0]
            if src in init_map and dst not in init_map:
                src_data = numpy_helper.to_array(init_map[src])
                inits_to_add.append(numpy_helper.from_array(src_data.copy(), name=dst))
                nodes_to_remove.append(node)
    if nodes_to_remove:
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        for init in inits_to_add:
            model.graph.initializer.append(init)
        onnx.save(model, str(onnx_path))


# ---------------------------------------------------------------------------
# Mimi decoder ONNX patches + wrapper (250-slot circular buffer, split fp16)
# ---------------------------------------------------------------------------

_PATCHED_MIMI = False


def _patch_mimi_for_onnx():
    global _PATCHED_MIMI
    if _PATCHED_MIMI:
        return
    _PATCHED_MIMI = True

    def _conv1d_forward_onnx(self, x, model_state):
        B, C, T = x.shape
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self.get_state(model_state)
        TP = state["previous"].shape[-1]

        if TP and self.pad_mode == "replicate":
            assert T >= TP
            init = x[..., :1]
            new_prev = torch.where(
                state["first"].view(-1, 1, 1),
                init.expand_as(state["previous"]),
                state["previous"],
            )
            if model_state is not None:
                self.get_state(model_state)["previous"] = new_prev
            state["previous"] = new_prev

        if TP:
            x = torch.cat([state["previous"], x], dim=-1)
        y = self.conv(x)
        if TP:
            new_prev = x[..., -TP:].contiguous()
            if model_state is not None:
                self.get_state(model_state)["previous"] = new_prev
            if self.pad_mode == "replicate":
                new_first = torch.zeros_like(state["first"])
                if model_state is not None:
                    self.get_state(model_state)["first"] = new_first
        return y

    StreamingConv1d.forward = _conv1d_forward_onnx

    def _convtr_forward_onnx(self, x, mimi_state):
        state = self.get_state(mimi_state)
        layer_state = state["partial"]
        y = self.convtr(x)
        PT = layer_state.shape[-1]
        if PT > 0:
            y_start = y[..., :PT] + layer_state
            new_partial = y[..., -PT:].clone()
            bias = self.convtr.bias
            if bias is not None:
                new_partial = new_partial - bias[:, None]
            self.get_state(mimi_state)["partial"] = new_partial
            y = torch.cat([y_start, y[..., PT:-PT]], dim=-1)
        return y

    StreamingConvTranspose1d.forward = _convtr_forward_onnx

    orig_init_state = _LinearKVCacheBackend.init_state

    def _init_state_split_kv(self, batch_size, sequence_length, device=None, dtype=None):
        orig = orig_init_state(self, batch_size, sequence_length, device, dtype)
        cache = orig["cache"]
        return {
            "cache_k": torch.zeros_like(cache[0], dtype=torch.float16),
            "cache_v": torch.zeros_like(cache[1], dtype=torch.float16),
            "offset": orig["offset"],
        }

    _LinearKVCacheBackend.init_state = _init_state_split_kv

    def _append_and_get_onnx(self, k, v, state):
        if state is None:
            k_attn = k.permute(0, 2, 1, 3)
            v_attn = v.permute(0, 2, 1, 3)
            B = k_attn.shape[0]
            pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
            pos_k = pos_k.view(1, -1).expand(B, -1)
            offset = torch.zeros(B, device=k_attn.device, dtype=torch.long)
            return k_attn, v_attn, pos_k, offset

        cache_k = state["cache_k"]
        cache_v = state["cache_v"]
        offset = state["offset"]
        off = offset[0]

        B, T_new, H, D = k.shape
        capacity = cache_k.shape[1]

        k_fp16 = k.half()
        v_fp16 = v.half()

        write_pos = (off + torch.arange(T_new, device=k.device, dtype=torch.long)) % capacity
        idx = write_pos.view(1, T_new, 1, 1).expand(B, T_new, H, D)

        updated_k = cache_k.scatter(1, idx, k_fp16)
        updated_v = cache_v.scatter(1, idx, v_fp16)
        state["cache_k"] = updated_k
        state["cache_v"] = updated_v

        total_written = off + T_new
        all_slots = torch.arange(capacity, device=k.device, dtype=torch.long)
        last_abs = total_written - 1
        last_slot = last_abs % capacity
        delta = all_slots - last_slot
        abs_positions = torch.where(delta <= 0, last_abs + delta, last_abs + delta - capacity)
        pos_k = torch.where(abs_positions >= 0, abs_positions, torch.full_like(abs_positions, -1))
        pos_k = pos_k.view(1, -1).expand(B, -1)

        k_attn = updated_k.float().permute(0, 2, 1, 3)
        v_attn = updated_v.float().permute(0, 2, 1, 3)
        return k_attn, v_attn, pos_k, offset

    _LinearKVCacheBackend.append_and_get = _append_and_get_onnx

    def _increment_step_onnx(self, state, increment):
        state["offset"] = state["offset"] + increment

    _LinearKVCacheBackend.increment_step = _increment_step_onnx


def _flatten_mimi_states(mimi, seq_len=MIMI_CACHE_LEN):
    states = init_states(mimi, batch_size=1, sequence_length=seq_len)
    for mn in states:
        for key in states[mn]:
            t = states[mn][key]
            if t.is_floating_point():
                states[mn][key] = torch.where(torch.isnan(t), torch.zeros_like(t), t)
    layout = []
    for mn in sorted(states.keys()):
        for key in sorted(states[mn].keys()):
            t = states[mn][key]
            layout.append((mn, key, list(t.shape), t.dtype))
    return layout, states


class MimiDecoderWrapper(nn.Module):
    def __init__(self, model: TTSModel):
        super().__init__()
        self.emb_std = model.flow_lm.emb_std
        self.emb_mean = model.flow_lm.emb_mean
        self.quantizer = model.mimi.quantizer
        self.upsample = model.mimi.upsample
        self.decoder_transformer = model.mimi.decoder_transformer
        self.decoder = model.mimi.decoder
        self.state_layout, _ = _flatten_mimi_states(model.mimi, seq_len=MIMI_CACHE_LEN)
        self.num_states = len(self.state_layout)

    def forward(self, latent, *states):
        model_state = {}
        for i, (mn, key, _, _) in enumerate(self.state_layout):
            model_state.setdefault(mn, {})[key] = states[i]

        x = latent * self.emb_std + self.emb_mean
        x = x.transpose(-1, -2)
        x = self.quantizer(x)
        x = self.upsample(x, model_state)
        (x,) = self.decoder_transformer(x, model_state)
        audio = self.decoder(x, model_state)

        increment = latent.shape[1] * 16
        for mn in sorted(model_state.keys()):
            if "decoder_transformer" in mn and "self_attn" in mn:
                model_state[mn]["offset"] = model_state[mn]["offset"] + increment

        out_states = [model_state[mn][key] for mn, key, _, _ in self.state_layout]
        return (audio,) + tuple(out_states)


# ---------------------------------------------------------------------------
# Voice .safetensors -> PocketTTS.cpp .kv converter
# ---------------------------------------------------------------------------


def convert_safetensors_to_kv(src_safetensors: Path, dst_kv: Path, n_layers: int):
    states = []
    with safe_open(str(src_safetensors), framework="numpy") as f:
        for i in range(n_layers):
            base = f"transformer.layers.{i}.self_attn"
            cache = f.get_tensor(base + "/cache")  # [2, 1, T, H, D] fp32
            step = int(np.asarray(f.get_tensor(base + "/offset")).reshape(-1)[0])
            assert cache.shape[0] == 2 and cache.shape[1] == 1, cache.shape
            k = np.ascontiguousarray(cache[0, 0, :step].astype(np.float16))
            v = np.ascontiguousarray(cache[1, 0, :step].astype(np.float16))
            h, d = k.shape[1], k.shape[2]
            states.append(([1, step, h, d], T_FLOAT16, k.tobytes()))
            states.append(([1, step, h, d], T_FLOAT16, v.tobytes()))
            states.append(([1], T_INT64, np.array([step], dtype="<i8").tobytes()))

    blob = struct.pack("<i", 0) + struct.pack("<i", len(states))
    for shape, typ, raw in states:
        blob += struct.pack("<i", len(shape))
        blob += struct.pack(f"<{len(shape)}q", *shape)
        blob += struct.pack("<i", typ)
        blob += struct.pack("<q", len(raw))
        blob += raw

    dst_kv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_kv, "wb") as f:
        f.write(struct.pack("<I", KV_MAGIC))
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    print(f"  ✓ voices/{dst_kv.name} ({n_layers} layers, {step} slots, {len(blob) / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------


def export_language(language: str, output_dir: Path, max_seq: int, voices: list[str], keep_fp32: bool):
    torch.manual_seed(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(language) if Path(language).is_file() else (CONFIGS_DIR / f"{language}.yaml")
    cfg = load_config(cfg_path)
    if not cfg.weights_path_without_voice_cloning:
        raise RuntimeError(f"Config '{language}' has no weights_path_without_voice_cloning!")

    # FORCE public no-clone weights so we NEVER mix gated weights with no-clone voices!
    cfg.weights_path = cfg.weights_path_without_voice_cloning
    print(f"=== Exporting {language} (no-clone: {cfg.weights_path}) ===")

    tts = TTSModel._from_pydantic_config_with_weights(
        cfg,
        temp=cfg.default_temperature or 0.7,
        sampler_decode_steps=1,
        noise_clamp=None,
        eos_threshold=-4.0,
        origin=None,
    ).cpu().eval()
    tts.has_voice_cloning = False

    n_layers = len(tts.flow_lm.transformer.layers)
    print(f"  Loaded: layers={n_layers}, sample_rate={tts.sample_rate}")

    # 1. Text conditioner
    cond_path = output_dir / "text_conditioner.onnx"
    cond_wrapper = TextConditionerWrapper(tts.flow_lm.conditioner).eval()
    for p in cond_wrapper.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        torch.onnx.export(
            cond_wrapper,
            (torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long),),
            str(cond_path),
            input_names=["token_ids"],
            output_names=["embeddings"],
            dynamic_axes={"token_ids": {1: "text_len"}, "embeddings": {1: "text_len"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  ✓ {cond_path.name} ({cond_path.stat().st_size / 1e6:.1f} MB)")

    # 2. FlowLM Main (split fp16 KV)
    main_fp32 = output_dir / "flow_lm_main.onnx"
    main_int8 = output_dir / "flow_lm_main_int8.onnx"
    main_wrapper = FlowLMMainCppWrapper(tts.flow_lm).eval()
    for p in main_wrapper.parameters():
        p.requires_grad_(False)

    n, h, d = main_wrapper.num_layers, main_wrapper.num_heads, main_wrapper.dim_per_head
    dummy_seq = torch.randn(1, 1, tts.flow_lm.ldim)
    dummy_text = torch.zeros(1, 0, 1024)
    dummy_states = []
    for _ in range(n):
        dummy_states.append(torch.zeros(1, max_seq, h, d, dtype=torch.float16))
        dummy_states.append(torch.zeros(1, max_seq, h, d, dtype=torch.float16))
        dummy_states.append(torch.tensor([10], dtype=torch.long))

    in_names = ["sequence", "text_embeddings"]
    out_names = ["conditioning", "eos_logit"]
    for i in range(n):
        in_names.extend([f"state_{i*3}", f"state_{i*3+1}", f"state_{i*3+2}"])
        out_names.extend([f"out_state_{i*3}", f"out_state_{i*3+1}", f"out_state_{i*3+2}"])

    with torch.no_grad():
        torch.onnx.export(
            main_wrapper,
            (dummy_seq, dummy_text, *dummy_states),
            str(main_fp32),
            input_names=in_names,
            output_names=out_names,
            dynamic_axes={"sequence": {1: "seq_len"}, "text_embeddings": {1: "text_len"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  ✓ {main_fp32.name} ({main_fp32.stat().st_size / 1e6:.1f} MB)")

    # Numeric validation of flow_lm_main vs PyTorch
    sess = ort.InferenceSession(str(main_fp32), providers=["CPUExecutionProvider"])
    tokens = tts.flow_lm.conditioner.tokenizer("Bonjour le monde.")
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    with torch.no_grad():
        text_emb = tts.flow_lm.conditioner.embed(tokens)

    def fresh_states():
        out = []
        for _ in range(n):
            out.append(torch.zeros(1, max_seq, h, d, dtype=torch.float16))
            out.append(torch.zeros(1, max_seq, h, d, dtype=torch.float16))
            out.append(torch.tensor([0], dtype=torch.long))
        return out

    with torch.no_grad():
        pt_text = main_wrapper(torch.zeros(1, 0, tts.flow_lm.ldim), text_emb, *fresh_states())
    feed = {"sequence": np.zeros((1, 0, tts.flow_lm.ldim), dtype=np.float32), "text_embeddings": text_emb.numpy()}
    for i, s in enumerate(fresh_states()):
        feed[f"state_{i}"] = s.numpy()
    ort_text = sess.run(None, feed)
    cond_diff = float(np.abs(pt_text[0].numpy().reshape(-1) - ort_text[0].reshape(-1)).max())
    eos_diff = float(np.abs(pt_text[1].numpy().reshape(-1) - ort_text[1].reshape(-1)).max())
    print(f"  ✓ numeric check: cond_diff={cond_diff:.2e}, eos_diff={eos_diff:.2e}")
    assert cond_diff < 2e-3 and eos_diff < 5e-3, f"Numeric check failed: {cond_diff}, {eos_diff}"
    del sess

    quantize_dynamic(
        str(main_fp32), str(main_int8),
        weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
    )
    print(f"  ✓ {main_int8.name} ({main_int8.stat().st_size / 1e6:.1f} MB)")

    # 3. FlowLM Flow network
    flow_fp32 = output_dir / "flow_lm_flow.onnx"
    flow_int8 = output_dir / "flow_lm_flow_int8.onnx"
    flow_wrapper = FlowLMFlowWrapper(tts.flow_lm.flow_net).eval()
    for p in flow_wrapper.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        torch.onnx.export(
            flow_wrapper,
            (torch.randn(1024), torch.zeros(1, 1), torch.ones(1, 1), torch.randn(1, 32)),
            str(flow_fp32),
            input_names=["c", "s", "t", "x"],
            output_names=["flow_dir"],
            dynamic_axes={},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    fix_onnx_identity_dedup(flow_fp32)
    quantize_dynamic(
        str(flow_fp32), str(flow_int8),
        weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
    )
    print(f"  ✓ {flow_int8.name} ({flow_int8.stat().st_size / 1e6:.1f} MB)")

    # 4. Mimi Decoder (250-slot circular buffer)
    mimi_fp32 = output_dir / "mimi_decoder.onnx"
    mimi_int8 = output_dir / "mimi_decoder_int8.onnx"
    _patch_mimi_for_onnx()
    mimi_wrapper = MimiDecoderWrapper(tts).eval()
    for p in mimi_wrapper.parameters():
        p.requires_grad_(False)
    _, init_mimi = _flatten_mimi_states(tts.mimi, seq_len=MIMI_CACHE_LEN)
    dummy_mimi_states = [
        init_mimi[mn][key] for mn in sorted(init_mimi.keys()) for key in sorted(init_mimi[mn].keys())
    ]
    mimi_in = ["latent"] + [f"state_{i}" for i in range(len(dummy_mimi_states))]
    mimi_out = ["audio_frame"] + [f"out_state_{i}" for i in range(len(dummy_mimi_states))]
    with torch.no_grad():
        torch.onnx.export(
            mimi_wrapper,
            (torch.randn(1, 1, 32), *dummy_mimi_states),
            str(mimi_fp32),
            input_names=mimi_in,
            output_names=mimi_out,
            dynamic_axes={"latent": {1: "latent_frames"}, "audio_frame": {2: "audio_samples"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    quantize_dynamic(
        str(mimi_fp32), str(mimi_int8),
        weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
    )
    print(f"  ✓ {mimi_int8.name} ({mimi_int8.stat().st_size / 1e6:.1f} MB)")

    if not keep_fp32:
        for p in (main_fp32, flow_fp32, mimi_fp32):
            p.unlink(missing_ok=True)

    # 5. Tokenizer
    tok_src = download_if_necessary(cfg.flow_lm.lookup_table.tokenizer_path)
    tok_name = "tokenizer.json" if str(tok_src).endswith(".json") else "tokenizer.model"
    shutil.copy2(tok_src, output_dir / tok_name)
    print(f"  ✓ {tok_name}")

    # 6. Predefined no-clone voices -> .kv
    exported_voices = []
    for voice in voices:
        try:
            st_path = download_if_necessary(get_predefined_voice(language=language, name=voice))
            convert_safetensors_to_kv(st_path, output_dir / "voices" / f"{voice}.kv", n_layers)
            exported_voices.append(voice)
        except Exception as e:
            print(f"  ! skipped voice '{voice}': {e}")

    assert exported_voices, f"No voices exported for {language}!"

    # 7. bundle.json
    meta = {
        "format": "pocket-tts-cpp-v1",
        "language": language,
        "num_layers": n_layers,
        "max_seq": max_seq,
        "mimi_cap": MIMI_CACHE_LEN,
        "sample_rate": int(tts.sample_rate),
        "tokenizer_file": tok_name,
        "default_temperature": float(cfg.default_temperature or 0.7),
        "remove_semicolons": bool(cfg.remove_semicolons),
        "replace_characters": dict(cfg.replace_characters or {}),
        "voices": exported_voices,
    }
    (output_dir / "bundle.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"=== Done: {output_dir} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, help="e.g. french, german, italian, portuguese, spanish, english_2026-10")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=1000)
    ap.add_argument("--voices", nargs="+", default=list(DEFAULT_VOICES))
    ap.add_argument("--keep-fp32", action="store_true")
    args = ap.parse_args()
    voices = list(args.voices)
    native = NATIVE_VOICE_BY_LANG.get(args.language)
    if native and native not in voices:
        voices.append(native)
    export_language(args.language, Path(args.output_dir), args.max_seq, voices, args.keep_fp32)


if __name__ == "__main__":
    main()
