#!/usr/bin/env python3
"""Write a complete, deterministic, tiny Qwen4Exp GGUF using only stdlib.

The random weights are test data, not a trained language model. The fixture has
three gated-delta-net layers, one QSA layer, hyper-connections, routed/shared
experts, PLE n-gram embeddings and a byte-fallback tokenizer. No MTP or vision.
Tensor dimensions below follow the pinned llama.cpp qwen4exp loader.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct


def string(value):
    value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def scalar(kind, value):
    if kind == 8:
        return string(value)
    return struct.pack({2: "<H", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q"}[kind], value)


def metadata_value(kind, value):
    if kind == 9:
        item_kind, items = value
        return struct.pack("<IQ", item_kind, len(items)) + b"".join(scalar(item_kind, x) for x in items)
    return scalar(kind, value)


def write_gguf(path, metadata, tensors):
    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        header += string(key) + struct.pack("<I", kind) + metadata_value(kind, value)
    offset = 0
    for name, dimensions, kind, data in tensors:
        header += string(name) + struct.pack("<I", len(dimensions))
        header += struct.pack("<" + "Q" * len(dimensions), *dimensions)
        header += struct.pack("<IQ", kind, offset)
        offset += (len(data) + 31) // 32 * 32
    header += bytes(-len(header) % 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(header)
        for _, _, _, data in tensors:
            out.write(data)
            out.write(bytes(-len(data) % 32))


def fixture(seed=20261002, expert_type="f32", ple_type="f32"):
    # Quantized PLE rows need 32 elements per block; two PLE heads concatenate
    # to the model width, so only these optional variants use width 64.
    d, hc, lowrank = (64 if ple_type in ("q8_0", "iq4_nl") else 32), 2, 8
    layers, experts, used, hidden, shared = 4, 4, 2, 64, 64
    state, k_heads, v_heads, kernel = 32, 1, 2, 4
    attn_heads, kv_heads, head_dim, idx_heads, idx_dim = 2, 1, 32, 2, 32
    ple_head_dim, ple_rows = d // 2, 68
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{x:02X}>" for x in range(256)]
    tokens += ["▁"] + list("abcdefghijklmnopqrstuvwxyz.,!?0123456789")
    types = [2, 3, 3] + [6] * 256 + [1] * (len(tokens) - 259)
    meta = [
        ("general.architecture", 8, "qwen4exp"),
        ("general.name", 8, "PMI synthetic complete Qwen4Exp fixture"),
        ("general.alignment", 4, 32),
        ("general.file_type", 4, {"f32": 0, "f16": 1, "q8_0": 7}[expert_type]),
        ("tokenizer.ggml.model", 8, "llama"),
        ("tokenizer.ggml.tokens", 9, (8, tokens)),
        ("tokenizer.ggml.scores", 9, (6, [0.0] * len(tokens))),
        ("tokenizer.ggml.token_type", 9, (5, types)),
        ("tokenizer.ggml.bos_token_id", 4, 1),
        ("tokenizer.ggml.eos_token_id", 4, 2),
        ("tokenizer.ggml.unknown_token_id", 4, 0),
        ("tokenizer.ggml.add_bos_token", 7, True),
        ("tokenizer.ggml.add_eos_token", 7, False),
    ]
    arch = {
        "context_length": 256,
        "embedding_length": d,
        "block_count": layers,
        "feed_forward_length": hidden * used,
        "expert_feed_forward_length": hidden,
        "expert_shared_feed_forward_length": shared,
        "expert_count": experts,
        "expert_used_count": used,
        "attention.head_count": attn_heads,
        "attention.head_count_kv": kv_heads,
        "attention.key_length": head_dim,
        "attention.value_length": head_dim,
        "rope.dimension_count": head_dim,
        "ssm.conv_kernel": kernel,
        "ssm.inner_size": state * v_heads,
        "ssm.state_size": state,
        "ssm.time_step_rank": v_heads,
        "ssm.group_count": k_heads,
        "hyper_connection.count": hc,
        "hyper_connection.low_rank": lowrank,
        "attention.indexer.head_count": idx_heads,
        "attention.indexer.key_length": idx_dim,
        "attention.indexer.top_k": 8,
        "full_attention_interval": 4,
        "ple.ngram_size": 3,
        "ple.heads_per_ngram": 1,
        "ple.conv_kernel": 3,
        "ple.eos_token_id": 2,
        "embedding_length_per_layer_input": ple_head_dim,
    }
    meta += [("qwen4exp." + key, 4, value) for key, value in arch.items()]
    meta += [
        ("qwen4exp.attention.layer_norm_rms_epsilon", 6, 1e-6),
        ("qwen4exp.rope.freq_base", 6, 10000.0),
        ("qwen4exp.rope.dimension_sections", 9, (5, [16, 0, 0, 0])),
        ("qwen4exp.attention.compress_ratios", 9, (4, [0, 0, 0, 2])),
        ("qwen4exp.ple.layers", 9, (4, [1])),
        ("qwen4exp.ple.layer_multipliers", 9, (10, [17379548441, 21253603391, 22313444153])),
        ("qwen4exp.ple.head_offsets", 9, (10, [0, 31])),
        ("qwen4exp.ple.head_vocab_sizes", 9, (10, [31, 37])),
    ]
    tensors = []

    def add(name, dimensions, mode="matrix"):
        # Seed per tensor so variants retain exactly the same logical weights.
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "little"))
        count = math.prod(dimensions)
        if mode == "norm":
            values = [1.0 + rng.uniform(-0.03, 0.03) for _ in range(count)]
        elif mode == "ssm_a":
            values = [-0.5 - rng.random() for _ in range(count)]
        elif mode == "bias":
            values = [-1.0 + rng.uniform(-0.1, 0.1) for _ in range(count)]
        else:
            scale = {"embedding": 0.3, "conv": 0.15}.get(mode, 0.7 / math.sqrt(dimensions[0]))
            values = [rng.uniform(-scale, scale) for _ in range(count)]
        if name == "output.weight":
            # Keep generated pieces printable, and prevent random EOS from
            # stopping parity tests before the recurrent decode path runs.
            # Byte fallback remains available for arbitrary input text.
            values[:259*d] = [0.0] * (259*d)
        kind = {"f32": 0, "f16": 1, "q8_0": 8}[expert_type] if "_exps.weight" in name else 0
        if name == "per_layer_token_embd.weight":
            kind = {"f32": 0, "f16": 1, "q8_0": 8, "iq4_nl": 20}[ple_type]
        if kind in (8, 20):
            assert dimensions[0] % 32 == 0
            data = bytearray()
            for first in range(0, count, 32):
                block = values[first:first + 32]
                scale = max(abs(value) for value in block) / 127.0
                if kind == 8:
                    quantized = [max(-127, min(127, round(value / scale))) if scale else 0 for value in block]
                    data += struct.pack("<e32b", scale, *quantized)
                else:
                    # A valid nearest-code IQ4_NL encoder for synthetic data;
                    # no claim to reproduce the optimized production quantizer.
                    levels = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
                    codes = [min(range(16), key=lambda code: abs(value - scale * levels[code])) for value in block]
                    packed = [codes[j] | codes[j+16] << 4 for j in range(16)]
                    data += struct.pack("<e16B", scale, *packed)
            data = bytes(data)
        else:
            data = struct.pack("<" + ("e" if kind == 1 else "f") * count, *values)
        tensors.append((name, dimensions, kind, data))

    add("token_embd.weight", (d, len(tokens)), "embedding")
    add("output.weight", (d, len(tokens)))
    add("output_hc_norm.weight", (d, hc), "norm")
    add("output_hc_down.weight", (d * hc, lowrank))
    add("output_hc_up.weight", (lowrank, d * hc))
    add("per_layer_token_embd.weight", (ple_head_dim, ple_rows), "embedding")
    for layer in range(layers):
        prefix = f"blk.{layer}."

        def weight(name, dimensions, mode="matrix", suffix="weight"):
            add(prefix + name + ("." + suffix if suffix else ""), dimensions, mode)

        for sub in ("attn", "ffn"):
            weight("hc_" + sub + "_norm", (d, hc), "norm")
            weight("hc_" + sub + "_down", (d * hc, lowrank))
            weight("hc_" + sub + "_up", (lowrank, d * hc))
            weight("hc_" + sub + "_inject", (d * hc, hc))
        if layer == 3:
            weight("attn_q", (d, head_dim * attn_heads * 2))
            weight("attn_k", (d, head_dim * kv_heads))
            weight("attn_v", (d, head_dim * kv_heads))
            weight("attn_output", (head_dim * attn_heads, d))
            weight("attn_q_norm", (head_dim,), "norm")
            weight("attn_k_norm", (head_dim,), "norm")
            weight("indexer.q_proj", (d, idx_heads * idx_dim))
            weight("indexer.k_proj", (d, idx_dim))
            weight("indexer.q_norm", (idx_dim,), "norm")
            weight("indexer.k_norm", (idx_dim,), "norm")
        else:
            conv_dim = state * (2 * k_heads + v_heads)
            weight("attn_qkv", (d, conv_dim))
            weight("attn_gate", (d, state * v_heads))
            weight("ssm_conv1d", (kernel, conv_dim), "conv")
            weight("ssm_dt", (v_heads,), "bias", "bias")
            weight("ssm_a", (v_heads,), "ssm_a", "")
            weight("ssm_beta", (d, v_heads))
            weight("ssm_alpha", (d, v_heads))
            weight("ssm_norm", (state,), "norm")
            weight("ssm_out", (state * v_heads, d))
        if layer == 1:
            weight("ple_key", (d, d * hc))
            weight("ple_value", (d, d))
            for name in ("key", "query", "conv"):
                weight("ple_norm_" + name, (d, hc), "norm")
            weight("ple_conv1d", (3, d * hc), "conv")
        weight("ffn_gate_inp", (d, experts))
        weight("ffn_down_exps", (hidden, d, experts))
        weight("ffn_gate_exps", (d, hidden, experts))
        weight("ffn_up_exps", (d, hidden, experts))
        weight("ffn_gate_inp_shexp", (d,))
        weight("ffn_gate_shexp", (d, shared))
        weight("ffn_up_shexp", (d, shared))
        weight("ffn_down_shexp", (shared, d))
    return meta, tensors


def generate(output, *, seed=20261002, shards=1, expert_type="f32", ple_type="f32"):
    metadata, tensors = fixture(seed, expert_type, ple_type)
    output = Path(output)
    paths = []
    for shard in range(shards):
        path = output if shards == 1 else output.with_name(f"{output.stem}-{shard+1:05d}-of-{shards:05d}.gguf")
        shard_meta = metadata.copy()
        if shards > 1:
            shard_meta += [("split.no", 2, shard), ("split.count", 2, shards), ("split.tensors.count", 5, len(tensors))]
        # Interleaving puts experts and PLE on distinct shards and exercises lookup.
        write_gguf(path, shard_meta, tensors[shard::shards])
        paths.append(path)
    report = {
        "fixture": "synthetic-untrained-qwen4exp", "seed": seed, "expert_type": expert_type, "ple_type": ple_type,
        "layers": 4, "gdn_layers": [0, 1, 2], "qsa_layers": [3], "ple_layers": [1],
        "tensors": len(tensors), "tensor_bytes": sum(len(t[3]) for t in tensors),
        "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--shards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--expert-type", choices=("f32", "f16", "q8_0"), default="f32")
    parser.add_argument("--ple-type", choices=("f32", "f16", "q8_0", "iq4_nl"), default="f32")
    args = parser.parse_args()
    for path in generate(args.output, seed=args.seed, shards=args.shards, expert_type=args.expert_type, ple_type=args.ple_type):
        print(path)
