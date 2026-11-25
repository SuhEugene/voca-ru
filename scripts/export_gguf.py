"""Export a trained checkpoint to a GGUF the JavaScript page can read.

The layout follows ``n4ze3m/tiny-voca``: GGUF v3, float32 tensors named
``model/<attribute>``, plus embedded ``config.json`` and the front-end tables
under ``phonemizer/``. ``js/tiny-voca.js`` reads exactly those names, so a file
written here loads with no adapter.

    python scripts/export_gguf.py inference.pt voca-ru-js.gguf \
        --stress .cache/models/voca-ru/stress_dict.txt \
        --lexicon .cache/models/voca-ru/g2p_lexicon.txt
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

MAGIC = 0x46554747
UINT8, UINT32, UINT64, STRING, ARRAY = 0, 4, 10, 8, 9
F32 = 0
ALIGN = 32


def key_string(key: str, value: str) -> bytes:
    return _kv(key, STRING, _str(value))


def key_uint(key: str, value: int) -> bytes:
    return _kv(key, UINT32, struct.pack("<I", value))


def _str(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _str(key) + struct.pack("<I", kind) + payload


def key_array_u32(key: str, values) -> bytes:
    body = b"".join(struct.pack("<I", v) for v in values)
    return _kv(key, ARRAY, struct.pack("<I", UINT32) + struct.pack("<Q", len(values)) + body)


def key_array_u64(key: str, values) -> bytes:
    body = b"".join(struct.pack("<Q", v) for v in values)
    return _kv(key, ARRAY, struct.pack("<I", UINT64) + struct.pack("<Q", len(values)) + body)


def key_array_string(key: str, values) -> bytes:
    body = b"".join(_str(v) for v in values)
    return _kv(key, ARRAY, struct.pack("<I", STRING) + struct.pack("<Q", len(values)) + body)


def key_array_u8(key: str, values: bytes) -> bytes:
    return _kv(key, ARRAY, struct.pack("<I", UINT8) + struct.pack("<Q", len(values)) + values)


def model_spec(voices, symbols) -> dict:
    """The audiocpp model spec; the JS ignores it, the Rust runtime reads it."""
    return {
        "schema_version": 1,
        "family": "tiny_voca",
        "display_name": "voca-ru",
        "description": "A 4.5M-parameter two-voice Russian text-to-speech model.",
        "status": "community",
        "modes": ["offline"],
        "category": "tts",
        "tasks": ["tts"],
        "languages": ["ru"],
        "capabilities": {"tts": ["built_in_voices", "long_form"]},
        "sources": [{
            "tensors": {"weights": {"prefix": "model", "source": "weights:"}},
            "files": {"config": "model:config.json"},
            "roots": {"weights": "$gguf", "model": "."},
            "format": "gguf",
        }],
        "dependencies": [],
        "ui": {"default_voice": voices[1] if len(voices) > 1 else voices[0],
               "builtin_voices": voices, "tags": ["TTS", "GGUF"], "docs": []},
        "packages": [{"files": ["gguf/voca-ru-js.gguf"], "strip_prefix": "gguf", "precision": "orig",
                      "target_directory": "voca-ru-GGUF", "format": "gguf", "default": True,
                      "display_name": "voca-ru Original-Dtype GGUF", "id": "voca_ru_orig"}],
        "runtime": {"tags": ["gguf"]},
        "options": {"load": [], "session": [], "request": [
            {"name": "voice_id", "type": "enum", "default": voices[1] if len(voices) > 1 else voices[0],
             "values": voices, "required": False, "description": "Built-in voice: F2 (female) or M2 (male)."},
            {"name": "speed", "type": "float", "default": 1, "min": 0.25, "max": 4, "required": False,
             "description": "Speech speed multiplier; larger values produce faster speech."},
        ]},
        "symbols": symbols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stress", type=Path, help="stress dictionary shipped inside the file")
    parser.add_argument("--lexicon", type=Path, help="word -> phonemes lexicon shipped inside the file")
    parser.add_argument("--name", default="voca-ru")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    symbols = state["frontend"]["symbols"]
    vocab = {symbol: index for index, symbol in enumerate(symbols)}

    files: dict[str, bytes] = {
        "config.json": json.dumps({"model": config, "vocab": vocab}, ensure_ascii=False).encode("utf-8"),
    }
    if args.stress:
        files["phonemizer/stress_dict.tsv"] = args.stress.read_bytes()
    if args.lexicon:
        files["phonemizer/g2p_lexicon.tsv"] = args.lexicon.read_bytes()

    tensors = [(name, value.float().contiguous()) for name, value in state["model"].items()]
    tensor_infos, blobs, offset = [], [], 0
    for name, value in tensors:
        data = value.numpy().tobytes()
        tensor_infos.append(_str(f"model/{name}")
                            + struct.pack("<I", value.ndim)
                            + b"".join(struct.pack("<Q", d) for d in reversed(value.shape))
                            + struct.pack("<I", F32) + struct.pack("<Q", offset))
        padded = (len(data) + ALIGN - 1) // ALIGN * ALIGN
        blobs.append(data + b"\0" * (padded - len(data)))
        offset += padded

    names = list(files)
    offsets, running = [], 0
    for name in names:
        offsets.append(running)
        running += len(files[name])
    offsets.append(running)
    blob = b"".join(files[name] for name in names)

    spec = model_spec(config["speakers"], symbols)
    keys = [
        key_string("general.architecture", "audiocpp"),
        key_string("general.name", args.name),
        key_uint("general.alignment", ALIGN),
        key_string("audiocpp.tensor_name_format", "native"),
        key_string("audiocpp.source_format", "orig"),
        key_string("audiocpp.weight_type", "orig"),
        key_uint("audiocpp.model_spec.version", 1),
        key_string("audiocpp.model_spec.family", "tiny_voca"),
        key_string("audiocpp.model_spec.json", json.dumps(spec, ensure_ascii=False)),
        key_array_string("audiocpp.tensor_sources.names", ["model"]),
        key_array_string("audiocpp.tensor_sources.paths", ["model.safetensors"]),
        key_array_string("audiocpp.tensor_names", [f"model/{name}" for name, _ in tensors]),
        key_array_u32("audiocpp.tensor_ranks", [value.ndim for _, value in tensors]),
        key_array_u64("audiocpp.tensor_shapes", [d for _, value in tensors for d in value.shape]),
        key_array_string("audiocpp.embedded_files.names", names),
        key_array_u64("audiocpp.embedded_files.offsets", offsets),
        key_array_u8("audiocpp.embedded_files.data", blob),
    ]

    header = struct.pack("<IIQQ", MAGIC, 3, len(tensors), len(keys)) + b"".join(keys) + b"".join(tensor_infos)
    body = header + b"\0" * ((-len(header)) % ALIGN) + b"".join(blobs)
    args.output.write_bytes(body)
    print(json.dumps({"file": str(args.output), "bytes": len(body), "tensors": len(tensors),
                      "parameters": int(sum(v.numel() for _, v in tensors)),
                      "embedded": names}, ensure_ascii=False))


if __name__ == "__main__":
    main()
