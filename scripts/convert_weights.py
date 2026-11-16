"""Convert the PyTorch checkpoint to the safetensors file the Rust port reads.

    python scripts/convert_weights.py weights/tiny-voca.pt weights/tiny-voca.safetensors

The model configuration, voices and symbol inventory travel in the metadata.
"""
import json
import sys

import torch
from safetensors.torch import save_file

source, target = sys.argv[1], sys.argv[2]
state = torch.load(source, map_location="cpu", weights_only=True)
if state.get("format_version") != 4:
    raise SystemExit("unsupported checkpoint format")
tensors = {name: value.float().contiguous() for name, value in state["model"].items()}
metadata = {"config": json.dumps(state["config"]), "step": str(state.get("step")),
            "frontend": json.dumps({k: v for k, v in state["frontend"].items() if k != "symbols"}, ensure_ascii=False),
            "symbols": json.dumps(state["frontend"]["symbols"], ensure_ascii=False)}
save_file(tensors, target, metadata=metadata)
print(f"wrote {target}: {len(tensors)} tensors, {sum(t.numel() for t in tensors.values()):,} parameters")
