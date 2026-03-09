from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch


class CheckpointConverter:
    def export(self, checkpoint_path: Path, output_dir: Path) -> Dict[str, str]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_map = {}
        for name, tensor in payload["model_state"].items():
            tensor_map[name] = tensor.detach().cpu().numpy()
        npz_path = output_dir / "model.npz"
        np.savez(npz_path, **tensor_map)

        metadata = {
            "has_mlx": self.available(),
            "state_dict_keys": sorted(tensor_map.keys()),
            "model_config": payload.get("model_config", {}),
            "source_checkpoint": str(checkpoint_path),
            "training_metadata": payload.get("metadata", {}),
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return {"weights": str(npz_path), "metadata": str(metadata_path)}

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("mlx") is not None
