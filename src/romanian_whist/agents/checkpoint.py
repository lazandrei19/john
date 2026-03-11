from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from romanian_whist.agents.model import PolicyNetworkConfig, WhistPolicyNetwork


def save_checkpoint(
    path: Path,
    model: WhistPolicyNetwork,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy_checkpoint(path: Path, device: str = "cpu") -> Tuple[WhistPolicyNetwork, Dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    raw_config = payload.get("model_config", {})
    model_config = PolicyNetworkConfig(
        **dict(
            (field.name, raw_config.get(field.name, field.default))
            for field in fields(PolicyNetworkConfig)
        )
    )
    model = WhistPolicyNetwork.from_config(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, payload
