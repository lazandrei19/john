from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from romanian_whist.agents.model import PolicyNetworkConfig, WhistPolicyNetwork

CHECKPOINT_VERSION = 2


def save_checkpoint(
    path: Path,
    model: WhistPolicyNetwork,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
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
    checkpoint_version = int(payload.get("checkpoint_version", 0))
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint {path} is incompatible with the current model architecture. "
            "Expected checkpoint_version={expected}, found {actual}. Retrain or resume from a new-format checkpoint.".format(
                path=path,
                expected=CHECKPOINT_VERSION,
                actual=checkpoint_version,
            )
        )
    raw_config = payload.get("model_config", {})
    model_config = PolicyNetworkConfig(
        **dict(
            (field.name, raw_config.get(field.name, field.default))
            for field in fields(PolicyNetworkConfig)
        )
    )
    model = WhistPolicyNetwork.from_config(model_config)
    try:
        model.load_state_dict(payload["model_state"])
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint {path} is incompatible with the current model architecture: {error}".format(
                path=path,
                error=exc,
            )
        ) from exc
    model.to(device)
    return model, payload
