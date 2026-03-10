from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from romanian_whist.agents.model import PolicyForwardOutputs, WhistPolicyNetwork, batch_observations, masked_logits


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    belief_loss_coef: float = 0.2
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 4
    imitation_batch_size: int = 128
    imitation_epochs: int = 3
    max_grad_norm: float = 1.0
    mixed_precision: bool = True


@dataclass
class RolloutBuffer:
    observations: List[Mapping[str, object]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    next_values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    trajectory_ids: List[int] = field(default_factory=list)

    def add(
        self,
        observation: Mapping[str, object],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        trajectory_id: int = 0,
    ) -> int:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.next_values.append(0.0)
        self.rewards.append(reward)
        self.dones.append(done)
        self.trajectory_ids.append(trajectory_id)
        return len(self.actions) - 1

    def __len__(self) -> int:
        return len(self.actions)


class PPOTrainer:
    def __init__(self, policy: WhistPolicyNetwork, config: PPOConfig, device: str = "cpu"):
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        scaler_enabled = config.mixed_precision and device.startswith("cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def select_action(self, observation: Mapping[str, object]) -> tuple[int, float, float]:
        self.policy.eval()
        with torch.no_grad():
            batch = batch_observations([observation], device=torch.device(self.device))
            outputs = self.policy.forward_with_aux(batch)
            filtered_logits = masked_logits(outputs.logits, batch["legal_action_mask"])
            distribution = torch.distributions.Categorical(logits=filtered_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(outputs.values.item())

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if not buffer:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "belief_loss": 0.0}

        returns, advantages = self._returns_and_advantages(buffer)
        observations = batch_observations(buffer.observations, device=torch.device(self.device))
        actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std().clamp_min(1e-6))

        metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "belief_loss": 0.0}
        indices = np.arange(len(buffer))
        for _ in range(self.config.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                batch_obs = dict((key, value[batch_idx]) for key, value in observations.items())
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                autocast_enabled = self.config.mixed_precision and self.device.startswith("cuda")
                autocast = torch.amp.autocast if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast
                with autocast("cuda", enabled=autocast_enabled) if autocast is torch.amp.autocast else autocast(enabled=autocast_enabled):
                    outputs = self.policy.forward_with_aux(batch_obs)
                    filtered_logits = masked_logits(outputs.logits, batch_obs["legal_action_mask"])
                    distribution = torch.distributions.Categorical(logits=filtered_logits)
                    new_log_probs = distribution.log_prob(batch_actions)
                    entropy = distribution.entropy().mean()
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                    clipped = clipped * batch_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(outputs.values, batch_returns)
                    belief_loss = self._belief_loss(outputs, batch_obs)
                    loss = (
                        policy_loss
                        + (self.config.value_coef * value_loss)
                        + (self.config.belief_loss_coef * belief_loss)
                        - (self.config.entropy_coef * entropy)
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                metrics["loss"] += float(loss.item())
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["belief_loss"] += float(belief_loss.item())

        divisor = float(max(1, self.config.epochs * max(1, int(np.ceil(len(indices) / self.config.batch_size)))))
        return dict((key, value / divisor) for key, value in metrics.items())

    def imitation_update(self, observations: List[Mapping[str, object]], actions: List[int]) -> Dict[str, float]:
        if not observations:
            return {"imitation/loss": 0.0, "imitation/action_loss": 0.0, "imitation/belief_loss": 0.0}

        batched = batch_observations(observations, device=torch.device(self.device))
        target_actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        metrics = {"imitation/loss": 0.0, "imitation/action_loss": 0.0, "imitation/belief_loss": 0.0}
        indices = np.arange(len(observations))
        for _ in range(self.config.imitation_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.imitation_batch_size):
                batch_idx = indices[start : start + self.config.imitation_batch_size]
                batch_obs = {key: value[batch_idx] for key, value in batched.items()}
                batch_actions = target_actions[batch_idx]
                autocast_enabled = self.config.mixed_precision and self.device.startswith("cuda")
                autocast = torch.amp.autocast if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast
                with autocast("cuda", enabled=autocast_enabled) if autocast is torch.amp.autocast else autocast(enabled=autocast_enabled):
                    outputs = self.policy.forward_with_aux(batch_obs)
                    filtered_logits = masked_logits(outputs.logits, batch_obs["legal_action_mask"])
                    action_loss = F.cross_entropy(filtered_logits, batch_actions)
                    belief_loss = self._belief_loss(outputs, batch_obs)
                    loss = action_loss + (self.config.belief_loss_coef * belief_loss)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                metrics["imitation/loss"] += float(loss.item())
                metrics["imitation/action_loss"] += float(action_loss.item())
                metrics["imitation/belief_loss"] += float(belief_loss.item())

        divisor = float(max(1, self.config.imitation_epochs * max(1, int(np.ceil(len(indices) / self.config.imitation_batch_size)))))
        return {key: value / divisor for key, value in metrics.items()}

    def _returns_and_advantages(self, buffer: RolloutBuffer) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32)
        next_values = np.asarray(buffer.next_values, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        trajectory_ids = np.asarray(buffer.trajectory_ids, dtype=np.int64)

        advantages = np.zeros_like(rewards)
        for trajectory_id in np.unique(trajectory_ids):
            gae = 0.0
            indices = np.nonzero(trajectory_ids == trajectory_id)[0]
            for step in reversed(indices):
                mask = 1.0 - dones[step]
                delta = rewards[step] + (self.config.gamma * next_values[step] * mask) - values[step]
                gae = delta + (self.config.gamma * self.config.gae_lambda * mask * gae)
                advantages[step] = gae
        returns = advantages + values
        return returns, advantages

    def _belief_loss(self, outputs: PolicyForwardOutputs, batch_obs: Dict[str, Tensor]) -> Tensor:
        targets = batch_obs.get("belief_target_owner")
        target_mask = batch_obs.get("belief_target_mask")
        if targets is None or target_mask is None:
            return torch.zeros((), device=outputs.logits.device, dtype=outputs.logits.dtype)
        flat_mask = target_mask.float().reshape(-1)
        if float(flat_mask.sum().item()) <= 0.0:
            return torch.zeros((), device=outputs.logits.device, dtype=outputs.logits.dtype)
        losses = F.cross_entropy(
            outputs.belief_logits.reshape(-1, outputs.belief_logits.shape[-1]),
            targets.long().reshape(-1),
            reduction="none",
        )
        return (losses * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
