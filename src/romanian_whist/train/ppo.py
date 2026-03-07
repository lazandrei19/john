from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np
import torch
from torch import Tensor

from romanian_whist.agents.model import WhistPolicyNetwork, batch_observations, masked_logits


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: bool = True


@dataclass
class RolloutBuffer:
    observations: List[Mapping[str, object]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(
        self,
        observation: Mapping[str, object],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> int:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
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
            logits, value = self.policy(batch)
            filtered_logits = masked_logits(logits, batch["legal_action_mask"])
            distribution = torch.distributions.Categorical(logits=filtered_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if not buffer:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        returns, advantages = self._returns_and_advantages(buffer)
        observations = batch_observations(buffer.observations, device=torch.device(self.device))
        actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std().clamp_min(1e-6))

        metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
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
                    logits, values = self.policy(batch_obs)
                    filtered_logits = masked_logits(logits, batch_obs["legal_action_mask"])
                    distribution = torch.distributions.Categorical(logits=filtered_logits)
                    new_log_probs = distribution.log_prob(batch_actions)
                    entropy = distribution.entropy().mean()
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                    clipped = clipped * batch_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                    loss = (
                        policy_loss
                        + (self.config.value_coef * value_loss)
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

        divisor = float(max(1, self.config.epochs * max(1, int(np.ceil(len(indices) / self.config.batch_size)))))
        return dict((key, value / divisor) for key, value in metrics.items())

    def _returns_and_advantages(self, buffer: RolloutBuffer) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        values = np.asarray(buffer.values + [0.0], dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + (self.config.gamma * values[step + 1] * mask) - values[step]
            gae = delta + (self.config.gamma * self.config.gae_lambda * mask * gae)
            advantages[step] = gae
        returns = advantages + values[:-1]
        return returns, advantages
