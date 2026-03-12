from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

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
    final_entropy_coef: Optional[float] = None
    bid_entropy_scale: float = 1.5
    play_entropy_scale: float = 1.0
    value_coef: float = 0.5
    belief_loss_coef: float = 0.2
    expected_trick_loss_coef: float = 0.15
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
    def __init__(self, policy: WhistPolicyNetwork, config: PPOConfig, device: str = "cpu", total_updates: int = 0):
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.initial_entropy_coef = config.entropy_coef
        self.final_entropy_coef = config.final_entropy_coef if config.final_entropy_coef is not None else config.entropy_coef
        self.entropy_coef = config.entropy_coef
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        if total_updates > 0:
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_updates,
            )
        else:
            self.scheduler = None
        scaler_enabled = config.mixed_precision and device.startswith("cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def set_anneal_progress(self, progress: float) -> None:
        clamped_progress = min(1.0, max(0.0, float(progress)))
        self.entropy_coef = self.initial_entropy_coef + (
            (self.final_entropy_coef - self.initial_entropy_coef) * clamped_progress
        )

    def select_action(self, observation: Mapping[str, object]) -> tuple[int, float, float]:
        if self.policy.training:
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
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "belief_loss": 0.0,
                "expected_trick_loss": 0.0,
            }

        self.policy.train()
        returns, advantages = self._returns_and_advantages(buffer)
        observations = batch_observations(buffer.observations, device=torch.device(self.device))
        actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False).clamp_min(1e-6))

        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "belief_loss": 0.0,
            "expected_trick_loss": 0.0,
        }
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
                    entropy_values = distribution.entropy()
                    entropy_bonus = self._entropy_bonus(entropy_values, batch_obs)
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                    clipped = clipped * batch_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(outputs.values, batch_returns)
                    belief_loss = self._belief_loss(outputs, batch_obs)
                    expected_trick_loss = self._expected_trick_loss(outputs, batch_obs)
                    loss = (
                        policy_loss
                        + (self.config.value_coef * value_loss)
                        + (self.config.belief_loss_coef * belief_loss)
                        + (self.config.expected_trick_loss_coef * expected_trick_loss)
                        - entropy_bonus
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
                metrics["entropy"] += float(entropy_values.mean().item())
                metrics["belief_loss"] += float(belief_loss.item())
                metrics["expected_trick_loss"] += float(expected_trick_loss.item())

        divisor = float(max(1, self.config.epochs * max(1, int(np.ceil(len(indices) / self.config.batch_size)))))
        if self.scheduler is not None:
            self.scheduler.step()
        return dict((key, value / divisor) for key, value in metrics.items())

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

    def _expected_trick_loss(self, outputs: PolicyForwardOutputs, batch_obs: Dict[str, Tensor]) -> Tensor:
        targets = batch_obs.get("expected_trick_target")
        phases = batch_obs.get("phase")
        hand_sizes = batch_obs.get("hand_size")
        if targets is None or phases is None or hand_sizes is None:
            return torch.zeros((), device=outputs.logits.device, dtype=outputs.logits.dtype)
        bidding_mask = phases.eq(0).float()
        if float(bidding_mask.sum().item()) <= 0.0:
            return torch.zeros((), device=outputs.logits.device, dtype=outputs.logits.dtype)
        class_ids = torch.arange(outputs.expected_trick_logits.shape[-1], device=outputs.expected_trick_logits.device).unsqueeze(0)
        valid_classes = class_ids.le(hand_sizes.long().clamp(min=0, max=8).unsqueeze(1))
        filtered_logits = outputs.expected_trick_logits.masked_fill(
            ~valid_classes,
            torch.finfo(outputs.expected_trick_logits.dtype).min,
        )
        losses = F.cross_entropy(
            filtered_logits,
            targets.long().clamp(min=0, max=8),
            reduction="none",
        )
        return (losses * bidding_mask).sum() / bidding_mask.sum().clamp_min(1.0)

    def _entropy_bonus(self, entropy_values: Tensor, batch_obs: Dict[str, Tensor]) -> Tensor:
        phases = batch_obs.get("phase")
        if phases is None:
            return entropy_values.mean() * self.entropy_coef
        scales = torch.where(
            phases.eq(0),
            torch.full_like(entropy_values, self.config.bid_entropy_scale),
            torch.full_like(entropy_values, self.config.play_entropy_scale),
        )
        return (entropy_values * scales).mean() * self.entropy_coef
