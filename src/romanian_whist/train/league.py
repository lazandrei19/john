from __future__ import annotations

import copy
import json
import math
import random
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter

from romanian_whist.agents.baselines import BidPlayHeuristicAgent, RandomLegalAgent, SafeHeuristicAgent
from romanian_whist.agents.checkpoint import save_checkpoint
from romanian_whist.agents.model import PolicyAgent, PolicyNetworkConfig, WhistPolicyNetwork, batch_observations, masked_logits
from romanian_whist.env.romanian_whist import RomanianWhistEnv
from romanian_whist.rules.config import OneCardMode, WhistVariantConfig
from romanian_whist.train.curriculum import CurriculumScheduler
from romanian_whist.train.eval import TournamentRunner, average_stat_dicts, stats_to_dict
from romanian_whist.train.ppo import PPOConfig, PPOTrainer, RolloutBuffer

SCRIPTED_AGENT_TYPES = (RandomLegalAgent, SafeHeuristicAgent, BidPlayHeuristicAgent)


@dataclass
class LeagueConfig:
    total_updates: int = 100
    episodes_per_update: int = 24
    snapshot_interval: int = 5
    max_snapshots: int = 8
    latest_weight: float = 0.50
    snapshot_weight: float = 0.35
    scripted_weight: float = 0.15
    checkpoint_dir: Optional[Path] = None
    device: str = "cpu"
    seed: int = 0
    balanced_player_count_sampling: bool = True
    evaluation_matches: int = 4
    evaluation_interval: int = 1
    evaluation_player_counts: tuple[int, ...] = (3, 4, 5, 6)
    rollout_player_counts: Optional[tuple[int, ...]] = None
    rollout_one_card_modes: Optional[tuple[OneCardMode, ...]] = None
    reward_shaping_coef: float = 0.5
    imitation_episodes: int = 0
    rollout_workers: int = 1
    eval_workers: int = 1
    save_best_checkpoint: bool = True
    max_active_snapshots: int = 3
    tensorboard_log_dir: Optional[Path] = None


def _policy_state_cpu(policy: WhistPolicyNetwork) -> Dict[str, object]:
    return {
        "state_dict": {name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()},
        "config": policy.config_dict(),
    }


def _load_policy(state_bundle: Dict[str, object], device: str | torch.device = "cpu") -> WhistPolicyNetwork:
    raw_config = state_bundle.get("config", {})
    config = PolicyNetworkConfig(
        **dict(
            (field.name, raw_config.get(field.name, field.default))
            for field in fields(PolicyNetworkConfig)
        )
    )
    policy = WhistPolicyNetwork.from_config(config)
    policy.load_state_dict(state_bundle["state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def _load_cpu_policy(state_bundle: Dict[str, object]) -> WhistPolicyNetwork:
    return _load_policy(state_bundle, device="cpu")


def _policy_select_action(
    policy: WhistPolicyNetwork,
    observation: Dict[str, object],
    *,
    greedy: bool,
    device: Optional[torch.device] = None,
) -> tuple[int, float, float]:
    inference_device = device or next(policy.parameters()).device
    with torch.no_grad():
        batch = batch_observations([observation], device=inference_device)
        logits, values = policy(batch)
        filtered_logits = masked_logits(logits, batch["legal_action_mask"])
        distribution = torch.distributions.Categorical(logits=filtered_logits)
        action = torch.argmax(filtered_logits, dim=-1) if greedy else distribution.sample()
        log_prob = distribution.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(values.item())


def _policy_select_actions(
    policy: WhistPolicyNetwork,
    observations: Sequence[Dict[str, object]],
    *,
    greedy_flags: Sequence[bool],
    device: Optional[torch.device] = None,
    use_amp: bool = False,
) -> List[tuple[int, float, float]]:
    inference_device = device or next(policy.parameters()).device
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        batch = batch_observations(list(observations), device=inference_device)
        logits, values = policy(batch)
        filtered_logits = masked_logits(logits, batch["legal_action_mask"])
        distribution = torch.distributions.Categorical(logits=filtered_logits.float())
        sampled_actions = distribution.sample()
        greedy_actions = torch.argmax(filtered_logits, dim=-1)
        actions = torch.where(
            torch.as_tensor(greedy_flags, dtype=torch.bool, device=filtered_logits.device),
            greedy_actions,
            sampled_actions,
        )
        log_probs = distribution.log_prob(actions)
    return [
        (int(action.item()), float(log_prob.item()), float(value.item()))
        for action, log_prob, value in zip(actions, log_probs, values)
    ]


def _request_gpu_inference(
    message_queue: object,
    response_queue: object,
    worker_index: int,
    policy_kind: str,
    snapshot_index: int,
    observations: Sequence[Dict[str, object]],
    greedy_flags: Sequence[bool],
) -> List[tuple[int, float, float]]:
    message_queue.put(
        {
            "type": "infer",
            "worker_index": worker_index,
            "policy_kind": policy_kind,
            "snapshot_index": snapshot_index,
            "observations": list(observations),
            "greedy_flags": list(greedy_flags),
        }
    )
    return response_queue.get()


def _request_gpu_inference_multi(
    message_queue: object,
    response_queue: object,
    worker_index: int,
    groups: Sequence[tuple[str, int, Sequence[Dict[str, object]], Sequence[bool]]],
) -> List[List[tuple[int, float, float]]]:
    """Send all policy groups in a single round-trip to the GPU inference server."""
    message_queue.put(
        {
            "type": "infer_multi",
            "worker_index": worker_index,
            "groups": [
                {
                    "policy_kind": policy_kind,
                    "snapshot_index": snapshot_index,
                    "observations": list(observations),
                    "greedy_flags": list(greedy_flags),
                }
                for policy_kind, snapshot_index, observations, greedy_flags in groups
            ],
        }
    )
    return response_queue.get()


def _baseline_agent(role: str, seed: int) -> object:
    if role == "random":
        return RandomLegalAgent(seed=seed)
    if role == "safe":
        return SafeHeuristicAgent(seed=seed)
    if role == "heuristic":
        return BidPlayHeuristicAgent(seed=seed)
    raise ValueError("Unknown baseline role: {role}".format(role=role))


def _round_potential(env: RomanianWhistEnv, seat: int) -> float:
    state = env.game.round_state
    if state is None:
        return 0.0
    bid = state.bids[seat]
    if bid is None:
        return 0.0
    tricks = state.tricks_won[seat]
    hand_remaining = len(state.hands[seat])
    max_possible = tricks + hand_remaining
    over_bid = max(0, tricks - bid)
    under_bid = max(0, bid - max_possible)
    denominator = float(max(1, state.hand_size))
    pressure = abs(bid - tricks) / float(max(1, hand_remaining + 1))
    return -((over_bid + under_bid) / denominator) - (0.1 * pressure)


def _belief_targets(env: RomanianWhistEnv, observer_seat: int, observation: Dict[str, object]) -> tuple[list[int], list[float]]:
    state = env.game.round_state
    if state is None:
        return ([env.config.max_players + 1] * 52, [0.0] * 52)
    labels = [env.config.max_players + 1 for _ in range(52)]
    for seat, hand in enumerate(state.hands):
        for card_id in hand:
            labels[card_id] = seat
    for card_id in state.played_cards:
        labels[card_id] = env.config.max_players

    mask = [1.0 for _ in range(52)]
    for card_id, enabled in enumerate(observation["hand_mask"]):
        if int(enabled):
            mask[card_id] = 0.0
    for card_id, enabled in enumerate(observation["played_card_mask"]):
        if int(enabled):
            mask[card_id] = 0.0
    for public_card in observation["public_card_by_player"]:
        card_id = int(public_card)
        if card_id >= 0:
            mask[card_id] = 0.0
    return labels, mask


def _training_observation(env: RomanianWhistEnv, seat: int, observation: Dict[str, object]) -> Dict[str, object]:
    labels, mask = _belief_targets(env, seat, observation)
    enriched = dict(observation)
    enriched["belief_target_owner"] = labels
    enriched["belief_target_mask"] = mask
    return enriched


def _chunk_items(items: Sequence[dict], chunks: int) -> List[List[dict]]:
    if not items:
        return []
    chunk_size = int(math.ceil(len(items) / float(max(1, chunks))))
    return [list(items[index : index + chunk_size]) for index in range(0, len(items), chunk_size)]


def _rollout_worker(
    task: Dict[str, object],
    *,
    inference_message_queue: Optional[object] = None,
    inference_response_queue: Optional[object] = None,
    worker_index: int = 0,
) -> Dict[str, list]:
    torch.set_num_threads(1)
    rng = random.Random(int(task["seed"]))
    torch.manual_seed(int(task["seed"]))
    start_time = time.perf_counter()
    use_gpu_inference = inference_message_queue is not None and inference_response_queue is not None
    latest_policy = None if use_gpu_inference else _load_cpu_policy(task["latest_policy_state"])
    snapshot_policies = [] if use_gpu_inference else [_load_cpu_policy(state) for state in task["snapshot_policy_states"]]
    observations = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    next_values = []
    trajectory_ids = []

    episode_states = []
    for episode in task["episodes"]:
        config = WhistVariantConfig(
            players=int(episode["players"]),
            seed=int(episode["seed"]),
            one_card_modes=tuple(episode["one_card_modes"]),
        )
        env = RomanianWhistEnv(config)
        env.reset(seed=int(episode["seed"]))
        focal_seat = int(episode["focal_seat"])
        focal_agent_name = env.possible_agents[focal_seat]
        opponent_agents = []
        for spec in episode["opponent_specs"]:
            role = spec["role"]
            if role == "focal":
                opponent_agents.append({"kind": "focal"})
            elif role == "latest":
                opponent_agents.append({"kind": "latest", "greedy": bool(spec.get("greedy", True))})
            elif role == "snapshot":
                opponent_agents.append(
                    {
                        "kind": "snapshot",
                        "snapshot_index": int(spec["snapshot_index"]),
                        "greedy": bool(spec.get("greedy", True)),
                    }
                )
            else:
                opponent_agents.append(
                    {
                        "kind": "scripted",
                        "agent": _baseline_agent(role, int(spec["seed"])),
                    }
                )
        episode_states.append(
            {
                "env": env,
                "focal_seat": focal_seat,
                "focal_agent_name": focal_agent_name,
                "opponent_agents": opponent_agents,
                "last_transition_index": None,
                "trajectory_id": int(episode["trajectory_id"]),
            }
        )

    while True:
        active_states = [state for state in episode_states if not all(state["env"].terminations.values())]
        if not active_states:
            break
        policy_batches = {}  # type: Dict[tuple[str, int], List[Dict[str, object]]]
        progress_made = False
        for state in active_states:
            env = state["env"]
            acting_agent = env.agent_selection
            seat = env.agent_index(acting_agent)
            if seat == state["focal_seat"]:
                policy_observation = env.observe(acting_agent)
                observation = _training_observation(env, seat, policy_observation)
                policy_batches.setdefault(("latest", -1), []).append(
                    {
                        "state": state,
                        "observation": policy_observation,
                        "training_observation": observation,
                        "greedy": False,
                        "record_transition": True,
                    }
                )
                continue
            agent_spec = state["opponent_agents"][seat]
            if agent_spec["kind"] == "scripted":
                before_potential = _round_potential(env, state["focal_seat"])
                action = agent_spec["agent"].select_action_from_game(env.game, seat)
                outcome = env.step_outcome(action)
                after_potential = 0.0 if outcome.match_finished else _round_potential(env, state["focal_seat"])
                shaped_reward = outcome.rewards[state["focal_seat"]] + (
                    float(task.get("reward_shaping_coef", 0.0)) * (after_potential - before_potential)
                )
                if state["last_transition_index"] is not None:
                    rewards[state["last_transition_index"]] += shaped_reward
                    dones[state["last_transition_index"]] = outcome.match_finished
                progress_made = True
                continue
            observation = env.observe(acting_agent)
            if agent_spec["kind"] == "latest":
                policy_batches.setdefault(("latest", -1), []).append(
                    {
                        "state": state,
                        "observation": observation,
                        "greedy": bool(agent_spec["greedy"]),
                        "record_transition": False,
                    }
                )
            else:
                policy_batches.setdefault(("snapshot", int(agent_spec["snapshot_index"])), []).append(
                    {
                        "state": state,
                        "observation": observation,
                        "greedy": bool(agent_spec["greedy"]),
                        "record_transition": False,
                    }
                )

        if use_gpu_inference and policy_batches:
            ordered_keys = list(policy_batches.keys())
            groups = [
                (policy_kind, snapshot_index, [item["observation"] for item in policy_batches[(policy_kind, snapshot_index)]], [bool(item["greedy"]) for item in policy_batches[(policy_kind, snapshot_index)]])
                for policy_kind, snapshot_index in ordered_keys
            ]
            multi_results = _request_gpu_inference_multi(
                inference_message_queue,
                inference_response_queue,
                worker_index,
                groups,
            )
            for key, batch_results in zip(ordered_keys, multi_results):
                for item, (action, log_prob, value) in zip(policy_batches[key], batch_results):
                    state = item["state"]
                    before_potential = _round_potential(state["env"], state["focal_seat"])
                    outcome = state["env"].step_outcome(action)
                    after_potential = 0.0 if outcome.match_finished else _round_potential(state["env"], state["focal_seat"])
                    shaped_reward = outcome.rewards[state["focal_seat"]] + (
                        float(task.get("reward_shaping_coef", 0.0)) * (after_potential - before_potential)
                    )
                    if item["record_transition"]:
                        if state["last_transition_index"] is not None:
                            next_values[state["last_transition_index"]] = value
                        observations.append(item.get("training_observation", item["observation"]))
                        actions.append(action)
                        log_probs.append(log_prob)
                        values.append(value)
                        next_values.append(0.0)
                        rewards.append(shaped_reward)
                        dones.append(outcome.match_finished)
                        trajectory_ids.append(state["trajectory_id"])
                        state["last_transition_index"] = len(actions) - 1
                    elif state["last_transition_index"] is not None:
                        rewards[state["last_transition_index"]] += shaped_reward
                        dones[state["last_transition_index"]] = outcome.match_finished
                    progress_made = True
        elif policy_batches:
            for (policy_kind, snapshot_index), items in policy_batches.items():
                policy = latest_policy if policy_kind == "latest" else snapshot_policies[snapshot_index]
                batch_results = _policy_select_actions(
                    policy,
                    [item["observation"] for item in items],
                    greedy_flags=[bool(item["greedy"]) for item in items],
                )
                for item, (action, log_prob, value) in zip(items, batch_results):
                    state = item["state"]
                    before_potential = _round_potential(state["env"], state["focal_seat"])
                    outcome = state["env"].step_outcome(action)
                    after_potential = 0.0 if outcome.match_finished else _round_potential(state["env"], state["focal_seat"])
                    shaped_reward = outcome.rewards[state["focal_seat"]] + (
                        float(task.get("reward_shaping_coef", 0.0)) * (after_potential - before_potential)
                    )
                    if item["record_transition"]:
                        if state["last_transition_index"] is not None:
                            next_values[state["last_transition_index"]] = value
                        observations.append(item.get("training_observation", item["observation"]))
                        actions.append(action)
                        log_probs.append(log_prob)
                        values.append(value)
                        next_values.append(0.0)
                        rewards.append(shaped_reward)
                        dones.append(outcome.match_finished)
                        trajectory_ids.append(state["trajectory_id"])
                        state["last_transition_index"] = len(actions) - 1
                    elif state["last_transition_index"] is not None:
                        rewards[state["last_transition_index"]] += shaped_reward
                        dones[state["last_transition_index"]] = outcome.match_finished
                    progress_made = True

        if not progress_made:
            raise RuntimeError("Rollout worker made no progress while active environments remain.")

    return {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "values": values,
        "next_values": next_values,
        "rewards": rewards,
        "dones": dones,
        "trajectory_ids": trajectory_ids,
        "episodes": len(task["episodes"]),
        "elapsed_sec": time.perf_counter() - start_time,
    }


def _gpu_inference_worker_loop(
    message_queue: object,
    control_result_queue: object,
    response_queues: Sequence[object],
    device_name: str,
) -> None:
    torch.set_num_threads(1)
    device = torch.device(device_name)
    use_amp = device.type == "cuda"
    use_compile = device.type == "cuda" and hasattr(torch, "compile")
    latest_policy = None  # type: Optional[WhistPolicyNetwork]
    snapshot_policies = []  # type: List[WhistPolicyNetwork]
    compiled_cache = {}  # type: Dict[int, WhistPolicyNetwork]
    pending_messages = []  # type: List[Dict[str, object]]

    def _maybe_compile(policy: WhistPolicyNetwork) -> WhistPolicyNetwork:
        if not use_compile:
            return policy
        key = id(policy)
        if key not in compiled_cache:
            compiled_cache[key] = torch.compile(policy)  # type: ignore[attr-defined]
        return compiled_cache[key]

    while True:
        if pending_messages:
            message = pending_messages.pop(0)
        else:
            message = message_queue.get()
        message_type = str(message["type"])

        if message_type == "close":
            break

        if message_type == "update_models":
            try:
                latest_policy = _load_policy(message["latest_policy_state"], device=device)
                compiled_cache.clear()
                snapshot_states = message.get("snapshot_policy_states")
                if snapshot_states is not None:
                    snapshot_policies = [_load_policy(state, device=device) for state in snapshot_states]
                control_result_queue.put({"ok": True})
            except Exception as exc:  # pragma: no cover - exercised via process boundary
                control_result_queue.put({"ok": False, "error": repr(exc)})
            continue

        if message_type == "infer_multi":
            if latest_policy is None:
                response_queues[int(message["worker_index"])].put(RuntimeError("GPU inference server has no loaded policy."))
                continue
            groups = list(message["groups"])
            group_results = []  # type: List[List[tuple[int, float, float]]]
            # Coalesce all groups by policy to minimize GPU forward passes
            policy_batches = {}  # type: Dict[tuple[str, int], List[tuple[int, int, int]]]
            all_observations = []  # type: List[Dict[str, object]]
            all_greedy_flags = []  # type: List[bool]
            for group_index, group in enumerate(groups):
                key = (str(group["policy_kind"]), int(group["snapshot_index"]))
                obs = list(group["observations"])
                flags = [bool(f) for f in group["greedy_flags"]]
                start = len(all_observations)
                all_observations.extend(obs)
                all_greedy_flags.extend(flags)
                policy_batches.setdefault(key, []).append((group_index, start, len(obs)))
            # Run one forward pass per distinct policy
            all_results = [None] * len(all_observations)  # type: ignore[list-item]
            for (policy_kind, snapshot_index), entries in policy_batches.items():
                raw_policy = latest_policy if policy_kind == "latest" else snapshot_policies[snapshot_index]
                policy = _maybe_compile(raw_policy)
                indices = []
                for _, start, count in entries:
                    indices.extend(range(start, start + count))
                batch_obs = [all_observations[i] for i in indices]
                batch_flags = [all_greedy_flags[i] for i in indices]
                batch_results = _policy_select_actions(policy, batch_obs, greedy_flags=batch_flags, device=device, use_amp=use_amp)
                for idx, result in zip(indices, batch_results):
                    all_results[idx] = result
            # Repackage per-group results
            for group_index, group in enumerate(groups):
                key = (str(group["policy_kind"]), int(group["snapshot_index"]))
                obs = list(group["observations"])
                # Find this group's slice
                for g_idx, start, count in policy_batches[key]:
                    if g_idx == group_index:
                        group_results.append(all_results[start : start + count])  # type: ignore[arg-type]
                        break
            response_queues[int(message["worker_index"])].put(group_results)
            continue

        if message_type != "infer":
            continue

        if latest_policy is None:
            response_queues[int(message["worker_index"])].put(RuntimeError("GPU inference server has no loaded policy."))
            continue

        pending_requests = [message]
        while True:
            try:
                extra = message_queue.get_nowait()
            except Empty:
                break
            if str(extra["type"]) == "infer":
                pending_requests.append(extra)
            else:
                pending_messages.append(extra)

        grouped = {}  # type: Dict[tuple[str, int], List[tuple[int, Dict[str, object]]]]
        for request_index, request in enumerate(pending_requests):
            key = (str(request["policy_kind"]), int(request["snapshot_index"]))
            grouped.setdefault(key, []).append((request_index, request))

        request_results = [None] * len(pending_requests)  # type: ignore[list-item]
        for (policy_kind, snapshot_index), grouped_requests in grouped.items():
            raw_policy = latest_policy if policy_kind == "latest" else snapshot_policies[snapshot_index]
            policy = _maybe_compile(raw_policy)
            flat_observations = []
            flat_greedy_flags = []
            counts = []
            for _, request in grouped_requests:
                observations = list(request["observations"])
                greedy_flags = list(request["greedy_flags"])
                flat_observations.extend(observations)
                flat_greedy_flags.extend(bool(flag) for flag in greedy_flags)
                counts.append(len(observations))
            flat_results = _policy_select_actions(
                policy,
                flat_observations,
                greedy_flags=flat_greedy_flags,
                device=device,
                use_amp=use_amp,
            )
            offset = 0
            for (request_index, _), count in zip(grouped_requests, counts):
                request_results[request_index] = flat_results[offset : offset + count]
                offset += count

        for request, result in zip(pending_requests, request_results):
            response_queues[int(request["worker_index"])].put(result)


def _evaluation_worker(task: Dict[str, object]) -> Dict[str, object]:
    torch.set_num_threads(1)
    start_time = time.perf_counter()
    latest_policy = _load_cpu_policy(task["latest_policy_state"])
    players = int(task["players"])
    seed = int(task["seed"])
    matches = list(task["matches"])
    config = WhistVariantConfig(players=players, seed=seed)
    participants = [("policy_0", PolicyAgent(latest_policy, device="cpu", greedy=True))]
    family_names = ("random", "safe", "heuristic")
    for seat_index in range(1, players):
        family_name = family_names[(seat_index - 1) % len(family_names)]
        label = "{family}_{index}".format(family=family_name, index=seat_index - 1)
        participants.append((label, _baseline_agent(family_name, seed + 100 + players + seat_index)))

    results = []
    for match_index in matches:
        env = RomanianWhistEnv(config)
        env.reset(seed=seed + int(match_index))
        rotation = int(match_index) % len(participants)
        seat_order = participants[rotation:] + participants[:rotation]
        while not all(env.terminations.values()):
            acting_agent = env.agent_selection
            seat = env.agent_index(acting_agent)
            _, agent = seat_order[seat]
            if isinstance(agent, SCRIPTED_AGENT_TYPES):
                action = agent.select_action_from_game(env.game, seat)
            else:
                action = agent.select_action(env.observe(acting_agent))
            env.step(action, include_observation=False)
        replay = env.serialize_replay()
        results.append(
            {
                "match_index": int(match_index),
                "seat_labels": [name for name, _ in seat_order],
                "final_scores": list(replay["scores"]),
                "events": replay["events"],
            }
        )
    return {
        "players": players,
        "matches": results,
        "matches_count": len(matches),
        "elapsed_sec": time.perf_counter() - start_time,
    }


def _rollout_worker_loop(
    task_queue: object,
    result_queue: object,
    worker_index: int,
    inference_message_queue: Optional[object] = None,
    inference_response_queue: Optional[object] = None,
) -> None:
    torch.set_num_threads(1)
    while True:
        item = task_queue.get()
        if item is None:
            break
        task_id, task = item
        result_queue.put(
            (
                task_id,
                _rollout_worker(
                    task,
                    inference_message_queue=inference_message_queue,
                    inference_response_queue=inference_response_queue,
                    worker_index=worker_index,
                ),
            )
        )


class PersistentGpuInferenceServer:
    def __init__(self, workers: int, device: str):
        self.workers = workers
        self.device = device
        ctx = mp.get_context("spawn")
        self.message_queue = ctx.Queue()
        self.control_result_queue = ctx.Queue()
        self.response_queues = [ctx.Queue() for _ in range(workers)]
        self.process = ctx.Process(
            target=_gpu_inference_worker_loop,
            args=(self.message_queue, self.control_result_queue, self.response_queues, device),
            daemon=True,
        )
        self.process.start()

    def update_models(
        self,
        latest_policy_state: Dict[str, object],
        snapshot_policy_states: Optional[Sequence[Dict[str, object]]],
    ) -> None:
        self.message_queue.put(
            {
                "type": "update_models",
                "latest_policy_state": latest_policy_state,
                "snapshot_policy_states": list(snapshot_policy_states) if snapshot_policy_states is not None else None,
            }
        )
        result = self.control_result_queue.get()
        if not bool(result.get("ok", False)):
            raise RuntimeError("GPU inference server failed to load policies: {error}".format(error=result.get("error", "unknown")))

    def response_queue(self, worker_index: int) -> object:
        return self.response_queues[worker_index]

    def close(self) -> None:
        self.message_queue.put({"type": "close"})
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)


class PersistentRolloutPool:
    def __init__(
        self,
        workers: int,
        *,
        inference_message_queue: Optional[object] = None,
        inference_response_queues: Optional[Sequence[object]] = None,
    ):
        self.workers = workers
        ctx = mp.get_context("spawn")
        self.task_queues = [ctx.Queue() for _ in range(workers)]
        self.result_queue = ctx.Queue()
        self.processes = [
            ctx.Process(
                target=_rollout_worker_loop,
                args=(
                    task_queue,
                    self.result_queue,
                    index,
                    inference_message_queue,
                    None if inference_response_queues is None else inference_response_queues[index],
                ),
                daemon=True,
            )
            for index, task_queue in enumerate(self.task_queues)
        ]
        for process in self.processes:
            process.start()

    def map(self, tasks: Sequence[Dict[str, object]]) -> List[Dict[str, list]]:
        results = [None] * len(tasks)  # type: ignore[list-item]
        for task_id, task in enumerate(tasks):
            queue = self.task_queues[task_id % len(self.task_queues)]
            queue.put((task_id, task))
        for _ in range(len(tasks)):
            task_id, result = self.result_queue.get()
            results[task_id] = result
        return results  # type: ignore[return-value]

    def close(self) -> None:
        for task_queue in self.task_queues:
            task_queue.put(None)
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)


@dataclass
class LeagueTrainer:
    variant_config: WhistVariantConfig
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    league_config: LeagueConfig = field(default_factory=LeagueConfig)
    policy_config: PolicyNetworkConfig = field(default_factory=PolicyNetworkConfig)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.league_config.seed)
        self.policy = WhistPolicyNetwork.from_config(self.policy_config)
        self.ppo = PPOTrainer(self.policy, self.ppo_config, device=self.league_config.device)
        self.curriculum = CurriculumScheduler(self.league_config.total_updates)
        self.snapshots = []  # type: List[WhistPolicyNetwork]
        self.best_snapshot = None  # type: Optional[WhistPolicyNetwork]
        self.best_selection_score = float("-inf")
        self.rollout_pool = None  # type: Optional[PersistentRolloutPool]
        self.gpu_inference_server = None  # type: Optional[PersistentGpuInferenceServer]
        self.gpu_inference_snapshot_keys = ()  # type: tuple[int, ...]
        self.last_rollout_stats = {}  # type: Dict[str, float]
        self.last_eval_stats = {}  # type: Dict[str, float]
        self.writer = (
            SummaryWriter(log_dir=str(self.league_config.tensorboard_log_dir))
            if self.league_config.tensorboard_log_dir is not None
            else None
        )
        self.scripted_pool = [
            RandomLegalAgent(seed=self.league_config.seed),
            SafeHeuristicAgent(seed=self.league_config.seed + 1),
            BidPlayHeuristicAgent(seed=self.league_config.seed + 2),
        ]

    @property
    def _all_snapshots(self) -> List[WhistPolicyNetwork]:
        if self.best_snapshot is None:
            return self.snapshots
        return self.snapshots + [self.best_snapshot]

    def train(self, updates: Optional[int] = None, start_update: int = 0) -> List[Dict[str, float]]:
        total_updates = updates or self.league_config.total_updates
        history = []
        try:
            if start_update == 0 and self.league_config.imitation_episodes > 0:
                imitation_metrics = self.pretrain_imitation(self.league_config.imitation_episodes)
                if self.writer is not None:
                    for key, value in imitation_metrics.items():
                        self.writer.add_scalar("pretrain/{key}".format(key=key), float(value), 0)
                    self.writer.flush()
            for local_update_index in range(total_updates):
                update_index = start_update + local_update_index + 1
                stage = self.curriculum.stage_for_update(update_index - 1)
                rollout_player_counts = self._rollout_player_counts(stage.player_counts)
                rollout_one_card_modes = self._rollout_one_card_modes(stage.one_card_modes)
                update_start = time.perf_counter()
                rollout_start = time.perf_counter()
                buffer = self.collect_rollouts(
                    rollout_player_counts,
                    rollout_one_card_modes,
                    self.league_config.episodes_per_update,
                )
                rollout_time = time.perf_counter() - rollout_start
                ppo_start = time.perf_counter()
                metrics = self.ppo.update(buffer)
                ppo_time = time.perf_counter() - ppo_start
                metrics["stage"] = stage.name
                metrics["rollout_player_counts"] = float(len(rollout_player_counts))
                metrics["rollout_one_card_modes"] = float(len(rollout_one_card_modes))
                metrics["transitions"] = float(len(buffer))
                metrics["timing/rollout_sec"] = rollout_time
                metrics["timing/ppo_sec"] = ppo_time
                metrics["timing/transitions_per_sec"] = (
                    float(len(buffer)) / rollout_time if rollout_time > 0.0 else 0.0
                )
                metrics.update(self.last_rollout_stats)
                evaluation = None
                should_evaluate = (
                    update_index % self.league_config.evaluation_interval == 0
                    or update_index == start_update + total_updates
                )
                if should_evaluate:
                    eval_start = time.perf_counter()
                    evaluation = self.evaluate(matches=self.league_config.evaluation_matches)
                    eval_time = time.perf_counter() - eval_start
                    metrics["selection_score"] = self.selection_score(evaluation)
                    metrics["timing/eval_sec"] = eval_time
                    metrics.update(self.last_eval_stats)
                else:
                    metrics["timing/eval_sec"] = 0.0
                history.append(metrics)
                if update_index % self.league_config.snapshot_interval == 0:
                    self._promote_snapshot()
                checkpoint_start = time.perf_counter()
                self._save_checkpoint(update_index, metrics, evaluation)
                self._save_best_checkpoint(update_index, metrics, evaluation)
                checkpoint_time = time.perf_counter() - checkpoint_start
                metrics["timing/checkpoint_sec"] = checkpoint_time
                metrics["timing/update_sec"] = time.perf_counter() - update_start
                self._log_update(update_index, metrics, evaluation)
        finally:
            if self.rollout_pool is not None:
                self.rollout_pool.close()
                self.rollout_pool = None
            if self.gpu_inference_server is not None:
                self.gpu_inference_server.close()
                self.gpu_inference_server = None
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        return history

    def _collect_rollouts_parallel_gpu_inference(
        self,
        player_counts: tuple[int, ...],
        one_card_modes: tuple[object, ...],
        episodes: int,
    ) -> RolloutBuffer:
        buffer = RolloutBuffer()
        player_count_schedule = self._player_count_schedule(player_counts, episodes)
        worker_count = min(max(1, self.league_config.rollout_workers), len(player_count_schedule))
        active_snapshot_indices = self._select_active_snapshot_indices()
        task_episodes = []
        for episode_index, players in enumerate(player_count_schedule):
            focal_seat = episode_index % players
            task_episodes.append(
                {
                    "players": players,
                    "seed": self.rng.randint(0, 1_000_000),
                    "focal_seat": focal_seat,
                    "trajectory_id": episode_index,
                    "one_card_modes": list(one_card_modes),
                    "opponent_specs": self._sample_opponent_specs(players, focal_seat, active_snapshot_indices),
                }
            )

        if self.rollout_pool is not None and self.rollout_pool.workers != worker_count:
            self.rollout_pool.close()
            self.rollout_pool = None
        if self.gpu_inference_server is not None and self.gpu_inference_server.workers != worker_count:
            self.gpu_inference_server.close()
            self.gpu_inference_server = None
            self.gpu_inference_snapshot_keys = ()

        if self.gpu_inference_server is None:
            self.gpu_inference_server = PersistentGpuInferenceServer(worker_count, self.league_config.device)

        latest_policy_state = _policy_state_cpu(self.policy)
        all_snapshots = self._all_snapshots
        snapshot_keys = tuple(id(snapshot) for snapshot in all_snapshots)
        snapshot_policy_states = None
        if snapshot_keys != self.gpu_inference_snapshot_keys:
            snapshot_policy_states = [_policy_state_cpu(snapshot) for snapshot in all_snapshots]
            self.gpu_inference_snapshot_keys = snapshot_keys
        self.gpu_inference_server.update_models(latest_policy_state, snapshot_policy_states)

        if self.rollout_pool is None:
            self.rollout_pool = PersistentRolloutPool(
                worker_count,
                inference_message_queue=self.gpu_inference_server.message_queue,
                inference_response_queues=self.gpu_inference_server.response_queues,
            )

        tasks = [
            {
                "seed": self.rng.randint(0, 1_000_000),
                "episodes": chunk,
                "latest_policy_state": latest_policy_state,
                "snapshot_policy_states": [],
                "reward_shaping_coef": self.league_config.reward_shaping_coef,
            }
            for chunk in _chunk_items(task_episodes, worker_count)
        ]

        results = self.rollout_pool.map(tasks)
        worker_elapsed = []
        for result in results:
            buffer.observations.extend(result["observations"])
            buffer.actions.extend(result["actions"])
            buffer.log_probs.extend(result["log_probs"])
            buffer.values.extend(result["values"])
            buffer.next_values.extend(result["next_values"])
            buffer.rewards.extend(result["rewards"])
            buffer.dones.extend(result["dones"])
            buffer.trajectory_ids.extend(result["trajectory_ids"])
            worker_elapsed.append(float(result["elapsed_sec"]))
        self.last_rollout_stats = {
            "rollout/episodes": float(episodes),
            "rollout/workers_used": float(worker_count),
            "rollout/gpu_inference_server": 1.0,
            "rollout/worker_task_sec_mean": (
                sum(worker_elapsed) / float(len(worker_elapsed)) if worker_elapsed else 0.0
            ),
            "rollout/worker_task_sec_max": max(worker_elapsed) if worker_elapsed else 0.0,
        }
        return buffer

    def collect_rollouts(
        self,
        player_counts: tuple[int, ...],
        one_card_modes: tuple[object, ...],
        episodes: int,
    ) -> RolloutBuffer:
        if self.league_config.rollout_workers > 1:
            if self.league_config.device.startswith("cuda"):
                return self._collect_rollouts_parallel_gpu_inference(player_counts, one_card_modes, episodes)
            return self._collect_rollouts_parallel(player_counts, one_card_modes, episodes)
        start_time = time.perf_counter()
        buffer = RolloutBuffer()
        player_count_schedule = self._player_count_schedule(player_counts, episodes)
        active_snapshot_indices = self._select_active_snapshot_indices()
        for episode_index, players in enumerate(player_count_schedule):
            config = self.variant_config.replace(players=players, one_card_modes=tuple(one_card_modes))
            env = RomanianWhistEnv(config)
            env.reset(seed=self.rng.randint(0, 1_000_000))
            focal_seat = episode_index % players
            focal_agent_name = env.possible_agents[focal_seat]
            opponent_agents = self._sample_opponents(players, focal_seat, active_snapshot_indices)
            last_transition_index = None
            while not all(env.terminations.values()):
                acting_agent = env.agent_selection
                seat = env.agent_index(acting_agent)
                scripted_agent = opponent_agents[seat] if seat != focal_seat and isinstance(opponent_agents[seat], SCRIPTED_AGENT_TYPES) else None
                observation = env.observe(acting_agent) if scripted_agent is None else None
                if seat == focal_seat:
                    assert observation is not None
                    training_observation = _training_observation(env, seat, observation)
                    action, log_prob, value = self.ppo.select_action(observation)
                    if last_transition_index is not None:
                        buffer.next_values[last_transition_index] = value
                    before_potential = _round_potential(env, focal_seat)
                    outcome = env.step_outcome(action)
                    after_potential = 0.0 if outcome.match_finished else _round_potential(env, focal_seat)
                    reward = outcome.rewards[focal_seat] + (
                        self.league_config.reward_shaping_coef * (after_potential - before_potential)
                    )
                    done = outcome.match_finished
                    last_transition_index = buffer.add(
                        observation=training_observation,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        done=done,
                        trajectory_id=episode_index,
                    )
                else:
                    before_potential = _round_potential(env, focal_seat)
                    action = scripted_agent.select_action_from_game(env.game, seat) if scripted_agent is not None else opponent_agents[seat].select_action(observation)
                    outcome = env.step_outcome(action)
                    after_potential = 0.0 if outcome.match_finished else _round_potential(env, focal_seat)
                    shaped_reward = outcome.rewards[focal_seat] + (
                        self.league_config.reward_shaping_coef * (after_potential - before_potential)
                    )
                    if last_transition_index is not None:
                        buffer.rewards[last_transition_index] += shaped_reward
                        buffer.dones[last_transition_index] = outcome.match_finished
        elapsed = time.perf_counter() - start_time
        self.last_rollout_stats = {
            "rollout/episodes": float(episodes),
            "rollout/workers_used": 1.0,
            "rollout/worker_task_sec_mean": elapsed,
            "rollout/worker_task_sec_max": elapsed,
        }
        return buffer

    def evaluate(self, matches: int = 8) -> Dict[str, object]:
        if self.league_config.eval_workers > 1:
            return self._evaluate_parallel(matches=matches)
        start_time = time.perf_counter()
        per_player_count = {}  # type: Dict[str, Dict[str, Dict[str, float]]]
        stat_dicts = []
        for players in self.league_config.evaluation_player_counts:
            runner = TournamentRunner(self.variant_config.replace(players=players))
            participants, label_groups = self._evaluation_participants(players)
            raw_stats = stats_to_dict(runner.run(participants, matches=matches, seed=self.league_config.seed + players))
            stats = self._aggregate_label_groups(raw_stats, label_groups)
            per_player_count[str(players)] = stats
            stat_dicts.append(stats)
        overall = average_stat_dicts(stat_dicts)
        self.last_eval_stats = {
            "eval/matches": float(matches * len(self.league_config.evaluation_player_counts)),
            "eval/workers_used": 1.0,
            "eval/worker_task_sec_mean": time.perf_counter() - start_time,
            "eval/worker_task_sec_max": time.perf_counter() - start_time,
        }
        return {
            "overall": overall,
            "by_player_count": per_player_count,
        }

    def _collect_rollouts_parallel(
        self,
        player_counts: tuple[int, ...],
        one_card_modes: tuple[object, ...],
        episodes: int,
    ) -> RolloutBuffer:
        buffer = RolloutBuffer()
        player_count_schedule = self._player_count_schedule(player_counts, episodes)
        worker_count = min(max(1, self.league_config.rollout_workers), len(player_count_schedule))
        active_snapshot_indices = self._select_active_snapshot_indices()
        task_episodes = []
        for episode_index, players in enumerate(player_count_schedule):
            focal_seat = episode_index % players
            task_episodes.append(
                {
                    "players": players,
                    "seed": self.rng.randint(0, 1_000_000),
                    "focal_seat": focal_seat,
                    "trajectory_id": episode_index,
                    "one_card_modes": list(one_card_modes),
                    "opponent_specs": self._sample_opponent_specs(players, focal_seat, active_snapshot_indices),
                }
            )
        latest_policy_state = _policy_state_cpu(self.policy)
        snapshot_policy_states = [_policy_state_cpu(snapshot) for snapshot in self._all_snapshots]
        tasks = [
            {
                    "seed": self.rng.randint(0, 1_000_000),
                    "episodes": chunk,
                    "latest_policy_state": latest_policy_state,
                    "snapshot_policy_states": snapshot_policy_states,
                    "reward_shaping_coef": self.league_config.reward_shaping_coef,
                }
                for chunk in _chunk_items(task_episodes, worker_count)
        ]
        if self.rollout_pool is not None and self.rollout_pool.workers != worker_count:
            self.rollout_pool.close()
            self.rollout_pool = None
        if self.rollout_pool is None:
            self.rollout_pool = PersistentRolloutPool(worker_count)
        results = self.rollout_pool.map(tasks)
        worker_elapsed = []
        for result in results:
            buffer.observations.extend(result["observations"])
            buffer.actions.extend(result["actions"])
            buffer.log_probs.extend(result["log_probs"])
            buffer.values.extend(result["values"])
            buffer.next_values.extend(result["next_values"])
            buffer.rewards.extend(result["rewards"])
            buffer.dones.extend(result["dones"])
            buffer.trajectory_ids.extend(result["trajectory_ids"])
            worker_elapsed.append(float(result["elapsed_sec"]))
        self.last_rollout_stats = {
            "rollout/episodes": float(episodes),
            "rollout/workers_used": float(worker_count),
            "rollout/worker_task_sec_mean": (
                sum(worker_elapsed) / float(len(worker_elapsed)) if worker_elapsed else 0.0
            ),
            "rollout/worker_task_sec_max": max(worker_elapsed) if worker_elapsed else 0.0,
        }
        return buffer

    def pretrain_imitation(self, episodes: int) -> Dict[str, float]:
        player_counts = self._rollout_player_counts(self.curriculum.stage_for_update(0).player_counts)
        one_card_modes = self._rollout_one_card_modes(self.variant_config.one_card_modes)
        observations = []
        actions = []
        teacher_roles = ("safe", "heuristic")
        for episode_index, players in enumerate(self._player_count_schedule(player_counts, episodes)):
            teacher_role = teacher_roles[episode_index % len(teacher_roles)]
            config = self.variant_config.replace(players=players, one_card_modes=tuple(one_card_modes))
            env = RomanianWhistEnv(config)
            env.reset(seed=self.rng.randint(0, 1_000_000))
            teachers = [_baseline_agent(teacher_role, self.rng.randint(0, 1_000_000)) for _ in range(players)]
            while not all(env.terminations.values()):
                acting_agent = env.agent_selection
                seat = env.agent_index(acting_agent)
                observation = env.observe(acting_agent)
                observations.append(_training_observation(env, seat, observation))
                teacher_action = teachers[seat].select_action_from_game(env.game, seat)
                actions.append(teacher_action)
                env.step(teacher_action, include_observation=False)
        return self.ppo.imitation_update(observations, actions)

    def _evaluate_parallel(self, matches: int = 8) -> Dict[str, object]:
        per_player_count = {}  # type: Dict[str, Dict[str, Dict[str, float]]]
        stat_dicts = []
        tasks = []
        worker_cap = max(1, self.league_config.eval_workers)
        latest_policy_state = _policy_state_cpu(self.policy)
        for players in self.league_config.evaluation_player_counts:
            match_specs = [{"match_index": match_index} for match_index in range(matches)]
            for chunk in _chunk_items(match_specs, min(worker_cap, matches)):
                tasks.append(
                    {
                        "players": players,
                        "seed": self.league_config.seed + players,
                        "matches": [item["match_index"] for item in chunk],
                        "latest_policy_state": latest_policy_state,
                    }
                )
        grouped_results = {}  # type: Dict[int, List[Dict[str, object]]]
        worker_elapsed = []
        with ProcessPoolExecutor(max_workers=min(worker_cap, len(tasks))) as executor:
            for result in executor.map(_evaluation_worker, tasks):
                grouped_results.setdefault(int(result["players"]), []).extend(result["matches"])
                worker_elapsed.append(float(result["elapsed_sec"]))
        for players in self.league_config.evaluation_player_counts:
            runner = TournamentRunner(self.variant_config.replace(players=players))
            participants, label_groups = self._evaluation_participants(players)
            stats = self._aggregate_label_groups(
                stats_to_dict(runner.summarize_match_results(grouped_results.get(players, []), participants)),
                label_groups,
            )
            per_player_count[str(players)] = stats
            stat_dicts.append(stats)
        overall = average_stat_dicts(stat_dicts)
        self.last_eval_stats = {
            "eval/matches": float(matches * len(self.league_config.evaluation_player_counts)),
            "eval/workers_used": float(min(worker_cap, len(tasks))),
            "eval/worker_task_sec_mean": (
                sum(worker_elapsed) / float(len(worker_elapsed)) if worker_elapsed else 0.0
            ),
            "eval/worker_task_sec_max": max(worker_elapsed) if worker_elapsed else 0.0,
        }
        return {
            "overall": overall,
            "by_player_count": per_player_count,
        }

    @staticmethod
    def selection_score(evaluation: Dict[str, object]) -> float:
        overall = evaluation["overall"]
        return float(overall["average_scores"]["policy"])

    def _player_count_schedule(self, player_counts: Sequence[int], episodes: int) -> List[int]:
        counts = list(dict.fromkeys(player_counts))
        if not counts:
            raise ValueError("At least one player count must be provided.")
        if not self.league_config.balanced_player_count_sampling or len(counts) == 1:
            return [self.rng.choice(counts) for _ in range(episodes)]
        self.rng.shuffle(counts)
        return [counts[index % len(counts)] for index in range(episodes)]

    def _rollout_player_counts(self, stage_player_counts: Sequence[int]) -> tuple[int, ...]:
        if self.league_config.rollout_player_counts is not None:
            return tuple(self.league_config.rollout_player_counts)
        return tuple(stage_player_counts)

    def _rollout_one_card_modes(self, stage_one_card_modes: Sequence[OneCardMode]) -> tuple[OneCardMode, ...]:
        if self.league_config.rollout_one_card_modes is not None:
            return tuple(self.league_config.rollout_one_card_modes)
        return tuple(stage_one_card_modes)

    def _evaluation_participants(self, players: int) -> tuple[List[tuple[str, object]], Dict[str, str]]:
        participants = [
            ("policy_0", PolicyAgent(self.policy, device=self.league_config.device, greedy=True)),
        ]
        label_groups = {"policy_0": "policy"}
        family_factories = [
            ("random", lambda offset: RandomLegalAgent(seed=self.league_config.seed + 100 + offset)),
            ("safe", lambda offset: SafeHeuristicAgent(seed=self.league_config.seed + 200 + offset)),
            ("heuristic", lambda offset: BidPlayHeuristicAgent(seed=self.league_config.seed + 300 + offset)),
        ]
        for seat_index in range(1, players):
            family_name, factory = family_factories[(seat_index - 1) % len(family_factories)]
            seat_label = "{family}_{index}".format(family=family_name, index=seat_index - 1)
            participants.append((seat_label, factory(seat_index)))
            label_groups[seat_label] = family_name
        return participants, label_groups

    @staticmethod
    def _aggregate_label_groups(
        raw_stats: Dict[str, Dict[str, float]],
        label_groups: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        families = ("policy", "random", "safe", "heuristic")
        aggregated = {}  # type: Dict[str, Dict[str, float]]
        for metric_name, values in raw_stats.items():
            aggregated[metric_name] = {}
            for family in families:
                family_values = [metric_value for label, metric_value in values.items() if label_groups[label] == family]
                aggregated[metric_name][family] = (
                    sum(family_values) / float(len(family_values)) if family_values else 0.0
                )
        return aggregated

    def _sample_opponents(self, players: int, focal_seat: int, active_snapshot_indices: Optional[List[int]] = None) -> List[object]:
        all_snapshots = self._all_snapshots
        available_indices = active_snapshot_indices if active_snapshot_indices is not None else (
            list(range(len(all_snapshots))) if all_snapshots else []
        )
        pool = []
        for seat in range(players):
            if seat == focal_seat:
                pool.append(PolicyAgent(self.policy, device=self.league_config.device, greedy=False))
                continue
            roll = self.rng.random()
            if available_indices and roll < self.league_config.snapshot_weight:
                snapshot = all_snapshots[self.rng.choice(available_indices)]
                pool.append(PolicyAgent(snapshot, device=self.league_config.device, greedy=True))
            elif roll < self.league_config.snapshot_weight + self.league_config.scripted_weight:
                pool.append(copy.deepcopy(self.rng.choice(self.scripted_pool)))
            else:
                pool.append(PolicyAgent(self.policy, device=self.league_config.device, greedy=True))
        return pool

    def _select_active_snapshot_indices(self) -> List[int]:
        all_snapshots = self._all_snapshots
        if not all_snapshots:
            return []
        max_active = self.league_config.max_active_snapshots
        if len(all_snapshots) <= max_active:
            return list(range(len(all_snapshots)))
        return self.rng.sample(range(len(all_snapshots)), max_active)

    def _sample_opponent_specs(
        self, players: int, focal_seat: int, active_snapshot_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, object]]:
        specs = []
        for seat in range(players):
            if seat == focal_seat:
                specs.append({"role": "focal"})
                continue
            roll = self.rng.random()
            available_snapshots = active_snapshot_indices if active_snapshot_indices is not None else (
                list(range(len(self._all_snapshots))) if self._all_snapshots else []
            )
            if available_snapshots and roll < self.league_config.snapshot_weight:
                snapshot_index = self.rng.choice(available_snapshots)
                specs.append({"role": "snapshot", "snapshot_index": snapshot_index, "greedy": True})
            elif roll < self.league_config.snapshot_weight + self.league_config.scripted_weight:
                baseline_role = self.rng.choice(("random", "safe", "heuristic"))
                specs.append({"role": baseline_role, "seed": self.rng.randint(0, 1_000_000)})
            else:
                specs.append({"role": "latest", "greedy": True})
        return specs

    def _promote_snapshot(self) -> None:
        snapshot = copy.deepcopy(self.policy).cpu()
        snapshot.eval()
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.league_config.max_snapshots:
            self.snapshots = self.snapshots[-self.league_config.max_snapshots :]

    def _save_checkpoint(self, update_index: int, metrics: Dict[str, float], evaluation: Optional[Dict[str, object]]) -> None:
        if self.league_config.checkpoint_dir is None:
            return
        path = self.league_config.checkpoint_dir / "update-{index:04d}.pt".format(index=update_index)
        save_checkpoint(
            path,
            self.policy,
            optimizer=self.ppo.optimizer,
            metadata={"metrics": metrics, "evaluation": evaluation, "update": update_index},
        )
        if evaluation is not None:
            report_path = self.league_config.checkpoint_dir / "update-{index:04d}.eval.json".format(index=update_index)
            report_path.write_text(json.dumps(evaluation, indent=2))

    def _save_best_checkpoint(self, update_index: int, metrics: Dict[str, float], evaluation: Optional[Dict[str, object]]) -> None:
        if self.league_config.checkpoint_dir is None or not self.league_config.save_best_checkpoint or evaluation is None:
            return
        selection_score = self.selection_score(evaluation)
        if selection_score <= self.best_selection_score:
            return
        self.best_selection_score = selection_score
        self.best_snapshot = copy.deepcopy(self.policy).cpu()
        self.best_snapshot.eval()
        path = self.league_config.checkpoint_dir / "best.pt"
        save_checkpoint(
            path,
            self.policy,
            optimizer=self.ppo.optimizer,
            metadata={"metrics": metrics, "evaluation": evaluation, "update": update_index, "best": True},
        )
        report_path = self.league_config.checkpoint_dir / "best.eval.json"
        report_path.write_text(json.dumps(evaluation, indent=2))

    def _log_update(self, update_index: int, metrics: Dict[str, float], evaluation: Optional[Dict[str, object]]) -> None:
        if self.writer is None:
            return
        stage_map = {"stage_1": 1.0, "stage_2": 2.0, "stage_3": 3.0}
        for key, value in metrics.items():
            if key == "stage":
                self.writer.add_scalar("train/stage_index", stage_map.get(str(value), 0.0), update_index)
                self.writer.add_text("train/stage_name", str(value), global_step=update_index)
                continue
            self.writer.add_scalar("train/{key}".format(key=key), float(value), update_index)
        if evaluation is not None:
            for tag, value in self._flatten_scalars(evaluation):
                self.writer.add_scalar(tag, value, update_index)
        self.writer.flush()

    def _flatten_scalars(self, value: object, prefix: str = "eval") -> List[tuple[str, float]]:
        scalars = []  # type: List[tuple[str, float]]
        if isinstance(value, dict):
            for key, nested_value in value.items():
                nested_prefix = "{prefix}/{key}".format(prefix=prefix, key=key)
                scalars.extend(self._flatten_scalars(nested_value, nested_prefix))
        elif isinstance(value, (int, float)):
            scalars.append((prefix, float(value)))
        return scalars
