from __future__ import annotations

# Training input:
#   models/model_000000.pt
#   gameplays/model_000000/game_000001.json
#
# Training output:
#   models/model_000001.pt
#
# The newest numeric checkpoint is used, and every replay in its matching
# gameplay directory is consumed once. The complete rollout is reused for all
# PPO epochs, shuffled again before each epoch, and split into mini-batches of
# at most --minibatch-size model decisions.
#
# Parameter format:
#   python train.py
#   python train.py --epochs 4 --minibatch-size 256
#   python train.py --learning-rate 0.0003 --device auto

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from constants import *
from model import HungryGeeseActorCritic
from utils import observation_to_tensor


@dataclass
class RolloutBatch:
    """Tensors for every model decision in one generated rollout batch.

    Explanation:
        Keeps observations and fixed behavior-policy quantities on CPU. A
        shuffled subset is moved to the training device for each mini-batch.

    Args:
        boards: Normalized channel-last observation tensors.
        remaining: Normalized remaining-time scalars.
        directions: Absolute directions used to orient the observations.
        actions: Selected relative actions.
        action_masks: Behavior-time masks in FORWARD, LEFT, RIGHT order.
        old_log_probabilities: Log probabilities under the behavior policy.
        old_values: Behavior-policy value predictions.
        advantages: GAE advantages, normalized only for actor-valid samples.
        returns: Value targets before advantage normalization.
        actor_valid: Whether each action was sampled from the masked policy.

    Returns:
        A structured collection of aligned rollout tensors.
    """

    boards: Tensor
    remaining: Tensor
    directions: Tensor
    actions: Tensor
    action_masks: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    actor_valid: Tensor

    def __len__(self) -> int:
        """Return the number of aligned model decisions.

        Explanation:
            Uses the leading board dimension because every stored tensor has
            the same number of samples.

        Args:
            None.

        Returns:
            Number of model decisions in the batch.
        """
        return self.boards.shape[0]


def _positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer.

    Explanation:
        Rejects zero and negative values for counts such as epochs and
        mini-batch size.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def _nonnegative_integer(value: str) -> int:
    """Parse a nonnegative command-line integer.

    Explanation:
        Accepts zero for a reproducible random seed while rejecting negative
        values.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed nonnegative integer.
    """
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed_value


def _positive_float(value: str) -> float:
    """Parse a finite positive command-line number.

    Explanation:
        Validates optimization parameters that must be greater than zero.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed finite positive number.
    """
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed_value


def _nonnegative_float(value: str) -> float:
    """Parse a finite nonnegative command-line number.

    Explanation:
        Validates loss coefficients that may be intentionally disabled with
        zero.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed finite nonnegative number.
    """
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite number of at least zero")
    return parsed_value


def _unit_interval(value: str) -> float:
    """Parse a command-line number in the closed unit interval.

    Explanation:
        Validates discount, GAE, and PPO clipping coefficients.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed value between zero and one inclusive.
    """
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed_value) or not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed_value


def parse_args() -> argparse.Namespace:
    """Parse PPO training command-line arguments.

    Explanation:
        Exposes the directories and commonly tuned PPO hyperparameters while
        taking every default value from ``constants.py``.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the next Hungry Geese model from generated replays.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(DEFAULT_MODELS_DIRECTORY),
        help=f"checkpoint directory (default: {DEFAULT_MODELS_DIRECTORY})",
    )
    parser.add_argument(
        "--gameplays-dir",
        type=Path,
        default=Path(DEFAULT_GAMEPLAYS_DIRECTORY),
        help=f"replay root directory (default: {DEFAULT_GAMEPLAYS_DIRECTORY})",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_integer,
        default=DEFAULT_PPO_EPOCHS,
        help=f"PPO passes over the rollout (default: {DEFAULT_PPO_EPOCHS})",
    )
    parser.add_argument(
        "--minibatch-size",
        type=_positive_integer,
        default=DEFAULT_MINIBATCH_SIZE,
        help=(
            "maximum model decisions per mini-batch "
            f"(default: {DEFAULT_MINIBATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Adam learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--gamma",
        type=_unit_interval,
        default=DEFAULT_DISCOUNT_FACTOR,
        help=f"reward discount factor (default: {DEFAULT_DISCOUNT_FACTOR})",
    )
    parser.add_argument(
        "--gae-lambda",
        type=_unit_interval,
        default=DEFAULT_GAE_LAMBDA,
        help=f"GAE trace decay (default: {DEFAULT_GAE_LAMBDA})",
    )
    parser.add_argument(
        "--clip-coefficient",
        type=_unit_interval,
        default=DEFAULT_CLIP_COEFFICIENT,
        help=f"PPO ratio clip range (default: {DEFAULT_CLIP_COEFFICIENT})",
    )
    parser.add_argument(
        "--value-loss-coefficient",
        type=_nonnegative_float,
        default=DEFAULT_VALUE_LOSS_COEFFICIENT,
        help=(
            "value loss weight "
            f"(default: {DEFAULT_VALUE_LOSS_COEFFICIENT})"
        ),
    )
    parser.add_argument(
        "--entropy-coefficient",
        type=_nonnegative_float,
        default=DEFAULT_ENTROPY_COEFFICIENT,
        help=(
            "masked-policy entropy weight "
            f"(default: {DEFAULT_ENTROPY_COEFFICIENT})"
        ),
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=_positive_float,
        default=DEFAULT_MAX_GRADIENT_NORM,
        help=(
            "gradient clipping norm "
            f"(default: {DEFAULT_MAX_GRADIENT_NORM})"
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_TRAINING_DEVICE,
        help=f"auto, cpu, cuda, cuda:N, or mps (default: {DEFAULT_TRAINING_DEVICE})",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        default=DEFAULT_TRAINING_SEED,
        help=f"training shuffle seed (default: {DEFAULT_TRAINING_SEED})",
    )
    return parser.parse_args()


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested training device.

    Explanation:
        Automatic selection prefers CUDA, then Apple MPS, and finally CPU.
        Explicit unavailable accelerators raise an error instead of silently
        changing the requested behavior.

    Args:
        requested_device: Device name supplied on the command line.

    Returns:
        Valid PyTorch device for training.
    """
    if requested_device == AUTO_DEVICE:
        if torch.cuda.is_available():
            return torch.device(CUDA_DEVICE)
        if torch.backends.mps.is_available():
            return torch.device(MPS_DEVICE)
        return torch.device(CPU_DEVICE)

    device = torch.device(requested_device)
    if device.type == CUDA_DEVICE and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == MPS_DEVICE and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _find_latest_checkpoint(models_directory: Path) -> tuple[Path, int]:
    """Find the checkpoint with the greatest numeric model ID.

    Explanation:
        Examines files matching the shared checkpoint naming convention and
        selects by numeric ID.

    Args:
        models_directory: Directory containing model checkpoints.

    Returns:
        Latest checkpoint path and its numeric model ID.
    """
    candidates: list[tuple[int, Path]] = []
    if models_directory.is_dir():
        for path in models_directory.iterdir():
            if not path.is_file():
                continue
            match = re.fullmatch(CHECKPOINT_FILENAME_PATTERN, path.name)
            if match is not None:
                candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(
            f"no model checkpoint found in {models_directory}; run generate.py first"
        )

    model_id, checkpoint_path = max(candidates, key=lambda candidate: candidate[0])
    return checkpoint_path, model_id


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HungryGeeseActorCritic, Mapping[str, Any] | None]:
    """Load a behavior checkpoint and optional Adam state.

    Explanation:
        Accepts either a raw model state dictionary or a training wrapper. A
        wrapper may also preserve the optimizer across PPO iterations.

    Args:
        checkpoint_path: Behavior checkpoint to load.
        device: Device used for PPO optimization.

    Returns:
        Actor-critic model in training mode and an optional optimizer state.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=CPU_DEVICE,
        weights_only=True,
    )
    optimizer_state: Mapping[str, Any] | None = None
    model_state: object = checkpoint
    if isinstance(checkpoint, Mapping) and CHECKPOINT_MODEL_STATE_KEY in checkpoint:
        model_state = checkpoint[CHECKPOINT_MODEL_STATE_KEY]
        possible_optimizer_state = checkpoint.get(CHECKPOINT_OPTIMIZER_STATE_KEY)
        if possible_optimizer_state is not None:
            if not isinstance(possible_optimizer_state, Mapping):
                raise TypeError("optimizer_state_dict must be a mapping")
            optimizer_state = cast(Mapping[str, Any], possible_optimizer_state)
    if not isinstance(model_state, Mapping):
        raise TypeError("checkpoint must be a state_dict or contain model_state_dict")

    model = HungryGeeseActorCritic()
    model.load_state_dict(cast(Mapping[str, Tensor], model_state))
    model.to(device)
    model.train()
    return model, optimizer_state


def _find_replay_paths(gameplays_directory: Path, model_id: int) -> list[Path]:
    """Find all replay files generated by one behavior model.

    Explanation:
        Reads only files matching the shared sequential replay name inside the
        directory associated with ``model_id`` and sorts them numerically.

    Args:
        gameplays_directory: Root gameplay directory.
        model_id: Behavior-model ID recorded by the rollouts.

    Returns:
        Numerically sorted replay paths.
    """
    replay_directory = gameplays_directory / REPLAY_MODEL_DIRECTORY_TEMPLATE.format(
        model_id=model_id,
    )
    candidates: list[tuple[int, Path]] = []
    if replay_directory.is_dir():
        for path in replay_directory.iterdir():
            if not path.is_file():
                continue
            match = re.fullmatch(REPLAY_FILENAME_PATTERN, path.name)
            if match is not None:
                candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(
            f"no replay JSON files found in {replay_directory}; run generate.py first"
        )
    return [path for _, path in sorted(candidates)]


def _read_replay(replay_path: Path) -> Mapping[str, Any]:
    """Read one replay JSON object.

    Explanation:
        Loads UTF-8 JSON and rejects a non-object root before rollout parsing.

    Args:
        replay_path: Replay file to read.

    Returns:
        Parsed replay mapping.
    """
    with replay_path.open("r", encoding=REPLAY_FILE_ENCODING) as replay_file:
        replay = json.load(replay_file)
    if not isinstance(replay, Mapping):
        raise ValueError(f"replay root must be an object: {replay_path}")
    return replay


def _observation_for_player(
    steps: Sequence[Any],
    step: int,
    player: int,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Recover one player's full observation and state entry.

    Explanation:
        Kaggle replays may store shared observation fields only on player zero.
        The shared and player-specific mappings are therefore merged, then the
        correct player index and step are enforced.

    Args:
        steps: Official replay steps.
        step: Observation step used for the decision.
        player: Player whose observation is required.

    Returns:
        Full observation dictionary and that player's state entry.
    """
    try:
        step_entries = steps[step]
        player_state = step_entries[player]
        shared_state = step_entries[0]
    except (IndexError, TypeError) as error:
        raise ValueError(f"missing replay state for step {step}, player {player}") from error
    if not isinstance(player_state, Mapping) or not isinstance(shared_state, Mapping):
        raise ValueError(f"invalid replay state for step {step}, player {player}")

    player_observation = player_state.get("observation")
    shared_observation = shared_state.get("observation")
    if not isinstance(player_observation, Mapping):
        raise ValueError(f"missing observation for step {step}, player {player}")

    observation: dict[str, Any] = {}
    if isinstance(shared_observation, Mapping):
        observation.update(shared_observation)
    observation.update(player_observation)
    observation["index"] = player
    observation["step"] = step
    return observation, player_state


def _numeric_reward(state: Mapping[str, Any], label: str) -> float:
    """Read one finite numeric reward from a replay state.

    Explanation:
        Converts integer or floating cumulative scores to a common float and
        rejects missing, Boolean, or non-finite values.

    Args:
        state: Player state containing the official cumulative reward.
        label: Location included in validation errors.

    Returns:
        Finite cumulative reward.
    """
    reward = state.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError(f"{label} reward must be numeric")
    numeric_reward = float(reward)
    if not math.isfinite(numeric_reward):
        raise ValueError(f"{label} reward must be finite")
    return numeric_reward


def _final_scores(replay: Mapping[str, Any], steps: Sequence[Any]) -> list[float]:
    """Obtain the four official final scores used for ranking.

    Explanation:
        Prefers the replay's top-level final rewards. If unavailable, reads the
        last official step, which contains the same cumulative player scores.

    Args:
        replay: Complete replay mapping.
        steps: Official replay steps.

    Returns:
        Four finite final scores in player order.
    """
    rewards = replay.get("rewards")
    if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes)):
        if len(rewards) == NUM_PLAYERS and all(
            not isinstance(reward, bool) and isinstance(reward, (int, float))
            for reward in rewards
        ):
            scores = [float(reward) for reward in rewards]
            if all(math.isfinite(score) for score in scores):
                return scores

    if not steps:
        raise ValueError("replay has no official steps")
    final_entries = steps[-1]
    if not isinstance(final_entries, Sequence) or len(final_entries) != NUM_PLAYERS:
        raise ValueError("final replay step must contain exactly four players")
    return [
        _numeric_reward(cast(Mapping[str, Any], state), f"final player {player}")
        for player, state in enumerate(final_entries)
    ]


def _terminal_rank_bonuses(final_scores: Sequence[float]) -> list[float]:
    """Convert final scores to competitive terminal rewards.

    Explanation:
        Higher official score ranks first. Ties take the lower occupied rank:
        two players tied for first both receive the second-place reward.

    Args:
        final_scores: Official final scores in player order.

    Returns:
        Terminal ranking bonus for every player.
    """
    if len(final_scores) != NUM_PLAYERS:
        raise ValueError(f"expected exactly {NUM_PLAYERS} final scores")
    bonuses: list[float] = []
    for score in final_scores:
        lower_tied_rank = sum(other_score >= score for other_score in final_scores)
        bonuses.append(TERMINAL_RANK_REWARDS[lower_tied_rank - 1])
    return bonuses


def _old_log_probability(
    logits: Sequence[Any],
    action_mask: Sequence[Any],
    action: int,
    actor_valid: bool,
) -> float:
    """Calculate a behavior-policy log probability from saved raw logits.

    Explanation:
        Applies the exact saved action mask before normalization. All-masked
        random fallbacks return a harmless placeholder because they are later
        excluded from actor and entropy losses.

    Args:
        logits: Three saved behavior-policy logits.
        action_mask: Three saved action-availability flags.
        action: Selected relative action.
        actor_valid: Whether the action came from the masked policy.

    Returns:
        Behavior-policy log probability, or zero for an actor-invalid sample.
    """
    if len(logits) != NUM_ACTIONS or len(action_mask) != NUM_ACTIONS:
        raise ValueError(f"logits and action_mask must each contain {NUM_ACTIONS} values")
    if action not in VALID_ACTIONS:
        raise ValueError("relative_action is outside the policy action space")

    mask = torch.tensor([bool(value) for value in action_mask], dtype=torch.bool)
    if not actor_valid:
        return 0.0
    if not bool(mask[action]):
        raise ValueError("policy-sampled action is disabled by its saved mask")

    numeric_logits = torch.tensor([float(value) for value in logits], dtype=torch.float32)
    if not bool(torch.isfinite(numeric_logits).all()):
        raise ValueError("saved action logits must be finite")
    masked_logits = numeric_logits.masked_fill(~mask, MASKED_LOGIT)
    return float(torch.log_softmax(masked_logits, dim=0)[action].item())


def _compute_gae(
    rewards: Sequence[float],
    dones: Sequence[bool],
    values: Sequence[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute generalized advantage estimates for one player trajectory.

    Explanation:
        Walks backward through a contiguous trajectory. A terminal transition
        has zero bootstrap value and stops advantage propagation.

    Args:
        rewards: Per-transition shaped rewards.
        dones: Whether each transition reaches a terminal state.
        values: Saved behavior-policy value estimates.
        gamma: Reward discount factor.
        gae_lambda: GAE trace decay.

    Returns:
        Unnormalized advantages and value targets in chronological order.
    """
    if not (len(rewards) == len(dones) == len(values)):
        raise ValueError("reward, done, and value sequences must have equal lengths")

    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if dones[index] else 1.0
        next_value = (
            values[index + 1]
            if nonterminal and index + 1 < len(values)
            else 0.0
        )
        delta = rewards[index] + gamma * nonterminal * next_value - values[index]
        next_advantage = (
            delta + gamma * gae_lambda * nonterminal * next_advantage
        )
        advantages[index] = next_advantage

    returns = [
        advantage + value
        for advantage, value in zip(advantages, values)
    ]
    return advantages, returns


def _load_rollouts(
    replay_paths: Sequence[Path],
    model_id: int,
    gamma: float,
    gae_lambda: float,
) -> RolloutBatch:
    """Convert generated replay files into one PPO rollout batch.

    Explanation:
        Reconstructs every model observation, calculates normalized official
        score deltas, adds the final competitive ranking bonus, computes GAE
        independently per player, and finally concatenates all games.

    Args:
        replay_paths: Replays generated by the behavior model.
        model_id: Required behavior-model identifier.
        gamma: Reward discount factor.
        gae_lambda: GAE trace decay.

    Returns:
        CPU tensors containing all model decisions.
    """
    boards: list[np.ndarray] = []
    remaining_values: list[float] = []
    directions: list[int] = []
    actions: list[int] = []
    action_masks: list[list[bool]] = []
    old_log_probabilities: list[float] = []
    old_values: list[float] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []
    actor_valid_flags: list[bool] = []

    for replay_path in replay_paths:
        replay = _read_replay(replay_path)
        ppo_data = replay.get(PPO_REPLAY_KEY)
        steps = replay.get("steps")
        if not isinstance(ppo_data, Mapping):
            raise ValueError(f"missing {PPO_REPLAY_KEY} object: {replay_path}")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ValueError(f"missing official steps: {replay_path}")
        if int(ppo_data.get("schema_version", -1)) != PPO_REPLAY_SCHEMA_VERSION:
            raise ValueError(f"unsupported PPO schema version: {replay_path}")
        if int(ppo_data.get("model_id", -1)) != model_id:
            raise ValueError(f"replay model_id does not match checkpoint: {replay_path}")

        configuration = replay.get("configuration")
        if isinstance(configuration, Mapping):
            episode_steps = int(
                configuration.get(EPISODE_STEPS_CONFIGURATION_KEY, MAX_STEPS)
            )
            max_length = int(
                configuration.get(MAX_LENGTH_CONFIGURATION_KEY, MAX_GOOSE_LENGTH)
            )
            if episode_steps != MAX_STEPS or max_length != MAX_GOOSE_LENGTH:
                raise ValueError(
                    "replay reward normalization requires "
                    f"{EPISODE_STEPS_CONFIGURATION_KEY}={MAX_STEPS} and "
                    f"{MAX_LENGTH_CONFIGURATION_KEY}={MAX_GOOSE_LENGTH}: "
                    f"{replay_path}"
                )

        trajectories = ppo_data.get("trajectories")
        trainable_players = ppo_data.get("trainable_players")
        if not isinstance(trajectories, Sequence) or len(trajectories) != NUM_PLAYERS:
            raise ValueError(f"invalid PPO trajectories: {replay_path}")
        if not isinstance(trainable_players, Sequence):
            raise ValueError(f"invalid trainable_players: {replay_path}")

        rank_bonuses = _terminal_rank_bonuses(_final_scores(replay, steps))
        for raw_player in trainable_players:
            player = int(raw_player)
            if not 0 <= player < NUM_PLAYERS:
                raise ValueError(f"invalid trainable player in {replay_path}")
            trajectory = trajectories[player]
            if not isinstance(trajectory, Sequence) or isinstance(
                trajectory,
                (str, bytes),
            ):
                raise ValueError(f"invalid player trajectory in {replay_path}")
            if not trajectory:
                continue

            trajectory_boards: list[np.ndarray] = []
            trajectory_remaining: list[float] = []
            trajectory_directions: list[int] = []
            trajectory_actions: list[int] = []
            trajectory_masks: list[list[bool]] = []
            trajectory_log_probabilities: list[float] = []
            trajectory_values: list[float] = []
            trajectory_actor_valid: list[bool] = []
            trajectory_rewards: list[float] = []
            trajectory_dones: list[bool] = []
            trajectory_steps: list[int] = []

            for raw_record in trajectory:
                if not isinstance(raw_record, Mapping):
                    raise ValueError(f"invalid trajectory record in {replay_path}")
                step = int(raw_record["step"])
                if step <= 0 or step + 1 >= len(steps):
                    raise ValueError(
                        f"trajectory step {step} has no aligned transition in {replay_path}"
                    )
                if trajectory_steps and step != trajectory_steps[-1] + 1:
                    raise ValueError(
                        f"player {player} trajectory is not contiguous in {replay_path}"
                    )

                observation, current_state = _observation_for_player(
                    steps,
                    step,
                    player,
                )
                _, next_state = _observation_for_player(
                    steps,
                    step + 1,
                    player,
                )
                if current_state.get("status") != ACTIVE_STATUS:
                    raise ValueError(
                        f"trajectory contains an inactive decision at step {step}, "
                        f"player {player} in {replay_path}"
                    )
                previous_action = current_state.get("action")
                previous_direction = (
                    DIRECTION_NAMES.index(previous_action)
                    if previous_action in DIRECTION_NAMES
                    else None
                )
                board, direction = observation_to_tensor(
                    observation,
                    previous_direction,
                )

                action = int(raw_record["relative_action"])
                if action not in VALID_ACTIONS:
                    raise ValueError(f"invalid relative action in {replay_path}")
                expected_absolute_action = DIRECTION_NAMES[
                    (direction + RELATIVE_DIRECTION_OFFSETS[action])
                    % len(VALID_DIRECTIONS)
                ]
                if next_state.get("action") != expected_absolute_action:
                    raise ValueError(
                        f"saved relative action does not match replay action at "
                        f"step {step}, player {player} in {replay_path}"
                    )
                raw_logits = raw_record["action_logit_vec"]
                raw_mask = raw_record["action_mask"]
                if not isinstance(raw_logits, Sequence) or not isinstance(
                    raw_mask,
                    Sequence,
                ):
                    raise ValueError(f"invalid logits or action mask in {replay_path}")
                mask = [bool(value) for value in raw_mask]
                actor_valid = bool(raw_record["policy_sampled"])
                old_log_probability = _old_log_probability(
                    raw_logits,
                    mask,
                    action,
                    actor_valid,
                )
                old_value = float(raw_record["value"])
                if not math.isfinite(old_value):
                    raise ValueError(f"saved value must be finite in {replay_path}")

                current_reward = _numeric_reward(
                    current_state,
                    f"step {step}, player {player}",
                )
                next_reward = _numeric_reward(
                    next_state,
                    f"step {step + 1}, player {player}",
                )
                shaped_reward = (
                    next_reward - current_reward
                ) / REWARD_NORMALIZATION_DENOMINATOR
                done = (
                    next_state.get("status") != ACTIVE_STATUS
                    or step + 1 == len(steps) - 1
                )

                trajectory_boards.append(board)
                trajectory_remaining.append(
                    min(
                        max(
                            (MAX_STEPS - step) / MAX_STEPS,
                            MIN_REMAINING_FRACTION,
                        ),
                        MAX_REMAINING_FRACTION,
                    )
                )
                trajectory_directions.append(direction)
                trajectory_actions.append(action)
                trajectory_masks.append(mask)
                trajectory_log_probabilities.append(old_log_probability)
                trajectory_values.append(old_value)
                trajectory_actor_valid.append(actor_valid)
                trajectory_rewards.append(shaped_reward)
                trajectory_dones.append(done)
                trajectory_steps.append(step)

            if not trajectory_dones[-1]:
                raise ValueError(
                    f"player {player} trajectory does not end at a terminal state in "
                    f"{replay_path}"
                )
            trajectory_rewards[-1] += rank_bonuses[player]
            advantages, returns = _compute_gae(
                trajectory_rewards,
                trajectory_dones,
                trajectory_values,
                gamma,
                gae_lambda,
            )

            boards.extend(trajectory_boards)
            remaining_values.extend(trajectory_remaining)
            directions.extend(trajectory_directions)
            actions.extend(trajectory_actions)
            action_masks.extend(trajectory_masks)
            old_log_probabilities.extend(trajectory_log_probabilities)
            old_values.extend(trajectory_values)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            actor_valid_flags.extend(trajectory_actor_valid)

    if not boards:
        raise ValueError("the selected replays contain no model decisions")

    advantage_tensor = torch.tensor(all_advantages, dtype=torch.float32)
    actor_valid_tensor = torch.tensor(actor_valid_flags, dtype=torch.bool)
    valid_advantages = advantage_tensor[actor_valid_tensor]
    if valid_advantages.numel() > 1:
        normalized_advantages = (
            valid_advantages - valid_advantages.mean()
        ) / (
            valid_advantages.std(unbiased=False) + ADVANTAGE_NORMALIZATION_EPSILON
        )
        advantage_tensor[actor_valid_tensor] = normalized_advantages

    return RolloutBatch(
        boards=torch.from_numpy(np.stack(boards)).to(dtype=torch.float32),
        remaining=torch.tensor(remaining_values, dtype=torch.float32),
        directions=torch.tensor(directions, dtype=torch.long),
        actions=torch.tensor(actions, dtype=torch.long),
        action_masks=torch.tensor(action_masks, dtype=torch.bool),
        old_log_probabilities=torch.tensor(
            old_log_probabilities,
            dtype=torch.float32,
        ),
        old_values=torch.tensor(old_values, dtype=torch.float32),
        advantages=advantage_tensor,
        returns=torch.tensor(all_returns, dtype=torch.float32),
        actor_valid=actor_valid_tensor,
    )


def _train_model(
    model: HungryGeeseActorCritic,
    rollout: RolloutBatch,
    args: argparse.Namespace,
    device: torch.device,
    optimizer_state: Mapping[str, Any] | None,
) -> torch.optim.Optimizer:
    """Optimize the actor-critic with clipped PPO updates.

    Explanation:
        Shuffles every model decision for each epoch. Actor probability ratios
        and entropy use the saved action mask; random all-masked fallbacks are
        excluded from actor terms but remain in the value loss.

    Args:
        model: Behavior model to update in place.
        rollout: Fixed rollout data and GAE targets.
        args: Parsed PPO hyperparameters.
        device: Device used for forward and backward passes.
        optimizer_state: Adam state from the previous iteration, if available.

    Returns:
        Updated optimizer whose state should be saved with the model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(dict(optimizer_state))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
    sample_count = len(rollout)

    for epoch in range(args.epochs):
        permutation = torch.randperm(sample_count)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        actor_count = 0
        value_count = 0

        for start in range(0, sample_count, args.minibatch_size):
            indices = permutation[start : start + args.minibatch_size]
            boards = rollout.boards[indices].to(device)
            remaining = rollout.remaining[indices].to(device)
            directions = rollout.directions[indices].to(device)
            actions = rollout.actions[indices].to(device)
            masks = rollout.action_masks[indices].to(device)
            old_log_probabilities = rollout.old_log_probabilities[indices].to(device)
            advantages = rollout.advantages[indices].to(device)
            returns = rollout.returns[indices].to(device)
            actor_valid = rollout.actor_valid[indices].to(device)

            new_logits, new_values = model(boards, remaining, directions)
            new_values = new_values.squeeze(-1)
            value_loss = 0.5 * F.mse_loss(new_values, returns)

            if bool(actor_valid.any()):
                valid_logits = new_logits[actor_valid]
                valid_masks = masks[actor_valid]
                valid_actions = actions[actor_valid]
                masked_logits = valid_logits.masked_fill(
                    ~valid_masks,
                    MASKED_LOGIT,
                )
                distribution = torch.distributions.Categorical(logits=masked_logits)
                new_log_probabilities = distribution.log_prob(valid_actions)
                probability_ratios = torch.exp(
                    new_log_probabilities - old_log_probabilities[actor_valid]
                )
                valid_advantages = advantages[actor_valid]
                unclipped_objective = probability_ratios * valid_advantages
                clipped_objective = torch.clamp(
                    probability_ratios,
                    1.0 - args.clip_coefficient,
                    1.0 + args.clip_coefficient,
                ) * valid_advantages
                policy_loss = -torch.minimum(
                    unclipped_objective,
                    clipped_objective,
                ).mean()
                entropy = distribution.entropy().mean()
                current_actor_count = int(actor_valid.sum().item())
            else:
                policy_loss = new_values.new_zeros(())
                entropy = new_values.new_zeros(())
                current_actor_count = 0

            total_loss = (
                policy_loss
                + args.value_loss_coefficient * value_loss
                - args.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_gradient_norm,
            )
            optimizer.step()

            current_value_count = len(indices)
            policy_loss_sum += float(policy_loss.item()) * current_actor_count
            entropy_sum += float(entropy.item()) * current_actor_count
            value_loss_sum += float(value_loss.item()) * current_value_count
            actor_count += current_actor_count
            value_count += current_value_count

        mean_policy_loss = policy_loss_sum / max(actor_count, 1)
        mean_entropy = entropy_sum / max(actor_count, 1)
        mean_value_loss = value_loss_sum / value_count
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"policy_loss={mean_policy_loss:.6f}, "
            f"value_loss={mean_value_loss:.6f}, "
            f"entropy={mean_entropy:.6f}"
        )
    return optimizer


def _save_next_checkpoint(
    model: HungryGeeseActorCritic,
    optimizer: torch.optim.Optimizer,
    models_directory: Path,
    model_id: int,
) -> Path:
    """Save the trained model under the next sequential ID.

    Explanation:
        Moves parameters and Adam tensors to CPU, stores both state dictionaries
        in one portable checkpoint, and refuses to overwrite an existing file.

    Args:
        model: Trained actor-critic.
        optimizer: Updated Adam optimizer.
        models_directory: Destination checkpoint directory.
        model_id: Behavior-model ID used to create the rollout.

    Returns:
        Path of the newly saved checkpoint.
    """
    next_model_id = model_id + 1
    checkpoint_path = models_directory / CHECKPOINT_FILENAME_TEMPLATE.format(
        model_id=next_model_id,
    )
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}")

    models_directory.mkdir(parents=True, exist_ok=True)
    model.to(CPU_DEVICE)
    optimizer_state = optimizer.state_dict()
    for parameter_state in optimizer_state["state"].values():
        for key, value in parameter_state.items():
            if isinstance(value, Tensor):
                parameter_state[key] = value.detach().cpu()
    torch.save(
        {
            CHECKPOINT_MODEL_STATE_KEY: model.state_dict(),
            CHECKPOINT_OPTIMIZER_STATE_KEY: optimizer_state,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train(args: argparse.Namespace) -> Path:
    """Train and save the next model iteration.

    Explanation:
        Locates the newest checkpoint and its generated games, constructs one
        rollout batch, performs PPO optimization, and writes the next numeric
        checkpoint only after training succeeds.

    Args:
        args: Parsed training configuration.

    Returns:
        Path of the newly saved checkpoint.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _resolve_device(args.device)
    checkpoint_path, model_id = _find_latest_checkpoint(args.models_dir)
    replay_paths = _find_replay_paths(args.gameplays_dir, model_id)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using {len(replay_paths)} replay(s) on {device}")

    rollout = _load_rollouts(
        replay_paths,
        model_id,
        args.gamma,
        args.gae_lambda,
    )
    actor_samples = int(rollout.actor_valid.sum().item())
    print(
        f"Loaded {len(rollout)} model decisions "
        f"({actor_samples} actor-valid)"
    )

    model, optimizer_state = _load_model(checkpoint_path, device)
    optimizer = _train_model(model, rollout, args, device, optimizer_state)
    saved_path = _save_next_checkpoint(
        model,
        optimizer,
        args.models_dir,
        model_id,
    )
    print(f"Saved checkpoint: {saved_path}")
    return saved_path


def main() -> None:
    """Run one PPO training iteration from the command line.

    Explanation:
        Parses CLI arguments and delegates the complete training iteration to
        ``train``.

    Args:
        None.

    Returns:
        None.
    """
    train(parse_args())


if __name__ == "__main__":
    main()
