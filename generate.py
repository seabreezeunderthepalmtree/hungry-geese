from __future__ import annotations

# Checkpoint format:
#   models/model_000000.pt
#   models/model_000001.pt
# The greatest numeric model ID is loaded. If none exists, model_000000.pt is
# created from the random initialization before any gameplay is generated.
#
# Replay format:
#   gameplays/model_000001/game_000001.json
# Every file contains the official Kaggle replay plus a top-level "ppo" object.
# ppo.trajectories[player] contains model decisions for that player. Step zero
# and SimpleAgent decisions are intentionally absent. A decision whose
# policy_sampled value is false used the all-masked random fallback and should
# be excluded from the PPO actor loss while remaining usable by the critic.
#
# Parameter format:
#   python generate.py --games 16
#   python generate.py --simple-agent-probability 0.1
#   python generate.py --models-dir models --gameplays-dir gameplays --debug

import argparse
import json
import random
import re
from pathlib import Path
from typing import Mapping, cast

import torch
from kaggle_environments import make
from torch import Tensor

from agent import Agent, SimpleAgent
from constants import *
from model import HungryGeeseActorCritic


def _positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer.

    Explanation:
        Converts an argparse value to an integer and rejects zero or negative
        values so generation always requests at least one game.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def _probability(value: str) -> float:
    """Parse a command-line probability from zero through one.

    Explanation:
        Converts an argparse value to float and validates the closed unit
        interval used for the SimpleAgent probability.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed probability.

    Raises:
        argparse.ArgumentTypeError: If the value is not in ``[0, 1]``.
    """
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error

    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed_value


def parse_args() -> argparse.Namespace:
    """Parse gameplay-generation command-line arguments.

    Explanation:
        Exposes commonly changed generation options while keeping every default
        value in ``constants.py``.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate PPO-ready Hungry Geese replay JSON files.",
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
        "--games",
        type=_positive_integer,
        default=DEFAULT_GAMES_PER_GENERATION,
        help=(
            "number of JSON replay files to generate "
            f"(default: {DEFAULT_GAMES_PER_GENERATION})"
        ),
    )
    parser.add_argument(
        "--simple-agent-probability",
        type=_probability,
        default=DEFAULT_SIMPLE_AGENT_PROBABILITY,
        help=(
            "probability that each seat uses SimpleAgent "
            f"(default: {DEFAULT_SIMPLE_AGENT_PROBABILITY})"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=DEFAULT_GENERATION_DEBUG,
        help="print Kaggle environment debug messages",
    )
    return parser.parse_args()


def _find_latest_checkpoint(models_directory: Path) -> tuple[Path, int] | None:
    """Find the checkpoint with the greatest numeric model ID.

    Explanation:
        Examines only files matching ``model_<id>.pt`` or ``model_<id>.pth``.
        Numeric IDs, rather than timestamps or lexicographic order, determine
        which model is latest.

    Args:
        models_directory: Directory containing model checkpoints.

    Returns:
        ``(checkpoint_path, model_id)`` for the latest checkpoint, or ``None``
        when the directory is absent or contains no valid checkpoint.
    """
    if not models_directory.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in models_directory.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(CHECKPOINT_FILENAME_PATTERN, path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return None

    model_id, checkpoint_path = max(candidates, key=lambda candidate: candidate[0])
    return checkpoint_path, model_id


def _load_model(checkpoint_path: Path | None) -> HungryGeeseActorCritic:
    """Create the inference model and optionally load checkpoint weights.

    Explanation:
        Initializes the actor-critic on CPU. A checkpoint may be either a raw
        PyTorch state dictionary or a wrapper containing ``model_state_dict``.
        With no checkpoint, the model keeps its normal random initialization.

    Args:
        checkpoint_path: Checkpoint path, or ``None`` for random initialization.

    Returns:
        Model in evaluation mode on the generation device.
    """
    model = HungryGeeseActorCritic().to(GENERATION_DEVICE)
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=GENERATION_DEVICE,
            weights_only=True,
        )
        if isinstance(checkpoint, Mapping) and CHECKPOINT_MODEL_STATE_KEY in checkpoint:
            checkpoint = checkpoint[CHECKPOINT_MODEL_STATE_KEY]
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "checkpoint must be a state_dict or contain model_state_dict"
            )
        model.load_state_dict(cast(Mapping[str, Tensor], checkpoint))

    model.eval()
    return model


def _prepare_model(
    models_directory: Path,
) -> tuple[HungryGeeseActorCritic, int, Path]:
    """Load the latest model or create and save model zero.

    Explanation:
        Ensures every generated trajectory has a persistent checkpoint matching
        the behavior policy. When no checkpoint exists, the random weights are
        saved before play as ``model_000000.pt``.

    Args:
        models_directory: Directory used to find and save checkpoints.

    Returns:
        ``(model, model_id, checkpoint_path)`` for the behavior policy.
    """
    latest_checkpoint = _find_latest_checkpoint(models_directory)
    if latest_checkpoint is not None:
        checkpoint_path, model_id = latest_checkpoint
        return _load_model(checkpoint_path), model_id, checkpoint_path

    model_id = 0
    model = _load_model(None)
    models_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_directory / CHECKPOINT_FILENAME_TEMPLATE.format(
        model_id=model_id,
    )
    torch.save(model.state_dict(), checkpoint_path)
    return model, model_id, checkpoint_path


def _next_replay_id(output_directory: Path) -> int:
    """Find the next unused sequential replay ID.

    Explanation:
        Scans files matching ``game_<id>.json`` and returns one above the
        greatest ID so resumed generation does not overwrite existing games.

    Args:
        output_directory: Model-specific replay directory.

    Returns:
        Positive integer ID for the next replay file.
    """
    greatest_id = 0
    if output_directory.is_dir():
        for path in output_directory.iterdir():
            if not path.is_file():
                continue
            match = re.fullmatch(REPLAY_FILENAME_PATTERN, path.name)
            if match is not None:
                greatest_id = max(greatest_id, int(match.group(1)))
    return greatest_id + 1


def _build_agents(
    model: HungryGeeseActorCritic,
    simple_agent_probability: float,
) -> tuple[list[object], list[str], list[Agent | None]]:
    """Build one randomized four-player matchup.

    Explanation:
        Independently assigns SimpleAgent to each seat with the configured
        probability and Agent otherwise. At least one model Agent is guaranteed
        so every saved game contributes trainable data.

    Args:
        model: Shared fixed behavior model for all model-controlled seats.
        simple_agent_probability: Probability that one seat uses SimpleAgent.

    Returns:
        Kaggle agents, player-type labels, and aligned model-agent recorders.
    """
    agents: list[object] = []
    player_types: list[str] = []
    model_agents: list[Agent | None] = []

    for _ in range(NUM_PLAYERS):
        if random.random() < simple_agent_probability:
            agents.append(SimpleAgent())
            player_types.append(SIMPLE_PLAYER_TYPE)
            model_agents.append(None)
        else:
            model_agent = Agent(model)
            agents.append(model_agent)
            player_types.append(MODEL_PLAYER_TYPE)
            model_agents.append(model_agent)

    if all(model_agent is None for model_agent in model_agents):
        player = random.randrange(NUM_PLAYERS)
        model_agent = Agent(model)
        agents[player] = model_agent
        player_types[player] = MODEL_PLAYER_TYPE
        model_agents[player] = model_agent

    return agents, player_types, model_agents


def _attach_ppo_data(
    replay: dict[str, object],
    model_id: int,
    player_types: list[str],
    model_agents: list[Agent | None],
) -> None:
    """Attach model rollout data without altering official replay steps.

    Explanation:
        Adds a separate top-level PPO object. Its trajectories remain aligned by
        player index, while non-model players receive empty trajectories.

    Args:
        replay: Mutable result returned by ``environment.toJSON()``.
        model_id: Numeric behavior-model identifier.
        player_types: Controller type for every player position.
        model_agents: Model Agent instances aligned with player positions.

    Returns:
        None.
    """
    trainable_players = [
        player
        for player, model_agent in enumerate(model_agents)
        if model_agent is not None
    ]
    trajectories = [
        [] if model_agent is None else model_agent.ppo_trajectory
        for model_agent in model_agents
    ]
    replay[PPO_REPLAY_KEY] = {
        "schema_version": PPO_REPLAY_SCHEMA_VERSION,
        "model_id": model_id,
        "player_types": player_types,
        "trainable_players": trainable_players,
        "trajectories": trajectories,
    }


def _save_replay(replay: Mapping[str, object], replay_path: Path) -> None:
    """Save one complete Kaggle replay and its PPO data as compact JSON.

    Explanation:
        Writes each game immediately so completed games remain available if a
        later game fails.

    Args:
        replay: Complete replay with the additional PPO object.
        replay_path: Destination JSON path.

    Returns:
        None.
    """
    with replay_path.open("w", encoding=REPLAY_FILE_ENCODING) as replay_file:
        json.dump(
            replay,
            replay_file,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def generate_games(args: argparse.Namespace) -> list[Path]:
    """Generate PPO-ready Hungry Geese gameplay files.

    Explanation:
        Uses one persistent CPU behavior model for the whole batch. Each game
        randomizes model and SimpleAgent seats, runs at most ``MAX_STEPS``, and
        saves both the official replay and aligned PPO decision data.

    Args:
        args: Parsed arguments from ``parse_args``.

    Returns:
        Paths of all replay files generated by this call.
    """
    model, model_id, checkpoint_path = _prepare_model(args.models_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    output_directory = args.gameplays_dir / REPLAY_MODEL_DIRECTORY_TEMPLATE.format(
        model_id=model_id,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    first_replay_id = _next_replay_id(output_directory)
    generated_paths: list[Path] = []

    for offset in range(args.games):
        replay_id = first_replay_id + offset
        agents, player_types, model_agents = _build_agents(
            model,
            args.simple_agent_probability,
        )
        environment = make(
            ENVIRONMENT_NAME,
            configuration={EPISODE_STEPS_CONFIGURATION_KEY: MAX_STEPS},
            debug=args.debug,
        )
        environment.run(agents)

        replay = environment.toJSON()
        _attach_ppo_data(
            replay,
            model_id,
            player_types,
            model_agents,
        )
        replay_path = output_directory / REPLAY_FILENAME_TEMPLATE.format(
            game_id=replay_id,
        )
        _save_replay(replay, replay_path)
        generated_paths.append(replay_path)
        print(f"Saved {replay_path}")

    return generated_paths


def main() -> None:
    """Run gameplay generation from the command line.

    Explanation:
        Parses user-editable arguments and generates the requested replay batch.

    Args:
        None.

    Returns:
        None.
    """
    generate_games(parse_args())


if __name__ == "__main__":
    main()
