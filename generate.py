from __future__ import annotations



import argparse
import json
import re
from pathlib import Path
from typing import Mapping, cast

import torch
from kaggle_environments import make
from torch import Tensor

from agent import Agent
from constants import *
from model import HungryGeeseActorCritic


def _positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer.

    Explanation:
        Converts an argparse value to an integer and rejects zero or negative
        values so a generation command always requests at least one game.

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


def parse_args() -> argparse.Namespace:
    """Parse gameplay-generation command-line arguments.

    Explanation:
        Provides editable model location, replay location, game count, and
        Kaggle debug output while keeping their defaults in ``constants.py``.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate Hungry Geese self-play replay JSON files.",
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
        "--debug",
        action="store_true",
        default=DEFAULT_GENERATION_DEBUG,
        help="print Kaggle environment debug messages",
    )
    return parser.parse_args()


def _find_latest_checkpoint(models_directory: Path) -> tuple[Path, int] | None:
    """Find the checkpoint with the greatest numeric model ID.

    Explanation:
        Examines only files matching the documented ``model_<id>.pt`` or
        ``model_<id>.pth`` convention. Numeric IDs, rather than timestamps or
        lexicographic filename order, determine which model is latest.

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
        Initializes the standard actor-critic on CPU. When a checkpoint is
        supplied, it accepts either a raw PyTorch state dictionary or a wrapper
        containing ``model_state_dict``. If no checkpoint exists, the model's
        normal random orthogonal initialization is retained.

    Args:
        checkpoint_path: Selected checkpoint path, or ``None`` for random
            initialization.

    Returns:
        Model in evaluation mode on the configured generation device.
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
        state_dict = cast(Mapping[str, Tensor], checkpoint)
        model.load_state_dict(state_dict)

    model.eval()
    return model


def _next_replay_id(output_directory: Path) -> int:
    """Find the next unused sequential replay ID.

    Explanation:
        Scans existing files matching ``game_<id>.json`` and returns one above
        the greatest ID. Unrelated files are ignored, preventing accidental
        overwrite when generation is resumed.

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


def _save_replay(replay: Mapping[str, object], replay_path: Path) -> None:
    """Save one complete Kaggle replay as compact JSON.

    Explanation:
        Serializes the full result of ``env.toJSON()`` immediately after a game
        so already completed games remain available if a later game fails.

    Args:
        replay: Complete Kaggle environment replay object.
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
    """Generate self-play games using the latest available model.

    Explanation:
        Loads the greatest numbered checkpoint once, or creates one randomly
        initialized model when no checkpoint exists. Every game creates four
        independent Agent instances sharing those fixed weights, runs for at
        most ``MAX_STEPS``, and saves the official Kaggle replay JSON.

    Args:
        args: Parsed arguments from ``parse_args``.

    Returns:
        Paths of all replay files generated by this call.
    """
    latest_checkpoint = _find_latest_checkpoint(args.models_dir)
    if latest_checkpoint is None:
        checkpoint_path = None
        model_id = None
        output_directory_name = RANDOM_INITIALIZATION_DIRECTORY
        print("No checkpoint found; using randomly initialized weights.")
    else:
        checkpoint_path, model_id = latest_checkpoint
        output_directory_name = REPLAY_MODEL_DIRECTORY_TEMPLATE.format(
            model_id=model_id,
        )
        print(f"Using checkpoint: {checkpoint_path}")

    model = _load_model(checkpoint_path)
    output_directory = args.gameplays_dir / output_directory_name
    output_directory.mkdir(parents=True, exist_ok=True)
    first_replay_id = _next_replay_id(output_directory)
    generated_paths: list[Path] = []

    for offset in range(args.games):
        replay_id = first_replay_id + offset
        environment = make(
            ENVIRONMENT_NAME,
            configuration={EPISODE_STEPS_CONFIGURATION_KEY: MAX_STEPS},
            debug=args.debug,
        )
        agents = [Agent(model) for _ in range(NUM_PLAYERS)]
        environment.run(agents)

        replay_path = output_directory / REPLAY_FILENAME_TEMPLATE.format(
            game_id=replay_id,
        )
        _save_replay(environment.toJSON(), replay_path)
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
