from __future__ import annotations

import random
from typing import Mapping, Sequence

import torch

from constants import *
from model import HungryGeeseActorCritic
from utils import get_survival_mask, observation_to_tensor


class Agent:
    def __init__(self, model: HungryGeeseActorCritic) -> None:
        """Initialize an inference agent with a trained model.

        Explanation:
            Stores the supplied model in evaluation mode and initializes the
            remembered absolute direction for a new game.

        Args:
            model: Trained Hungry Geese actor-critic model used for inference.

        Returns:
            None.
        """
        self.model = model
        self.model.eval()
        self.previous_direction: int | None = None

    def __call__(
        self,
        observation: Mapping[str, object],
        configuration: object,
    ) -> str:
        """Choose an action for the current observation.

        Explanation:
            On the first turn, resets internal state and chooses among all four
            absolute directions using the survival simulation and nearest-food
            heuristic. On later turns, converts the observation for the model,
            samples one of FORWARD, LEFT, or RIGHT after survival masking, and
            converts that relative action back to an absolute direction.

        Args:
            observation: Current Kaggle Hungry Geese observation.
            configuration: Kaggle environment configuration, reserved for later
                branches of the agent.

        Returns:
            One of ``NORTH``, ``EAST``, ``SOUTH``, or ``WEST``.
        """
        step = int(observation["step"])  # type: ignore[arg-type]
        if step != 0:
            board, direction = observation_to_tensor(
                observation,
                self.previous_direction,
            )
            remaining_steps = max(MAX_STEPS - step, 0)
            remaining = min(
                max(
                    remaining_steps / MAX_STEPS,
                    MIN_REMAINING_FRACTION,
                ),
                MAX_REMAINING_FRACTION,
            )

            model_parameter = next(self.model.parameters())
            board_tensor = torch.from_numpy(board).unsqueeze(0).to(
                device=model_parameter.device,
                dtype=model_parameter.dtype,
            )
            remaining_tensor = torch.tensor(
                [remaining],
                device=model_parameter.device,
                dtype=model_parameter.dtype,
            )
            direction_tensor = torch.tensor(
                [direction],
                device=model_parameter.device,
                dtype=torch.long,
            )

            with torch.no_grad():
                action_logits, _ = self.model(
                    board_tensor,
                    remaining_tensor,
                    direction_tensor,
                )
            action_logits = action_logits[0]

            absolute_directions = tuple(
                (direction + offset) % len(VALID_DIRECTIONS)
                for offset in RELATIVE_DIRECTION_OFFSETS
            )
            survival_steps = (
                SURVIVAL_HORIZON
                if remaining_steps > SURVIVAL_HORIZON
                else FALLBACK_SURVIVAL_HORIZON
            )
            survival_mask = get_survival_mask(
                observation,
                current_direction=direction,
                steps=survival_steps,
            )
            available_actions = [
                action
                for action, absolute_direction in zip(
                    VALID_ACTIONS,
                    absolute_directions,
                )
                if survival_mask[absolute_direction]
            ]

            if not available_actions and survival_steps == SURVIVAL_HORIZON:
                survival_mask = get_survival_mask(
                    observation,
                    current_direction=direction,
                    steps=FALLBACK_SURVIVAL_HORIZON,
                )
                available_actions = [
                    action
                    for action, absolute_direction in zip(
                        VALID_ACTIONS,
                        absolute_directions,
                    )
                    if survival_mask[absolute_direction]
                ]

            if not available_actions:
                chosen_action = random.choice(VALID_ACTIONS)
            else:
                action_mask = torch.tensor(
                    [action in available_actions for action in VALID_ACTIONS],
                    device=action_logits.device,
                    dtype=torch.bool,
                )
                masked_logits = action_logits.masked_fill(
                    ~action_mask,
                    MASKED_LOGIT,
                )
                distribution = torch.distributions.Categorical(
                    logits=masked_logits,
                )
                chosen_action = int(distribution.sample().item())

            chosen_direction = absolute_directions[chosen_action]
            self.previous_direction = chosen_direction
            return DIRECTION_NAMES[chosen_direction]

        self.previous_direction = None

        survival_mask = get_survival_mask(
            observation,
            current_direction=None,
            steps=SURVIVAL_HORIZON,
        )
        available_directions = [
            direction
            for direction in VALID_DIRECTIONS
            if survival_mask[direction]
        ]

        if not available_directions:
            survival_mask = get_survival_mask(
                observation,
                current_direction=None,
                steps=FALLBACK_SURVIVAL_HORIZON,
            )
            available_directions = [
                direction
                for direction in VALID_DIRECTIONS
                if survival_mask[direction]
            ]

        food: Sequence[int] = observation["food"]  # type: ignore[assignment]
        if not available_directions:
            chosen_direction = random.choice(VALID_DIRECTIONS)
        elif not food:
            chosen_direction = random.choice(available_directions)
        else:
            geese: Sequence[Sequence[int]] = observation["geese"]  # type: ignore[assignment]
            index = int(observation["index"])  # type: ignore[arg-type]
            head_row, head_col = divmod(geese[index][0], COLS)

            best_distance: int | None = None
            closest_directions: list[int] = []
            for direction in available_directions:
                row_delta, col_delta = DIRECTION_DELTAS[direction]
                next_row = (head_row + row_delta) % ROWS
                next_col = (head_col + col_delta) % COLS

                nearest_food_distance = min(
                    min(abs(next_row - food_row), ROWS - abs(next_row - food_row))
                    + min(abs(next_col - food_col), COLS - abs(next_col - food_col))
                    for food_row, food_col in (
                        divmod(food_position, COLS)
                        for food_position in food
                    )
                )

                if best_distance is None or nearest_food_distance < best_distance:
                    best_distance = nearest_food_distance
                    closest_directions = [direction]
                elif nearest_food_distance == best_distance:
                    closest_directions.append(direction)

            chosen_direction = random.choice(closest_directions)

        self.previous_direction = chosen_direction
        return DIRECTION_NAMES[chosen_direction]
