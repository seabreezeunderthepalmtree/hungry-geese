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
            remembered absolute direction and PPO decision history for a new
            game.

        Args:
            model: Trained Hungry Geese actor-critic model used for inference.

        Returns:
            None.
        """
        self.model = model
        self.model.eval()
        self.previous_direction: int | None = None
        self.ppo_trajectory: list[dict[str, object]] = []

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
                action_logits, state_value = self.model(
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

            action_mask = torch.tensor(
                [action in available_actions for action in VALID_ACTIONS],
                device=action_logits.device,
                dtype=torch.bool,
            )
            if not available_actions:
                chosen_action = random.choice(VALID_ACTIONS)
                policy_sampled = False
            else:
                masked_logits = action_logits.masked_fill(
                    ~action_mask,
                    MASKED_LOGIT,
                )
                distribution = torch.distributions.Categorical(
                    logits=masked_logits,
                )
                chosen_action = int(distribution.sample().item())
                policy_sampled = True

            self.ppo_trajectory.append(
                {
                    "step": step,
                    "relative_action": chosen_action,
                    "action_logit_vec": [
                        float(logit)
                        for logit in action_logits.detach().cpu().tolist()
                    ],
                    "value": float(state_value.reshape(-1)[0].item()),
                    "action_mask": [
                        bool(is_available)
                        for is_available in action_mask.detach().cpu().tolist()
                    ],
                    "policy_sampled": policy_sampled,
                }
            )

            chosen_direction = absolute_directions[chosen_action]
            self.previous_direction = chosen_direction
            return DIRECTION_NAMES[chosen_direction]

        self.previous_direction = None
        self.ppo_trajectory = []

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

class SimpleAgent:
    actions = ["NORTH", "SOUTH", "EAST", "WEST"]
    opposite = {
        "NORTH": "SOUTH",
        "SOUTH": "NORTH",
        "EAST": "WEST",
        "WEST": "EAST",
    }
    move = {
        "NORTH": (-1, 0),
        "SOUTH": (1, 0),
        "EAST": (0, 1),
        "WEST": (0, -1),
    }

    def __init__(self) -> None:
        """Initialize the fixed heuristic opponent.

        Explanation:
            Starts without a previous action so every direction is available on
            the first turn of a game.

        Args:
            None.

        Returns:
            None.
        """
        self.last_action: str | None = None

    def __call__(self, observation, configuration) -> str:
        """Choose an absolute action with a small fixed heuristic.

        Explanation:
            Applies the shared two-step survival mask, falling back to a
            one-step mask when every direction is rejected. Among surviving
            directions, it strongly discourages moves an opponent head could
            also reach, prefers moves toward the nearest food, and uses a small
            random tie-breaker. Internal state is reset whenever a new game
            starts at step zero.

        Args:
            observation: Current Kaggle Hungry Geese observation.
            configuration: Kaggle environment configuration containing board
                dimensions.

        Returns:
            One of ``NORTH``, ``SOUTH``, ``EAST``, or ``WEST``.
        """
        if observation["step"] == 0:
            self.last_action = None

        rows = configuration["rows"]
        cols = configuration["columns"]

        geese = observation["geese"]
        food = observation["food"]
        my_index = observation["index"]

        my_goose = geese[my_index]

        if not my_goose:
            return "NORTH"

        my_head = my_goose[0]

        current_direction = (
            None
            if self.last_action is None
            else DIRECTION_NAMES.index(self.last_action)
        )
        remaining_steps = max(MAX_STEPS - observation["step"], 0)
        survival_steps = (
            SURVIVAL_HORIZON
            if remaining_steps > SURVIVAL_HORIZON
            else FALLBACK_SURVIVAL_HORIZON
        )
        survival_mask = get_survival_mask(
            observation,
            current_direction=current_direction,
            steps=survival_steps,
        )
        available_actions = [
            action
            for action in self.actions
            if survival_mask[DIRECTION_NAMES.index(action)]
        ]

        if not available_actions and survival_steps == SURVIVAL_HORIZON:
            survival_mask = get_survival_mask(
                observation,
                current_direction=current_direction,
                steps=FALLBACK_SURVIVAL_HORIZON,
            )
            available_actions = [
                action
                for action in self.actions
                if survival_mask[DIRECTION_NAMES.index(action)]
            ]

        def to_rc(pos):
            """Convert a flattened board position to row and column.

            Explanation:
                Uses the configured column count to decode a board position.

            Args:
                pos: Flattened board position.

            Returns:
                ``(row, column)`` coordinates.
            """
            return pos // cols, pos % cols

        def to_pos(r, c):
            """Convert wrapped row and column coordinates to a position.

            Explanation:
                Wraps both axes to preserve the toroidal board geometry before
                flattening the coordinates.

            Args:
                r: Possibly unwrapped row coordinate.
                c: Possibly unwrapped column coordinate.

            Returns:
                Flattened wrapped board position.
            """
            return (r % rows) * cols + (c % cols)

        def next_pos(pos, action):
            """Find the wrapped destination of one absolute action.

            Explanation:
                Applies this class's hard-coded direction offset and converts
                the result back to a flattened board position.

            Args:
                pos: Flattened starting position.
                action: Absolute direction name.

            Returns:
                Flattened destination position.
            """
            r, c = to_rc(pos)
            dr, dc = self.move[action]
            return to_pos(r + dr, c + dc)

        def toroidal_distance(a, b):
            """Calculate toroidal Manhattan distance between two positions.

            Explanation:
                Uses the shorter wrapped displacement on each board axis.

            Args:
                a: First flattened board position.
                b: Second flattened board position.

            Returns:
                Toroidal Manhattan distance.
            """
            ar, ac = to_rc(a)
            br, bc = to_rc(b)

            dr = min(abs(ar - br), rows - abs(ar - br))
            dc = min(abs(ac - bc), cols - abs(ac - bc))

            return dr + dc

        # 敌方鹅头下一步可能到达的位置
        enemy_head_danger = set()

        for i, goose in enumerate(geese):
            if i == my_index or not goose:
                continue

            enemy_head = goose[0]

            for action in self.actions:
                enemy_head_danger.add(
                    next_pos(enemy_head, action)
                )

        candidates = []

        for action in available_actions:
            new_pos = next_pos(my_head, action)

            score = 0

            # 避开敌方鹅头附近
            if new_pos in enemy_head_danger:
                score -= 100

            # 靠近最近食物
            if food:
                old_dist = min(
                    toroidal_distance(my_head, f)
                    for f in food
                )

                new_dist = min(
                    toroidal_distance(new_pos, f)
                    for f in food
                )

                if new_dist < old_dist:
                    score += 10
                elif new_dist > old_dist:
                    score -= 2

            # 给一点随机扰动，避免四只完全同步
            score += random.random()

            candidates.append((score, action))

        if candidates:
            candidates.sort(reverse=True)
            action = candidates[0][1]

        else:
            # 实在没安全路了
            fallback = self.actions.copy()

            if self.last_action is not None:
                fallback.remove(self.opposite[self.last_action])

            action = random.choice(fallback)

        self.last_action = action
        return action
