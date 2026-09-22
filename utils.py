from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from constants import *

def observation_to_tensor(
    observation: Mapping[str, object],
    previous_direction: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Convert an observation to a centered, forward-facing one-hot tensor.

    Explanation:
        The 7x11 toroidal board is translated so the controlled goose's head is
        at position ``(5, 5)``, rotated so its current direction points toward
        the top of the tensor, and periodically repeated to fill an 11x11 view.
        The result has shape ``(11, 11, 13)`` and dtype ``float32``.

        Channel meanings are:

        - 0: controlled goose's head
        - 1-3: opponent heads
        - 4: controlled goose's tail
        - 5-7: opponent tails
        - 8: controlled goose's body (head and tail excluded)
        - 9-11: opponent bodies (heads and tails excluded)
        - 12: food

        Opponents are assigned to channels in ascending original player-index
        order after excluding the controlled player. Direction encoding is
        ``NORTH=0``, ``EAST=1``, ``SOUTH=2``, and ``WEST=3``.

    Args:
        observation: Current Kaggle observation containing ``index``, ``geese``,
            ``food``, and ``step``.
        previous_direction: Last successfully executed absolute direction,
            encoded as ``NORTH=0``, ``EAST=1``, ``SOUTH=2``, or ``WEST=3``.
            It is used only when the controlled goose has length one and its
            direction therefore cannot be inferred from the current observation.

    Returns:
        A pair ``(tensor, direction)``. ``tensor`` is the normalized one-hot
        array with shape ``(11, 11, 13)``; ``direction`` is the unrotated
        original-board self-moving direction encoded as an integer from 0 through 3.

    Raises:
        ValueError: If ``index`` is invalid, the controlled goose is dead, or
            the observation does not describe exactly four geese. Also raised
            when the current body is inconsistent, or when direction is
            ambiguous after the first turn and ``previous_direction`` is absent.
    """
    def _direction_between(old_position: int, new_position: int) -> Optional[int]:
        """Find the direction of one move on the toroidal board.

        Explanation:
            Converts both flattened positions to row-column coordinates and checks
            whether ``new_position`` is one legal cardinal step from
            ``old_position``. Wrap-around at every board edge is included.

        Args:
            old_position: Flattened board position before the move.
            new_position: Flattened board position after the move.

        Returns:
            The encoded direction (``NORTH=0``, ``EAST=1``, ``SOUTH=2``, or
            ``WEST=3``), or ``None`` when the positions are not one step apart.
        """
        old_row, old_col = divmod(old_position, COLS)
        new_row, new_col = divmod(new_position, COLS)

        if new_col == old_col:
            if new_row == (old_row - 1) % ROWS:
                return NORTH
            if new_row == (old_row + 1) % ROWS:
                return SOUTH
        if new_row == old_row:
            if new_col == (old_col + 1) % COLS:
                return EAST
            if new_col == (old_col - 1) % COLS:
                return WEST
        return None


    def _infer_direction(
        observation: Mapping[str, object],
        previous_direction: Optional[int],
    ) -> int:
        """Infer the current goose's most recent movement direction.

        Explanation:
            Direction is normally inferred from the current head and neck. A
            length-one goose has no neck, so its last known direction is used
            instead. Only the first turn may default to NORTH because no move
            has yet been made and there is no reverse direction to avoid.

        Args:
            observation: Current Kaggle Hungry Geese observation.
            previous_direction: Last successfully executed absolute direction,
                or ``None`` if no previous move exists.

        Returns:
            The encoded direction as an integer from 0 through 3.
        """
        index = int(observation["index"]) # type: ignore[arg-type]
        goose = observation["geese"][index]  # type: ignore[index]

        # The neck occupies the square from which the head just moved.
        if len(goose) >= 2:
            direction = _direction_between(goose[1], goose[0])
            if direction is None:
                raise ValueError("controlled goose's head and neck are not adjacent")
            return direction

        if previous_direction is not None:
            if previous_direction not in VALID_DIRECTIONS:
                raise ValueError("previous_direction must be NORTH, EAST, SOUTH, or WEST")
            return int(previous_direction)

        step = int(observation["step"])  # type: ignore[arg-type]
        if step == 0:
            return NORTH

        raise ValueError(
            "previous_direction is required for a length-one goose after step 0"
        )


    def _view_offset_to_board_offset(row: int, col: int, direction: int) -> Tuple[int, int]:
        """Map a forward-facing view cell back to an original-board offset.

        Explanation:
            The normalized view always places forward toward NORTH. This function
            reverses that rotation so a location in the 11x11 view can be sampled
            from the unrotated 7x11 board.

        Args:
            row: Row index in the normalized 11x11 view.
            col: Column index in the normalized 11x11 view.
            direction: Goose direction encoded as an integer from 0 through 3.

        Returns:
            A ``(row_offset, column_offset)`` pair relative to the goose's head on
            the original board.
        """
        forward_row = row - VIEW_CENTER
        forward_col = col - VIEW_CENTER

        if direction == NORTH:
            return forward_row, forward_col
        if direction == EAST:
            return forward_col, -forward_row
        if direction == SOUTH:
            return -forward_row, -forward_col
        return -forward_col, forward_row  # WEST

    geese: Sequence[Sequence[int]] = observation["geese"]  # type: ignore[assignment]
    food: set[int] = set(observation["food"])  # type: ignore[arg-type]
    index = int(observation["index"]) # type: ignore[arg-type]

    if not 0 <= index < len(geese):
        raise ValueError("observation['index'] does not identify a goose")
    if not geese[index]:
        raise ValueError("cannot center the view on a dead goose")

    direction = _infer_direction(observation, previous_direction)
    opponents = [player for player in range(len(geese)) if player != index]
    if len(opponents) != NUM_OPPONENTS:
        raise ValueError(f"expected exactly {NUM_PLAYERS} geese")

    head_channels = {
        goose[0]: OPPONENT_HEAD_CHANNEL_START + slot
        for slot, player in enumerate(opponents)
        if (goose := geese[player])
    }
    tail_channels = {
        goose[-1]: OPPONENT_TAIL_CHANNEL_START + slot
        for slot, player in enumerate(opponents)
        if len(goose := geese[player]) >= 2
    }
    body_channels = {
        position: OPPONENT_BODY_CHANNEL_START + slot
        for slot, player in enumerate(opponents)
        for position in geese[player][1:-1]
    }
    own_head = geese[index][0]
    own_tail = geese[index][-1] if len(geese[index]) >= 2 else None
    own_body = set(geese[index][1:-1])
    head_row, head_col = divmod(own_head, COLS)

    tensor = np.zeros((VIEW_SIZE, VIEW_SIZE, NUM_CHANNELS), dtype=np.float32)
    for view_row in range(VIEW_SIZE):
        for view_col in range(VIEW_SIZE):
            row_offset, col_offset = _view_offset_to_board_offset(
                view_row, view_col, direction
            )
            board_row = (head_row + row_offset) % ROWS
            board_col = (head_col + col_offset) % COLS
            position = board_row * COLS + board_col

            if position == own_head:
                tensor[view_row, view_col, OWN_HEAD_CHANNEL] = 1.0
            if position in head_channels:
                tensor[view_row, view_col, head_channels[position]] = 1.0
            if position == own_tail:
                tensor[view_row, view_col, OWN_TAIL_CHANNEL] = 1.0
            if position in tail_channels:
                tensor[view_row, view_col, tail_channels[position]] = 1.0
            if position in own_body:
                tensor[view_row, view_col, OWN_BODY_CHANNEL] = 1.0
            if position in body_channels:
                tensor[view_row, view_col, body_channels[position]] = 1.0
            if position in food:
                tensor[view_row, view_col, FOOD_CHANNEL] = 1.0

    return tensor, direction


def get_survival_mask(
    observation: Mapping[str, object],
    current_direction: Optional[int],
    steps: int = SURVIVAL_HORIZON,
) -> Tuple[bool, bool, bool, bool]:
    """Find absolute directions with an assumed survival path.

    Explanation:
        Tests every absolute first direction in NORTH, EAST, SOUTH, WEST order.
        A direction remains available only if the controlled goose survives its
        first move and at least one legal continuation survives until ``steps``
        moves have been simulated.

        Collision is checked against every body cell occupied at the start of
        each simulated move, including tails that disappear later in that move.
        After a safe move, the controlled goose's tail leaves unless it eats a
        currently known food, and it loses an additional tail segment on hunger
        turns. Each opponent head stays fixed while one opponent tail segment
        disappears per simulated move. Opponent movement, food replenishment,
        and possible future head-to-head attacks are intentionally ignored.

        ``True`` means the direction has a survival path and should remain
        selectable. ``False`` means it should be masked. When
        ``current_direction`` is ``None`` all four directions are considered;
        otherwise its immediate reverse is always masked.

    Args:
        observation: Current Kaggle observation containing ``index``, ``step``,
            ``geese``, and ``food``.
        current_direction: Current absolute direction, or ``None`` on the first
            turn before any direction has been chosen.
        steps: Number of future moves that must contain a survival path.

    Returns:
        Four booleans in NORTH, EAST, SOUTH, WEST order. A true entry indicates
        that the corresponding absolute direction has an assumed survival path.

    Raises:
        ValueError: If the observation is invalid, ``steps`` is not positive,
            or direction is missing after the first turn.
    """
    def _next_position(position: int, direction: int) -> int:
        """Move one cell in an absolute direction on the toroidal board.

        Explanation:
            Applies the direction's row-column delta and wraps both coordinates
            around the fixed board dimensions.

        Args:
            position: Flattened starting position.
            direction: Absolute direction encoded from NORTH through WEST.

        Returns:
            Flattened destination position after one wrapped move.
        """
        row, col = divmod(position, COLS)
        row_delta, col_delta = DIRECTION_DELTAS[direction]
        next_row = (row + row_delta) % ROWS
        next_col = (col + col_delta) % COLS
        return next_row * COLS + next_col

    def _advance_opponents(
        opponent_geese: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Advance opponents according to the simplified stationary-head model.

        Explanation:
            Keeps every living opponent head in place and removes one tail
            segment when that opponent has a body segment to remove.

        Args:
            opponent_geese: Opponent bodies with each head stored first.

        Returns:
            Copied opponent bodies after one simplified simulated move.
        """
        return [
            list(goose[:-1]) if len(goose) >= 2 else list(goose)
            for goose in opponent_geese
        ]

    def _simulate_move(
        own_goose: Sequence[int],
        opponent_geese: Sequence[Sequence[int]],
        food: set[int],
        direction: int,
        next_step: int,
    ) -> Optional[Tuple[list[int], list[list[int]], set[int]]]:
        """Simulate one controlled move and its simplified surroundings.

        Explanation:
            Finds the controlled goose's destination and first rejects a
            collision with any body cell occupied at the start of the move.
            This includes own and opponent tails even when they disappear later
            in the same turn. After that check, it updates growth and hunger and
            advances the simplified opponent bodies.

        Args:
            own_goose: Current controlled body with its head first.
            opponent_geese: Current simplified opponent bodies.
            food: Currently known uneaten food positions.
            direction: Absolute direction chosen for this move.
            next_step: Observation step reached after this move.

        Returns:
            Updated own body, opponent bodies, and food set when the controlled
            goose survives; otherwise ``None``.
        """
        new_head = _next_position(own_goose[0], direction)
        occupied_before_tail_removal = {
            position
            for goose in (own_goose, *opponent_geese)
            for position in goose
        }
        if new_head in occupied_before_tail_removal:
            return None

        next_own_goose = [new_head, *own_goose]
        next_food = set(food)

        if new_head in next_food:
            next_food.remove(new_head)
        else:
            next_own_goose.pop()

        if next_step % HUNGER_RATE == 0:
            next_own_goose.pop()
            if not next_own_goose:
                return None

        next_opponents = _advance_opponents(opponent_geese)
        return next_own_goose, next_opponents, next_food

    def _has_survival_path(
        own_goose: Sequence[int],
        opponent_geese: Sequence[Sequence[int]],
        food: set[int],
        last_direction: int,
        step: int,
        remaining_steps: int,
    ) -> bool:
        """Check whether any legal continuation survives the remaining depth.

        Explanation:
            Performs a small existential search. At every depth it excludes the
            immediate reverse direction and succeeds as soon as one continuation
            reaches the requested horizon alive.

        Args:
            own_goose: Simulated controlled body with its head first.
            opponent_geese: Simulated opponent bodies.
            food: Simulated set of known uneaten food.
            last_direction: Absolute direction used on the preceding move.
            step: Current simulated observation step.
            remaining_steps: Number of additional moves that must be survived.

        Returns:
            ``True`` if at least one legal continuation survives the horizon.
        """
        if remaining_steps == 0:
            return True

        reverse_direction = (last_direction + 2) % len(VALID_DIRECTIONS)
        for direction in VALID_DIRECTIONS:
            if direction == reverse_direction:
                continue

            result = _simulate_move(
                own_goose,
                opponent_geese,
                food,
                direction,
                step + 1,
            )
            if result is None:
                continue

            next_own_goose, next_opponents, next_food = result
            if _has_survival_path(
                next_own_goose,
                next_opponents,
                next_food,
                direction,
                step + 1,
                remaining_steps - 1,
            ):
                return True

        return False

    if steps <= 0:
        raise ValueError("steps must be positive")

    geese: Sequence[Sequence[int]] = observation["geese"]  # type: ignore[assignment]
    index = int(observation["index"])  # type: ignore[arg-type]
    step = int(observation["step"])  # type: ignore[arg-type]
    food: set[int] = set(observation["food"])  # type: ignore[arg-type]

    if len(geese) != NUM_PLAYERS:
        raise ValueError(f"expected exactly {NUM_PLAYERS} geese")
    if not 0 <= index < len(geese):
        raise ValueError("observation['index'] does not identify a goose")
    if not geese[index]:
        raise ValueError("cannot simulate a dead goose")
    if current_direction is None:
        if step != 0:
            raise ValueError("current_direction is required after step 0")
    elif current_direction not in VALID_DIRECTIONS:
        raise ValueError("current_direction must be NORTH, EAST, SOUTH, or WEST")

    own_goose = list(geese[index])
    opponent_geese = [
        list(goose)
        for player, goose in enumerate(geese)
        if player != index
    ]
    reverse_direction = (
        None
        if current_direction is None
        else (current_direction + 2) % len(VALID_DIRECTIONS)
    )

    mask = []
    for direction in VALID_DIRECTIONS:
        if direction == reverse_direction:
            mask.append(False)
            continue

        result = _simulate_move(
            own_goose,
            opponent_geese,
            food,
            direction,
            step + 1,
        )
        if result is None:
            mask.append(False)
            continue

        next_own_goose, next_opponents, next_food = result
        mask.append(
            _has_survival_path(
                next_own_goose,
                next_opponents,
                next_food,
                direction,
                step + 1,
                steps - 1,
            )
        )

    return mask[NORTH], mask[EAST], mask[SOUTH], mask[WEST]
