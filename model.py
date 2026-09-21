from __future__ import annotations

import torch
from torch import Tensor, nn

from constants import *

class HungryGeeseActorCritic(nn.Module):
    """Small CNN actor-critic for a normalized Hungry Geese observation.

    The board input must use the channel-last layout described by
    ``BOARD_TENSOR_SHAPE`` and returned by ``observation_to_tensor``. Direction
    and action encodings use the constants defined in ``constants.py``.

    ``remaining`` is a normalized scalar for every sample. It is concatenated
    after the board has been reduced to ``BOARD_FEATURES`` learned features.

    The policy output contains three unnormalized logits in the order
    FORWARD, LEFT, RIGHT. The value output is an unconstrained scalar.
    """

    def __init__(self) -> None:
        super().__init__()

        # Padding is applied explicitly in forward because one normalized axis
        # has period 7 and the other has period 11. Both convolutions therefore
        # use zero built-in padding and preserve 11x11 after explicit padding.
        self.conv1 = nn.Conv2d(
            NUM_CHANNELS,
            CONV_CHANNELS,
            kernel_size=CONV_KERNEL_SIZE,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        self.conv2 = nn.Conv2d(
            CONV_CHANNELS,
            CONV_CHANNELS,
            kernel_size=CONV_KERNEL_SIZE,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        self.activation = nn.SiLU()
        self.board_projection = nn.Linear(
            FLATTENED_FEATURES,
            BOARD_FEATURES,
        )

        # Actor and critic share the CNN representation but use independent
        # heads so their final features can specialize for their own losses.
        self.actor_hidden = nn.Linear(FUSED_FEATURES, FUSED_FEATURES)
        self.actor_output = nn.Linear(FUSED_FEATURES, NUM_ACTIONS)
        self.critic_hidden = nn.Linear(FUSED_FEATURES, FUSED_FEATURES)
        self.critic_output = nn.Linear(FUSED_FEATURES, VALUE_OUTPUTS)

        self._initialize_parameters()

    @staticmethod
    def _periodic_pad(x: Tensor, direction: Tensor) -> Tensor:
        """Pad an 11x11 forward-facing view by one true toroidal cell.

        NORTH/SOUTH views have vertical period 7 and horizontal period 11.
        EAST/WEST views have vertical period 11 and horizontal period 7.
        For a period-7 axis represented by view offsets -5 through 5, the cell
        before index 0 is found at index 6 and the cell after index 10 at index
        4. A period-11 axis uses the ordinary opposite-edge indices 10 and 0.

        Args:
            x: Feature tensor with shape ``(batch, channels, 11, 11)``.
            direction: Absolute direction for each sample with shape ``(batch,)``.

        Returns:
            The correctly padded tensor with shape
            ``(batch, channels, 13, 13)``.
        """
        short_low = slice(
            SHORT_AXIS_LOW_PAD_INDEX,
            SHORT_AXIS_LOW_PAD_INDEX + PERIODIC_PAD_WIDTH,
        )
        short_high = slice(
            SHORT_AXIS_HIGH_PAD_INDEX,
            SHORT_AXIS_HIGH_PAD_INDEX + PERIODIC_PAD_WIDTH,
        )
        long_low = slice(
            LONG_AXIS_LOW_PAD_INDEX,
            LONG_AXIS_LOW_PAD_INDEX + PERIODIC_PAD_WIDTH,
        )
        long_high = slice(
            LONG_AXIS_HIGH_PAD_INDEX,
            LONG_AXIS_HIGH_PAD_INDEX + PERIODIC_PAD_WIDTH,
        )

        # NORTH/SOUTH: rows repeat every 7 cells; columns every 11 cells.
        north_south = torch.cat(
            (x[:, :, short_low, :], x, x[:, :, short_high, :]),
            dim=2,
        )
        north_south = torch.cat(
            (
                north_south[:, :, :, long_low],
                north_south,
                north_south[:, :, :, long_high],
            ),
            dim=3,
        )

        # EAST/WEST: rows repeat every 11 cells; columns every 7 cells.
        east_west = torch.cat(
            (x[:, :, long_low, :], x, x[:, :, long_high, :]),
            dim=2,
        )
        east_west = torch.cat(
            (
                east_west[:, :, :, short_low],
                east_west,
                east_west[:, :, :, short_high],
            ),
            dim=3,
        )

        north_or_south = ((direction == NORTH) | (direction == SOUTH)).view(
            -1,
            1,
            1,
            1,
        )
        return torch.where(north_or_south, north_south, east_west)

    def _initialize_parameters(self) -> None:
        """Apply standard orthogonal initialization for PPO networks."""
        hidden_layers = (
            self.conv1,
            self.conv2,
            self.board_projection,
            self.actor_hidden,
            self.critic_hidden,
        )
        for layer in hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=HIDDEN_INIT_GAIN)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.actor_output.weight, gain=ACTOR_OUTPUT_GAIN)
        if self.actor_output.bias is not None:
            nn.init.zeros_(self.actor_output.bias)
        nn.init.orthogonal_(self.critic_output.weight, gain=CRITIC_OUTPUT_GAIN)
        if self.critic_output.bias is not None:
            nn.init.zeros_(self.critic_output.bias)

    def forward(
        self,
        board: Tensor,
        remaining: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute policy logits and state values for a batch.

        Args:
            board: Channel-last board tensor with shape
                ``(batch, 11, 11, 13)``.
            remaining: Normalized remaining-time values with shape ``(batch,)``
                or ``(batch, 1)``.
            direction: Absolute direction codes with shape ``(batch,)`` or
                ``(batch, 1)``. Values must be integers from 0 through 3.

        Returns:
            ``(logits, value)`` where ``logits`` has shape ``(batch, 3)`` and
            ``value`` has shape ``(batch, 1)``.
        """
        if board.ndim != 4 or board.shape[1:] != BOARD_TENSOR_SHAPE:
            raise ValueError(f"board trailing dimensions must be {BOARD_TENSOR_SHAPE}")

        batch_size = board.shape[0]
        if remaining.numel() != batch_size:
            raise ValueError("remaining must contain one value per board")
        if direction.numel() != batch_size:
            raise ValueError("direction must contain one value per board")

        # Convert the helper's channel-last layout to PyTorch's channel-first
        # convolution layout. Both tensors remain on the board's device.
        x = board.permute(0, 3, 1, 2).contiguous()
        direction = direction.reshape(batch_size).to(
            device=x.device,
            dtype=torch.long,
        )

        if torch.any((direction < NORTH) | (direction > WEST)):
            raise ValueError(
                f"direction values must be integers from {NORTH} through {WEST}"
            )

        x = self._periodic_pad(x, direction)
        x = self.activation(self.conv1(x))
        x = self._periodic_pad(x, direction)
        x = self.activation(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        board_features = self.activation(self.board_projection(x))

        remaining = remaining.reshape(batch_size, 1).to(
            device=board_features.device,
            dtype=board_features.dtype,
        )
        remaining = remaining.clamp(
            MIN_REMAINING_FRACTION,
            MAX_REMAINING_FRACTION,
        )
        fused_features = torch.cat((board_features, remaining), dim=1)

        actor_features = self.activation(self.actor_hidden(fused_features))
        logits = self.actor_output(actor_features)

        critic_features = self.activation(self.critic_hidden(fused_features))
        value = self.critic_output(critic_features)

        return logits, value
