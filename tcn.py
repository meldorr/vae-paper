from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    """Temporal Block, a causal and dilated convolution with activation
    and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        out_activ: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
            )
        )

        self.out_activ = out_activ
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0), "constant", 0)
        x = self.conv(x)
        x = self.out_activ(x) if self.out_activ is not None else x
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        h_activ: Optional[nn.Module] = None,
        is_last: bool = False,
    ) -> None:
        super().__init__()

        self.is_last = is_last

        self.tmp_block1 = TemporalBlock(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ,
        )

        self.tmp_block2 = TemporalBlock(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ if not is_last else None,  # deactivate last activation
        )

        # Optional convolution for matching in_channels and out_channels sizes
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        # self.out_activ = h_activ
        self.init_weights()

    def init_weights(self) -> None:
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.tmp_block1(x)
        y = self.tmp_block2(y)
        r = x if self.downsample is None else self.downsample(x)
        return y + r

class TCN(nn.Module):

    """Temporal Convolutional Network.

    Handle data shaped :math:`(N, C, L)` with :math:`N` the batch size,
    :math:`C` the number of channels and :math:`L` the length of signal
    sequence.

    Args:
        input_dim (int): number of channels in the input sequence.
        out_dim (int): number of channels produced by the network.
        h_dims (List[int]): number of channels outputed by the hidden layers.
        kernel_size (int): size of the convolving kernel.
        dilation_base (int): size of the dilation base to compute the dilation
            of a specific layer.
        activation (torch.nn.Module): activation function to use within
            temporal blocks.
        dropout (float, optional): probability of an element to be zeroed.
            Default :math:`0.2`.

    .. note::
        Keep the kernel size at least as big as the dilation base to avoid any
        holes in your receptive field i.e.

        .. math::
            \\text{kernel_size} \\geq \\text{dilatation_base}
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        layers = []

        for index in range(self.n_layers):
            dilation = dilation_base ** index
            in_channels = layer_dims[index]
            out_channels = layer_dims[index + 1]
            is_last = index == (self.n_layers - 1)
            layer = ResidualBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout,
                h_activ,
                is_last,
            )
            layers.append(layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
if __name__ == "__main__":
    tn = TCN(
        input_dim=4,
        out_dim=16,
        h_dims=[32, 64],
        kernel_size=3,
        dilation_base=2,
        h_activ=None,
        dropout=0.2)

    print(tn)
    