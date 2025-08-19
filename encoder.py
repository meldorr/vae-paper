from tcn import TCN
import torch.nn as nn
from typing import List, Optional

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.sampling_factor = sampling_factor

        self.tcn = TCN(
            input_dim=input_dim,
            out_dim=out_dim,
            h_dims=h_dims,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            h_activ=h_activ,
            dropout=dropout,
        )

    def forward(self, x):
        return nn.Sequential(
            self.tcn,
            nn.AvgPool1d(self.sampling_factor),
            nn.Flatten(),
        )(x)
    
