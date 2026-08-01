"""
A compact, dependency-free implementation of a Kolmogorov-Arnold Network
(KAN) layer using B-spline-parameterised edge functions, in the spirit of
the "efficient-kan" formulation (Liu et al., 2024 KAN paper; spline
implementation following the common efficient re-formulation used in the
community). No external KAN library required -- everything needed is built
on top of plain PyTorch so the project has no fragile extra dependency.

Each layer replaces a standard Linear+activation with a learnable
univariate spline function on every input-output edge, summed at each
output node -- this is what makes a KAN structurally different from an MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)  # (in_features, grid_size + 2*spline_order + 1)

        # Base (residual) linear weight, like a skip connection -- standard
        # in KAN implementations to stabilise training.
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def b_splines(self, x):
        """x: (batch, in_features) -> (batch, in_features, grid_size + spline_order)"""
        grid = self.grid  # (in_features, n_knots)
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k] + 1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases  # (batch, in_features, grid_size + spline_order)

    def forward(self, x):
        base_out = F.linear(F.silu(x), self.base_weight)
        spline_bases = self.b_splines(x)  # (batch, in, coeffs)
        spline_out = torch.einsum("bic,oic->bo", spline_bases, self.spline_weight)
        return base_out + spline_out


class KAN(nn.Module):
    """Stack of KANLinear layers forming the full network."""

    def __init__(self, layer_sizes, grid_size=5, spline_order=3):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(
                KANLinear(layer_sizes[i], layer_sizes[i + 1],
                          grid_size=grid_size, spline_order=spline_order)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
