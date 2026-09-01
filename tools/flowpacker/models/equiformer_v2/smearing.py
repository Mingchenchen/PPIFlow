import torch

# Different encodings for the atom distance embeddings
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))