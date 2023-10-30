import torch
from imo.flow import NeuralDensityEstimator
from imo.utils import load_desi_model


device = "cuda" if torch.cuda.is_available() else "cpu"


class OutlierModel(torch.nn.Module):
    """Wrapper around autoencoder and normalizing flow models."""

    def __init__(self, autoencoder, flow):
        super().__init__()
        self.autoencoder = autoencoder
        self.flow = flow

    def forward(self, x):
        s = self.autoencoder.encode(x)
        lp = self.flow.net.log_prob(s)
        return lp

    def reconstruct(self, x, z):
        return self.autoencoder(x.unsqueeze(0), z=z).squeeze(0)

    @classmethod
    def from_weights(cls, ae_weights, flow_weights, device=device):

        _, ae = load_desi_model(ae_weights, map_location=torch.device(device))
        ae = ae.to(device)
        n_latent = ae.encoder.n_latent

        fake_latent = torch.randn(size=(1, n_latent)).to(device)

        nde = NeuralDensityEstimator(
            normalize=False,
            initial_pos={"bounds": [[0, 0]] * n_latent, "std": [0.05] * n_latent},
            method="maf",
        )
        nde.build(fake_latent)

        nde.net.load_state_dict(torch.load(flow_weights))

        model = cls(ae, nde).to(device)

        return model
