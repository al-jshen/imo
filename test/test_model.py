import torch
from outlier_attribution.flow import NeuralDensityEstimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from outlier_attribution.utils import load_desi_model

desi, ae = load_desi_model("./weights/spender.desi-edr.galaxyae-b9bc8d12.pt")
ae = ae.to(device)
n_latent = ae.encoder.n_latent
n_observed = len(desi._wave_obs)

fake_spec = torch.randn(size=(1, n_observed)).to(device)
fake_latent = torch.randn(size=(1, n_latent)).to(device)

NDE_theta = NeuralDensityEstimator(
    normalize=False,
    initial_pos={"bounds": [[0, 0]] * n_latent, "std": [0.05] * n_latent},
    method="maf",
)
NDE_theta.build(fake_latent)

NDE_theta.net.load_state_dict(torch.load("./weights/galaxy-flow-state_dict.pt"))

from outlier_attribution.model import OutlierModel

model = OutlierModel(ae, NDE_theta).to(device)

print(model)

print(model(fake_spec))
