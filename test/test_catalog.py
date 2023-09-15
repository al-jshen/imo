import numpy as np
import torch
from outlier_attribution.model import OutlierModel
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"

with open("../data/DESI_EDR_top200_outliers.pkl", "rb") as f:
    data = pickle.load(f)


model = OutlierModel.from_weights(
    "../weights/spender.desi-edr.galaxyae-b9bc8d12.pt",
    "../weights/galaxy-flow-state_dict.pt",
)


spectra = torch.tensor(np.stack([data[i]["spectrum"] for i in range(1, 201)])).to(
    device
)
lps_catalog = -torch.tensor(np.array([data[i]["-logP"] for i in range(1, 201)])).to(
    device
)

lps = model(spectra)
print(lps)
print(lps_catalog)

print(torch.allclose(lps, lps_catalog, atol=1e-2, rtol=1e-2))
