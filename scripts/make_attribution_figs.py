import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import (
    DeepLift,
    FeatureAblation,
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
)
from spender.data.desi import DESI
from tqdm import tqdm

from outlier_attribution.model import OutlierModel

device = "cuda" if torch.cuda.is_available() else "cpu"

weight_dir = "/scratch/gpfs/js5013/programs/outlier-attribution/weights"
model = OutlierModel.from_weights(
    f"{weight_dir}/spender.desi-edr.galaxyae-b9bc8d12.pt",
    f"{weight_dir}/galaxy-flow-state_dict.pt",
).to(device)

with open("../data/DESI_EDR_top200_outliers.pkl", "rb") as f:
    data = pickle.load(f)

spectra = torch.tensor(np.stack([data[i]["spectrum"] for i in range(1, 201)])).to(
    device
)

ig = IntegratedGradients(model)
ig_nt = NoiseTunnel(ig)
dl = DeepLift(model)
fa = FeatureAblation(model)
oc = Occlusion(model)

attribution_labels = [
    "Integrated Gradients",
    "Noise Tunnel",
    "DeepLift",
    "Feature Ablation",
    "Occlusion",
]


def make_attribution_fig(spec, attribution, title, save_name):

    fig, ax = plt.subplots(
        6, 1, figsize=(10, 15), sharex=True, gridspec_kw=dict(hspace=0)
    )

    ax[0].plot(
        DESI._wave_obs,
        spec.detach().cpu().numpy(),
    )
    [
        ax[j + 1].plot(
            DESI._wave_obs,
            attribution[j].detach().cpu().numpy(),  # attribution is 5 x n_spec
            label=attribution_labels[j],
        )
        for j in range(5)
    ]
    ax[0].set_ylabel("Data")
    [ax[j + 1].set_ylabel(attribution_labels[j]) for j in range(5)]
    for a in ax:
        a.set_yticks([])
    plt.xlabel("Wavelength (A)")
    plt.suptitle(title, fontsize=16)
    plt.savefig(save_name, dpi=250)


batch_size = 5
ctr = 0
total_size = 200

pbar = tqdm(total=total_size)

while ctr < total_size:

    ig_attr = ig.attribute(spectra[ctr : ctr + batch_size], n_steps=50)
    ig_nt_attr = ig_nt.attribute(spectra[ctr : ctr + batch_size])
    dl_attr = dl.attribute(spectra[ctr : ctr + batch_size])
    fa_attr = fa.attribute(spectra[ctr : ctr + batch_size])
    oc_attr = oc.attribute(spectra[ctr : ctr + batch_size], sliding_window_shapes=(16,))

    attributions = torch.stack(
        [ig_attr, ig_nt_attr, dl_attr, fa_attr, oc_attr]
    )  # 5 x batch_size x n_spec

    cur_batch_size = attributions.shape[1]

    for i in range(cur_batch_size):
        make_attribution_fig(
            spectra[ctr + i],
            attributions[:, i],
            f"Outlier {ctr + i + 1}",
            f"../figures/attribution/outlier_{ctr + i + 1}.png",
        )
        pbar.update(1)

    ctr += cur_batch_size
