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

from imo.model import OutlierModel
import warnings

warnings.filterwarnings("ignore")

plt.style.use("js")

device = "cuda" if torch.cuda.is_available() else "cpu"

trainloader = DESI.get_data_loader(
    "/scratch/gpfs/yanliang/desi-dynamic",
    which="train",
    batch_size=256,
    shuffle=True,
    shuffle_instance=True,
)

get_baselines = lambda: next(iter(trainloader))[0].to(device).reshape(256, 1, -1)

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

selkeys = ["target_id", "z", "ra", "dec", "-logP"]
metadata = dict(
    zip(
        selkeys,
        (
            torch.tensor(np.stack([data[i][k] for i in range(1, 201)])).to(device)
            for k in selkeys
        ),
    )
)

ig = IntegratedGradients(model)
ig_nt = NoiseTunnel(ig)
dl = DeepLift(model)
fa = FeatureAblation(model)
oc = Occlusion(model)
gs = GradientShap(model)

attribution_labels = [
    "Integrated Gradients",
    "Expected Gradients",
    "Noise Tunnel",
    "DeepLift",
    "Feature Ablation",
    "Occlusion",
    "Gradient SHAP",
]


@torch.compile
def expected_attributes(spectrum, baselines):
    attributions = []
    for b in baselines:
        attributions.append(ig.attribute(spectrum, b))
    attributions = torch.stack(attributions)
    return torch.sum(attributions, dim=0)


torch._dynamo.config.suppress_errors = True


def make_attribution_fig(spec, attribution, id, save_name):

    fig, ax = plt.subplots(
        8, 1, figsize=(10, 20), sharex=True, gridspec_kw=dict(hspace=0.05)
    )

    ax[0].plot(
        DESI._wave_obs,
        spec.detach().cpu().numpy(),
    )
    [
        ax[j + 1].plot(
            DESI._wave_obs,
            attribution[j, 0].detach().cpu().numpy(),
            label=attribution_labels[j],
        )
        for j in range(7)
    ]
    ax[0].set_ylabel("Data")
    [ax[j + 1].axhline(0, c="gray", alpha=0.5, ls="--", zorder=-5) for j in range(7)]
    [ax[j + 1].set_ylabel(attribution_labels[j], fontsize=17) for j in range(7)]
    for a in ax:
        a.set_yticks([])
    plt.xlabel("Wavelength (A)")
    plt.suptitle(
        f"Outlier {id + 1}\nID={metadata['target_id'][id]}, z={metadata['z'][id]:.4f}\nRA={metadata['ra'][id]:.5f}, Dec={metadata['dec'][id]:.5f}\nlogP={-metadata['-logP'][id]:.3f}",
        fontsize=25,
    )
    plt.savefig(save_name, dpi=250)
    plt.close()


total_size = 200

for ix in tqdm(range(1, total_size + 1)):

    ig_attr = ig.attribute(spectra[ix - 1 : ix], n_steps=50)
    eg_attr = expected_attributes(spectra[ix - 1 : ix], get_baselines())
    ig_nt_attr = ig_nt.attribute(spectra[ix - 1 : ix])
    dl_attr = dl.attribute(spectra[ix - 1 : ix])
    fa_attr = fa.attribute(spectra[ix - 1 : ix])
    oc_attr = oc.attribute(spectra[ix - 1 : ix], sliding_window_shapes=(16,))
    gs_attr = gs.attribute(spectra[ix - 1 : ix], get_baselines()[:, 0], n_samples=256)

    attributions = torch.stack(
        [ig_attr, eg_attr, ig_nt_attr, dl_attr, fa_attr, oc_attr, gs_attr]
    )  # 5 x batch_size x n_spec

    make_attribution_fig(
        spectra[ix - 1],
        attributions,
        ix - 1,
        f"../figures/attribution_v2/outlier_{ix}.png",
    )
