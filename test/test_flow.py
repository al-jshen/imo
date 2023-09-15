import torch
from outlier_attribution.flow import NeuralDensityEstimator
from outlier_attribution.utils import get_latent_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# s_dir = "/home/yanliang/spender-desi/spender-desi/runtime"  # directory that saves latent vectors
# model_tag = "star_k2"  # spender model + data tag
# data_tag = "DESIStars"  # DESI data
# latent_tag = "%s-%s" % (model_tag, data_tag)

# print("latent data:", latent_tag)

# data_loader = get_latent_data_loader(s_dir, which="train", latent_tag=latent_tag)
# valid_data_loader = get_latent_data_loader(s_dir, which="valid", latent_tag=latent_tag)

# for k, batch in enumerate(data_loader):
#     sample = batch[0]
#     break

# print("sample to infer dimensionality", sample.shape, sample.device)
print("device:", device)
print("torch.cuda.device_count():", torch.cuda.device_count())
n_latent = 6

NDE_theta = NeuralDensityEstimator(
    normalize=False,
    initial_pos={"bounds": [[0, 0]] * n_latent, "std": [0.05] * n_latent},
    method="maf",
)
sample = torch.randn(size=(1, 6)).to(device)
NDE_theta.build(sample)

print(NDE_theta)

NDE_theta.net.load_state_dict(torch.load("./weights/galaxy-flow-state_dict.pt"))

print("loaded pretrained model")

print(NDE_theta.net.log_prob(sample))
