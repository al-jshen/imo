from outlier_attribution.utils import load_desi_model

ae = load_desi_model("./weights/spender.desi-edr.galaxyae-b9bc8d12.pt")

print(ae)
