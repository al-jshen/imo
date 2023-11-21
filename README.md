# Inverse Multiscale Occlusion (IMO): a multiscale feature attribution method for outliers

Paper: [https://arxiv.org/abs/2310.20012](https://arxiv.org/abs/2310.20012)

This feature attribution method can be used to explain a model that takes in some inputs and returns a scalar logP with the probability of the inputs relative to the training sample. This repository contains an implementation of IMO applied to a spectroscopic outlier detection model.

To get set up:
```bash
git clone git@github.com:al-jshen/imo.git
cd imo
pip install -r requirements.txt
```

Take a look at [notebooks/attributions.ipynb](notebooks/attributions.ipynb) for an example. All weights and sample data necessary are included in this repository.
