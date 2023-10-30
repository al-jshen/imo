# Inverse Multiscale Occlusion (IMO): a multiscale feature attribution method for outliers

Paper: [ARXIV LINK HERE]

This feature attribution method can be used to explain a model that takes in some inputs and returns a scalar logP with the probability of the inputs relative to the training sample. This repository contains an implementation of IMO applied to a spectroscopic outlier detection model.

To get set up:
```bash
git clone git@github.com:al-jshen/imo.git
cd imo
pip install -r requirements.txt
```

Take a look at [notebooks/attribution.ipynb](notebooks/attribution.ipynb) for an example. All weights and sample data necessary are included in this repository.
