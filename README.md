[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15602167.svg)](https://doi.org/10.5281/zenodo.15602167)

# Synthetic ICH for CAD Evaluations

This repository contains the code and methodologies to reproduce the results presented in the paper: [Knowledge-based synthetic intracranial hemorrhage CT datasets for device evaluations and generalizability assessments](https://zenodo.org/records/15602167/files/PREPRINT%20June5%20Synthetic%20ICH%20for%20CAD%20Evaluations.pdf?download=1).

Deep learning models for Computer-Assisted Detection (CAD) of intracranial hemorrhage (ICH) often struggle with generalizability when encountering CT data with characteristics underrepresented in their training sets (e.g., variations in patient demographics, hemorrhage types, or image acquisition parameters).

This project introduces an open-source framework to:

- Generate synthetic ICH CT data by inserting realistic, modeled hemorrhages (epidural, subdural, intraparenchymal) into a digital head phantom.
- Simulate mass effect and control hemorrhage volume and attenuation based on real data distributions.
- Create datasets with varied CT acquisition parameters (mAs, kVp) to robustly evaluate the generalizability of ICH detection models.

Our work validates this approach by demonstrating comparable performance of an ICH detection model on our synthetic dataset (AUC 0.877) versus an independent real dataset (AUC 0.919). This framework enables more comprehensive testing and evaluation of CAD devices for ICH.

See also [InSilicoICH](https://github.com/DIDSR/InSilicoICH)

## Installation:

See also [single_case_pipeline](notebooks/single_case_pipeline.ipynb)

```
# Best practice, use an environment rather than install in the base env
conda create -n "my-env" python=3.11.0 # tested on python=3.11.0
conda activate my-env
pip install insilicoCAD_requirements.txt
pip install git+https://github.com/DIDSR/InSilicoICH.git
# or: `pip install -e .` from local InSilicoICH clone 
```

## Reproducing Figures

Run notebooks or `bash figure_run_all.sh`

## CAD Model Weights

Model training was forked here: https://github.com/jmweaver-FDA/rsna_2019_gc

[Pretrained model weights](https://doi.org/10.5281/zenodo.15602166)
