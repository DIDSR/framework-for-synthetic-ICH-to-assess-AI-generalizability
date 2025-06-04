# Synthetic ICH for CAD Evaluations
Repository to reproduce data &amp; figures found in &lt;link to prepub/journal>.

See also [InSilicoICH](https://github.com/DIDSR/InSilicoICH)

## Installation for `notebooks/single_case_pipeline.ipynb`:
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

*Link to download pretrained weights*: <>
