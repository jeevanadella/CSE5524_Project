# Imageomics HDR Scientific Mood Challenge Sample

This repository contains sample training code and submissions for the [2025 HDR Scientific Mood (Modeling out of distribution) Challenge: Beetles as Sentinel Taxa](CODABENCH LINK COMING). It is designed to give participants a reference for both working on the challenge, and also the expected publication of their submissions following the challenge (i.e., how to open-source your submission).

## Repository Structure

For your repository, you will want to complete the structure information below and add other files (e.g., training code):
```
submission
  <model weights>
  model.py
  requirements.txt
```
We also recommend that you include a [CITATION.cff](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) for your work.

**Note:** If you have requirements not included in the [whitelist](https://github.com/Imageomics/HDR-SMood-challenge/blob/main/ingestion_program/whitelist.txt), please check the [issues](https://github.com/Imageomics/HDR-SMood-challenge/issues) on the challenge GitHub to see if someone else has requested it before making your own issue.

### Structure of this Repository
```
HDR-SMood-challenge-sample
│
├── baselines
│   ├── training
│   │   └── <MODEL NAME>
│   │       ├── evaluation.py
│   │       ├── model.py
│   │       ├── train.py
│   │       └── utils.py
│   └── submissions
│       └── <MODEL NAME>
│           ├── model.pth
│           ├── model.py
│           └── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

To-Do: add notebook like last year (in a [`notebooks/` folder](https://github.com/Imageomics/HDR-anomaly-challenge-sample/tree/main/notebook)).
<!--
This repository also includes butterfly_sample_notebook.ipynb which loads the metadata for the images and displays a histogram of the hybrid/non-hybrid distribution by subspecies. It then downloads 15% of the data and runs through a simplified sample submission training with that subset (the sample image amount can be adjusted to work within network constraints). To run this notebook, first clone this repository and create a fresh conda environment, then install the requirements file:

conda create -n butterfly-sample -c conda-forge pip -y
conda activate butterfly-sample
pip install -r requirements.txt
jupyter lab
-->

## Installation & Running

### Installation
If you have `uv` simply run `uv sync`, otherwise you can use the `requirements.txt` file with either `conda` or `pip`.

### Training
An example training run can be executed by running the following:
```
python baselines/training/train.py
```

with `uv` do:
```
uv run python baselines/training/train.py
```

### Evaluation
Aftering training, you can locally evaluate your model by running the following:
```
python baselines/training/evaluation.py
```

with `uv` do:
```
uv run python baselines/training/evaluation.py
```

## References
Baselines built off of BioCLIPv2 and Dinov2:

Gu, Jianyang, et al. "Bioclip 2: Emergent properties from scaling hierarchical contrastive learning." arXiv preprint arXiv:2505.23883 (2025).

Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." arXiv preprint arXiv:2304.07193 (2023).

[Sample repo from Y1 challenge](https://github.com/Imageomics/HDR-anomaly-challenge-sample).

