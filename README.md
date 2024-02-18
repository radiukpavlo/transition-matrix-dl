# Explainable Deep Learning: A Visual Analytics Approach with Transition Matrices

[**Pavlo Radiuk**](https://scholar.google.com/citations?user=qmxDbPoAAAAJ&hl=en),
[**Olexander Barmak**](https://scholar.google.com/citations?user=pl4wbzoAAAAJ&hl=en),
[**Eduard Manziuk**](https://scholar.google.com/citations?user=bwW-dBEAAAAJ&hl=en),
[**Iurii Krak**](https://scholar.google.com/citations?user=oJB9PpYAAAAJ&hl=en)

This is the **official repository** for the paper "*Explainable Deep Learning: A Visual Analytics Approach with Transition Matrices*", which has been submitted to [**Mathematics**](https://www.mdpi.com/journal/mathematics) and is currently under review.

## Overview

>**Abstract**: <br>
> The opacity of artificial intelligence (AI) systems, especially in deep learning (DL), poses significant challenges to their comprehensibility and trustworthiness. This study aims to enhance the explainability of DL models through visual analytics (VA) and human-in-the-loop (HITL) principles, making these systems more transparent and understandable to users. In this work, we propose a novel approach that utilizes a transition matrix to interpret results from DL models through more comprehensible machine learning (ML) models. The methodology involves constructing a transition matrix between the feature spaces of DL and ML models as formal and mental models, respectively, improving the explainability of separating hyperplanes for classification tasks. The effectiveness of our methods is validated using the MNIST and Iris datasets, with quantitative analysis based on the Structural Similarity Index (SSIM) and Peak Sig-nal-to-Noise Ratio (PSNR) metrics. The application of the transition matrix to the MNIST and Iris datasets demonstrated significant improvements in model transparency and user comprehension. For the Iris dataset, the separating hyperplane achieved enhanced classification accuracy. Validation results showed notable improvements with average SSIM values of 0.697 and PSNR values reaching 17.94, indicating high-quality reconstruction and interpretation of DL model outcomes. Our study underscores the importance of explainable AI in bridging the gap between the complex decision-making processes of DL models and human understanding. By employing VA and a transition matrix, we have significantly improved the explainability of DL models without compromising their performance, marking a step forward in developing more transparent and trustworthy AI systems.

## Citation
If you make use of our work, please cite our paper:

```bibtex
@inproceedings{baldrati2023multimodal,
  title={Explainable Deep Learning: A Visual Analytics Approach with Transition Matrices},
  author={Pavlo Radiuk, Olexander Barmak, Eduard Manziuk and Iurii Krak},
  journal={Mathematics},
  year={2024}
}
```

## Getting Started

We recommend using the [**Anaconda**](https://www.anaconda.com/) package manager to avoid dependency/reproducibility
problems.
For Linux systems, you can find a conda installation
guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Installation

1. Clone the repository

```sh
git clone https://github.com/radiukpavlo/transition-matrix-dl
```

2. Install Python dependencies

```sh
conda env create -n my_project -f environment.yml
conda activate my_project
```

Alternatively, you can create a new conda environment and install the required packages manually:

```sh
conda create -n my_project -y python=3.9
conda activate my_project
pip install torch==1.12.1 torchmetrics==0.11.0 opencv-python==4.7.0.68 diffusers==0.12.0 transformers==4.25.1 accelerate==0.15.0 clean-fid==0.1.35 torchmetrics[image]==0.11.0
```

## Pre-trained models
The model and checkpoints are available via folders [**.\models**](https://github.com/radiukpavlo/transition-matrix-dl/tree/main/models) and [**.\checkpoints**](https://github.com/radiukpavlo/transition-matrix-dl/tree/main/checkpoints).

## Datasets
The original datasets used in this research can be freely downloaded. 

Start by downloading the original datasets from the following links:
- MNIST **[[link](https://github.com/radiukpavlo/transition-matrix-dl)]**
- Iris **[[link](https://archive.ics.uci.edu/dataset/53/iris)]**


## TODO
- [ ] include additional figures

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** you've made.
