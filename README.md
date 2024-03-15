# Introduction

This repository contains implementations of two differentiable 3D mesh renderers using PyTorch:
- `mesh_renderer`: A port of Google's [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) from Tensorflow to PyTorch. Based on the barycentric formulation from [Genova et al. 2018 "Unsupervised training for 3d morphable model regression."](https://openaccess.thecvf.com/content_cvpr_2018/papers/Genova_Unsupervised_Training_for_CVPR_2018_paper.pdf)
- `soft_mesh_renderer`: An alternate implementation of [SoftRas](https://github.com/ShichenLiu/SoftRas) that I built for my own learning. Based on the probabilistic rasterization formulation by [Liu et al. 2019 "Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning"](https://arxiv.org/abs/1904.01786).

# Setup

1. Create a virtual environment with `python3 -m venv env`
2. Activate it with `source env/bin/activate`
3. Install external dependencies with `pip install -r requirements.txt`

# Testing

Tests are included for both renderers.

- mesh_renderer: See [mesh_renderer docs](https://github.com/andrewkchan/pytorch_mesh_renderer/blob/master/src/mesh_renderer/README.md) for how to run these tests.
- soft_mesh_renderer: See [soft_mesh_renderer docs](https://github.com/andrewkchan/pytorch_mesh_renderer/blob/master/src/soft_mesh_renderer/README.md) for how to run these tests.