# Liemorph
This repository is still under construction!

This repository connects the TransMorph architecture for 3D deformable image registration with a standalone module to solve a flow equation on Lie groups.
If you find this code useful, please cite:
```bibtex
@article{Lietorch2025,
  title={LieMorph: Transformer-based Image
Registration Using Flows on Lie Groups},
  author={Bostelmann,Johannes; Lellmann, Jan},
  journal={Prozeedings of BMVC2025 (in press)},
  year={2025}
}
```
This repository builds on https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

```bibtex
@article{chen2022transmorph,
title = {TransMorph: Transformer for unsupervised medical image registration},
journal = {Medical Image Analysis},
pages = {102615},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102615},
url = {https://www.sciencedirect.com/science/article/pii/S1361841522002432},
author = {Junyu Chen and Eric C. Frey and Yufan He and William P. Segars and Ye Li and Yong Du}
}
```
and uses code from https://github.com/voxelmorph/voxelmorph/, https://github.com/princeton-vl/lietorch. So please refer to it as well.
```bibtex
@article{balakrishnan2019voxelmorph,
  title={Voxelmorph: a learning framework for deformable medical image registration},
  author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R and Guttag, John and Dalca, Adrian V},
  journal={IEEE transactions on medical imaging},
  volume={38},
  number={8},
  pages={1788--1800},
  year={2019},
  publisher={IEEE}
}
```
```bibtex
@inproceedings{teed2021tangent,
  title={Tangent Space Backpropagation for 3D Transformation Groups},
  author={Teed, Zachary and Deng, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
}
```
## Installation
Make sure that the CUDA Toolkit is installed.  
If not, you can install it inside a Conda environment:

```bash
conda install ninja
```
```bash
conda install cuda-toolkit=12.6
```
```bash
pip install git+https://github.com/princeton-vl/lietorch.git
```
