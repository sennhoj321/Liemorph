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
