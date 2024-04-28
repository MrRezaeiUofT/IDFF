# Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling

IDFF simultaneously learns an implicit flow and a scoring model
that come together during the sampling process. This structure allows
**IDFF to reduce the number of function evaluations (NFE) by more than 10 times** 
compared to traditional CFMs, enabling rapid sampling and efficient handling
of image and time-series data generation tasks.
See bellow for an illustration:


<p align="center">
<img src="Figure_1.png" alt="alternators" width="80%"/>
</p>

<p align="center">
<em> The final samples are generated with NFE=10</em>
</p>



For more information, please see our paper,
[Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling]().

## Usage

You can use IDFF for image generation examples by running:
```
python simple_gen_test.py 
```
associated with each dataset (inside each directory). 
To be able to run the code you need to download the pretrained model from below or train it from scratch.
Each pretrained model should be moved to 
the ```results/IDFF``` directory associated with each example.
## Pretrained
-**CIFAR-10** pretrained model is available here []()

-**CelebA-64** pretrained model is available here []()

-**CelebA-HQ** pretrained model is available here []()

-**LSUN-Bed** pretrained model is available here []()

-**LSUN-Church** pretrained model is available here []()

## Dataset preparation 

## FID and 50K sample generation

##

## Citation
```bibtex
@article{xx,
  title={xx},
  author={xx},
  journal={xx},
  year={xx}
}
```
