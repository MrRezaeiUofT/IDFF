# Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling

<p align="center">
<em> Only 1 gpu is required.</em>
</p>

IDFF simultaneously learns an implicit flow and a scoring model
that come together during the sampling process. This structure allows
**IDFF to reduce the number of function evaluations (NFE) by more than 10 times** 
compared to traditional CFMs, enabling rapid sampling and efficient handling
of image and time-series data generation tasks.
See bellow for an illustration:


<p align="center">
<img src="Figure_1.png" alt="IDFF" width="80%"/>
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
<table>
  <tr>
    <th>Exp</th>
    <th>Args</th>
    <th>FID</th>
    <th>NFE</th>
    <th>Checkpoints</th>
  </tr>

  <tr>
    <td> CIFAR-10 </td>
    <td><a href="cifar10/simple_gen_test.py"> cifar10/simple_gen_test.py</a></td>
    <td>5.87</td>
    <td>10</td>
    <td><a href="">IDFF_cifar10_weights_step_final.pt</a></td>
  </tr>

  <tr>
    <td> CelebA-64 </td>
    <td><a href="celebA/simple_gen_test.py"> celebA/simple_gen_test.py</a></td>
    <td>11.83</td>
    <td>10</td>
    <td><a href="">IDFF_celeba_weights_step_final.pt</a></td>
  </tr>

  <tr>
    <td> CelebA-256 </td>
    <td><a href="celebA_HQ/simple_gen_test.py"> celebA_HQ/simple_gen_test.py</a></td>
    <td>---</td>
    <td>10</td>
    <td><a href="">IDFF_celeba_256_weights_step_final.pt</a></td>
  </tr>
    
  <tr>
    <td> LSUN-Bed </td>
    <td><a href="lsun_bed/simple_gen_test.py"> lsun_bed/simple_gen_test.py</a></td>
    <td>26.86</td>
    <td>10</td>
    <td><a href="">IDFF_lsun_bed_weights_step_final.pt</a></td>
  </tr>

  <tr>
    <td> LSUN-Church </td>
    <td><a href="lsun_church/simple_gen_test.py"> lsun_church/simple_gen_test.py</a></td>
    <td>12.86</td>
    <td>10</td>
    <td><a href="">IDFF_lsun_church_weights_step_final.pt</a></td>
  </tr>






</table>

## Dataset preparation 

For CelebA HQ 256, FFHQ 256 and LSUN, please check [NVAE's instructions](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) out.

The datasets for SST and MD experiments are provided [Here]().
## FID and 50K sample generation

##

## Citation
**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.
```bibtex
@article{xx,
  title={xx},
  author={xx},
  journal={xx},
  year={xx}
}
```

## Contacts

If you have any problems, please open an issue in this repository
or ping an email to [mr.rezaei](mailto:mr.rezaei@mail.utoronto.ca)