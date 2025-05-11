
# Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling

<p align="center">
<em> Only 1 GPU is required. </em>
</p>

**IDFF** is a novel generative modeling framework that simultaneously learns an implicit flow and a scoring model.  
These components operate jointly during sampling, enabling fast, accurate generation across both images and time-series data.

Thanks to its momentum-driven structure, **IDFF reduces the number of function evaluations (NFE) by more than 10×** compared to traditional conditional flow matching (CFM) models.

<p align="center">
<img src="2D_examples/sample_8gaussians/kde_with_baselines_and_gamma3_rotated.png" alt="IDFF 2D example" width="90%"/>
</p>

