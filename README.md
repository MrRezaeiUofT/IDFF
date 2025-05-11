
# Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling

<p align="center">
<em> Only 1 GPU is required. </em>
</p>

**IDFF** is a novel generative modeling framework that simultaneously learns an implicit flow and a scoring model.  
These components operate jointly during sampling, enabling fast, accurate generation across both images and time-series data.

Thanks to its momentum-driven structure, **IDFF reduces the number of function evaluations (NFE) by more than 10×** compared to traditional conditional flow matching (CFM) models.

<p align="center">
<img src="assets/Figure_1.png" alt="IDFF Overview" width="90%"/>
</p>

