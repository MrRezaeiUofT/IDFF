# Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling

<p align="center">
<em>Only 1 GPU is required.</em>
</p>

**IDFF** is a novel generative modeling framework that simultaneously learns an implicit flow and a scoring model.  
These components operate jointly during sampling, enabling fast, accurate generation across both images and time-series data.

Thanks to its momentum-driven structure, **IDFF reduces the number of function evaluations (NFE) by more than 10×** compared to traditional conditional flow matching (CFM) models, while improving sampling quality and trajectory coherence.

<p align="center">
<img src="2D_examples/ImageGen.png" alt="IDFF image generation example" width="90%"/>
</p>

------------------------
<p align="center">
<img src="2D_examples/sample_8gaussians/kde_with_baselines_and_gamma3_rotated.png" alt="IDFF 2D example" width="90%"/>
</p>


## 🔬 Key Features

- **Implicit Score-and-Flow Fusion**: Combines score-based and flow-based generative models in a unified second-order SDE framework.
- **Momentum-aware Sampling**: Dynamically modulates sample generation with first- and second-order gradients, controlled by `γ₁` and `γ₂`.
- **Versatile Domains**: Supports both 2D synthetic distributions (e.g., checkerboard, spirals) and high-dimensional molecular dynamics (MD) data.
- **Fast Inference**: High-quality samples with as few as 2–3 function evaluations.
- **Easy Integration**: Compatible with `torchdiffeq` and `torchsde` solvers; plug in your own models.

## 🧪 2D Toy Example: Checkerboard

We demonstrate the effect of IDFF on a synthetic 2D distribution using KDE overlays and trajectory visualization.

- Models tested:  
  - Standard CFM  
  - IDFF with various $(\gamma_1, \gamma_2)$ configurations

- Metric: **Maximum Mean Discrepancy (MMD)**  
  - IDFF achieves lower MMD than CFM even with fewer NFEs

**Run:**
```bash
python run_checkerboard.py
````

<p align="center">
<img src="sample_checkerboard/kde_with_baselines_and_gamma3_rotated.png" width="85%">
</p>

## 🧬 Molecular Dynamics: PolyALA

We apply IDFF to molecular dynamics simulations, specifically backbone dihedral trajectories from PolyALA protein chains.

* Objective: Learn to generate future frames conditioned on current dynamics.
* Architecture: MLP-based embedding model trained with Exact Optimal Transport CFM.
* Output: Generated trajectories, phase space plots, and metrics like RMSE, MAE, CC.

**Train & Evaluate:**

```bash
python run_md_simulation.py
```

**Example Phase Space Plot:**

<p align="center">
<img src="MD/polyALA/phase_plot_overlayed_initial.png" width="80%">
</p>

## ⚙️ Usage Overview

### 1. **2D Generation Example**

```bash
python run_checkerboard.py
```

### 2. **Molecular Dynamics Simulation**

```bash
python run_md_simulation.py
```

You can customize the following parameters:

* `gamma1`, `gamma2`: momentum modulation
* `nfe`: number of function evaluations
* `dataset_name`: e.g., `polyALA`


## 🧠 Citation

If you find this work useful, please cite:

```bibtex
@misc{rezaei2025implicitdynamicalflowfusion,
      title={Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling}, 
      author={Mohammad R. Rezaei and Milos R. Popovic and Milad Lankarany and Rahul G. Krishnan},
      year={2025},
      eprint={2409.14599},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.14599}, 
}
```

## 📁 Directory Structure

```
.
├── MD/                        # Molecular dynamics output (trajectories, phase plots, metrics)
├── sample_checkerboard/      # 2D KDE plots and visualizations
├── 2D_examples/              # Pre-rendered sample images
├── run_checkerboard.py       # 2D IDFF demo
├── run_md_simulation.py      # MD simulation with IDFF
├── models/                   # Model definitions (MLP, embeddings)
└── README.md
```


## 📦 Dependencies

* `torch`, `torchsde`, `torchdiffeq`
* `matplotlib`, `seaborn`, `scikit-learn`
* `scipy`, `pandas`, `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Citation

If you find this work useful, please cite:

```
@misc{rezaei2025implicitdynamicalflowfusion,
      title={Implicit Dynamical Flow Fusion (IDFF) for Generative Modeling}, 
      author={Mohammad R. Rezaei and Milos R. Popovic and Milad Lankarany and Rahul G. Krishnan},
      year={2025},
      eprint={2409.14599},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.14599}, 
}
```

## 📬 Contact

For questions, feel free to raise an issue.

