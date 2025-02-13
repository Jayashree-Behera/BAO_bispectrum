# Modelling the BAO Feature in the Bispectrum

This repository contains scripts for **modelling, MCMC chains, and visualization** related to the study presented in:

📄 **Behera et al. (2024), "Modelling the BAO Feature in the Bispectrum"**  
📚 Published in *MNRAS*  
🔗 [Read the Paper](https://doi.org/10.1093/mnras/stae1161)  
📜 [arXiv Preprint](https://arxiv.org/pdf/2409.16548)

### Abstract

We investigate how well a simple leading order perturbation theory model of the bispectrum can fit the baryon acoustic oscillation
(BAO) feature in the measured bispectrum monopole of galaxies. Previous works showed that perturbative models of the galaxy
bispectrum start failing at the wavenumbers of k ∼ 0.1 h Mpc−1. We show that when the BAO feature in the bispectrum is
separated, it can be successfully modelled up to much higher wavenumbers. We validate our modelling on GLAM simulations
that were run with and without the BAO feature in the initial conditions. We also quantify the amount of systematic error due to
BAO template being offset from the true cosmology. We find that the systematic errors do not exceed 0.3 per cent for reasonable
deviations of up to 3 per cent from the true value of the sound horizon.

---

## 📂 Repository Structure

- 📂 **spt_model/** - Analysis using [a simple leading order perturbation theory model](https://academic.oup.com/mnras/article/531/3/3326/7680003)
- 📂 **geofpt_model/** - Analysis using [GEO-FPT model](https://iopscience.iop.org/article/10.1088/1475-7516/2023/11/044)
- 📂 **my_utils/** - Scripts and utility functions for theoretical modelling and MCMC fitting  
- 📄 **bk_environement.yml** - Conda environemnt with Python dependencies


## Dependencies
Create the environement with the dependencies using:
```bash
conda env create -f bk_environment.yml -n env_name
```

## Contact and Citation
The project is still under development and documentation. If you need to use any scripts in research please contact me at jayashreeb@ksu.edu. If you would like to cite my paper, here is the full bibtex:
```bibtex
@ARTICLE{2024MNRAS.531.3326B,
       author = {{Behera}, Jayashree and {Rezaie}, Mehdi and {Samushia}, Lado and {Ereza}, Julia},
        title = "{Modelling the BAO feature in bispectrum}",
      journal = {\mnras},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = jul,
       volume = {531},
       number = {3},
        pages = {3326-3335},
          doi = {10.1093/mnras/stae1161},
archivePrefix = {arXiv},
       eprint = {2312.05942},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.3326B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
