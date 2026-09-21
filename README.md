# ML-Regression-Perovskite-bandgap
This repository contains the data and codes used in our study of band gap properties in halide based perovskites using machine learning. The band gap dataset includes both reference data and DFT calculated data for A2BX6 compounds in cubic and rhombohedral structures. 

The optimized structures of cubic and rhombohedral A2BX6 are provided in the "A2BX6 structures" folder.

Analysis of the effects of A-site, B-site, and X-site elements on band gap energies is provided in the "Eg analysis" folder.  The original band gap data from the reference dataset, together with the DFT-calculated band gaps for cubic and rhombohedral A2BX6 structures, are also included in this folder.

The Pearson, Spearman, and Kendall correlation coefficients between the input features and the target property are provided in the "Features correlation" folder.

The machine learning codes used to train the band gap prediction models are provided in the "Regression models" folder, including Kernel Ridge Regression (KRR), Random Forest Regression (RFR), Gradient-Boosted Regression Trees (GBRT), Extreme Gradient Boosting (XGB).

## License
This code is made available under the MIT license.

## Requirements
The ML training and prediction codes are compatible with Python 3 and the following open source Python packages should be installed:

* numpy

* matplotlib

* pandas

* scikit-learn (1.5.2)

## Citation

Khamdang, Chadawan, and Mengen Wang. "Machine learning insights into band gap properties in halide-based perovskites." Physical Chemistry Chemical Physics (2026).

https://pubs.rsc.org/cp/article/doi/10.1039/d6cp02399a/1299466/Machine-learning-insights-into-band-gap-properties

https://arxiv.org/abs/2606.17186

## Contact
Mengen Wang, University of North Carolina at Chapel Hill (mewang@unc.edu)

Chadawan Khamdang, SUNY Binghamton 
