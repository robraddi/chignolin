# Model selection using Bayesian inference of conformational populations with replica-averaging

### Authors

* Robert M. Raddi
    - Department of Chemistry, Temple University
* Tim Marshall
    - Department of Chemistry, Temple University
* Vincent A. Voelz
    - Department of Chemistry, Temple University
    

### Short description
Bayesian Inference of Conformational Populations (BICePs) reweighted conformational ensembles of the mini-protein chignolin simulated using nine different force fields in TIP3P water, using a set of 158 experimental measurements (139 NOE distances, 13 chemical shifts, and 6 vicinal J-coupling constants for HN and Hα.



### Repo contents

- [`scripts/`](scripts/): scripts used for analysis
  - [`scripts/score_correction.py`](scripts/score_correction.py): master script to run BICePs MCMC trajectories [(lambda=0.0, xi=0.0), (lambda=0.0, xi=1.0), (lambda=1.0, xi=1.0)] in parallel and save necessary files for analysis.
  - [`scripts/plot_energy_traces.py`](scripts/plot_energy_traces.py): plots traces of neglogP for (lambda, xi) trajectories
  - [`scripts/3-by-3-correlation_plot_for_observables.py`](scripts/3-by-3-correlation_plot_for_observables.py): plots experimental observables against forward model predicted observables and displays statistics like correlation and MAE.
  - [`scripts/fmu_compile_results_new.py`](scripts/fmu_compile_results_new.py): script to make various plots; BICePs score heatmaps, bar charts of populations, etc.
  - [`scripts/plot_sigma_trace.py`](scripts/plot_sigma_trace.py): plots the sigma traces for each step
  - [`scripts/prior_triangle_plot.py`](scripts/prior_triangle_plot.py): triangle plot of prior macrostate populations
  - [`scripts/triangle_plot.py`](scripts/triangle_plot.py): triangle plot of BICePs reweighted and prior macrostate populations


- [`tables/`](tables/): LaTeX tables 
- [`figures/`](figures/): 









































