# Model selection using Bayesian inference of conformational populations with replica-averaging


### Authors

* Robert M. Raddi
    - Department of Chemistry, Temple University
* Tim Marshall
    - Department of Chemistry, Temple University
* Vincent A. Voelz
    - Department of Chemistry, Temple University
  

Preprint DOI: [10.26434/chemrxiv-2023-396mm](https://doi.org/10.26434/chemrxiv-2023-396mm)

    

### Short description
Bayesian Inference of Conformational Populations (BICePs) reweighted conformational ensembles of the mini-protein chignolin simulated using nine different force fields in TIP3P water, using a set of 158 experimental measurements (139 NOE distances, 13 chemical shifts, and 6 vicinal J-coupling constants for HN and Hα.




### Repo contents
*NOTE: most Python scripts require `biceps` v3.0. Please see https://github.com/vvoelz/biceps for more details.*

- [`CLN001/`](CLN001/): chignolin data for each force field; conformational states (PDBs) from various MSMs, energies, populations
- [`data_processing/`](data_processing/): scripts used for processing the CLN001 data
 - [`data_processing/avg_over_snapshots.py`](data_processing/avg_over_snapshots.py): prepares data files for biceps 
- [`analysis_all_data/`](analysis_all_data/): scripts used for analysis (using all data)
 - [`analysis_all_data/final_scores`](analysis_all_data/final_scores): directory for compiled results of BICePs scores 
 - [`analysis_all_data/plot_energy_traces.py`](analysis_all_data/plot_energy_traces.py): plots traces of neglogP for (lambda, xi) trajectories
 - [`analysis_all_data/runme.py`](analysis_all_data/runme.py): script that uses biceps to compute the free energy from lambda=0 to lambda=1.
 - [`analysis_all_data/compile_results.py`](analysis_all_data/compile_results.py): compiles results generated from `runme.py` and makes plots; also performs chi-squared analysis
 - [`analysis_all_data/triangle_plot.py`](analysis_all_data/triangle_plot.py): triangle plot of BICePs reweighted and prior macrostate populations
 - [`analysis_all_data/plot_final_scores.py`](analysis_all_data/plot_final_scores.py): loads results from final scores and makes plots
 - [`analysis_all_data/thermodynamic_integration_of_xi_leg_using_mbar.py`](analysis_all_data/thermodynamic_integration_of_xi_leg_using_mbar.py): biceps script to get free energy of data restraints i.e., compute the free energy from xi=0 to xi=1
- [`analysis_J_only/`](analysis_J_only/): scripts used for analysis (only J-coupling data)
- [`analysis_cs_only/`](analysis_cs_only/): scripts used for analysis (only chemical shift data)
- [`analysis_noe_only/`](analysis_noe_only/): scripts used for analysis (only NOE distances)
-











































