# Python libraries:{{{
import numpy as np
import sys, time, os, gc, string, re
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
#from scipy import stats
import biceps
#from biceps.decorators import multiprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import uncertainties as u ################ Error Prop. Library
import uncertainties.unumpy as unumpy #### Error Prop.
from uncertainties import umath
from biceps.toolbox import three2one
#:}}}

# Methods:{{{

def plot_data():
    figure = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0,0])
    data1 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files('cineromycin_B/J_NOE/*.noe')])
    ax1 = data1["model"].plot.hist(alpha=0.5, bins=100,
        edgecolor='black', linewidth=1.2, color="b", label="Model")
    data1["exp"].plot.hist(alpha=0.5, bins=100,
        edgecolor='black', linewidth=1.2, color="orange", label="Experiment", ax=ax1)
    ax1.legend(fontsize=14)
    ax1.set_xlabel(r"NOE distance ($\AA$)", size=16)
    ax1.set_ylabel("")

    ax2 = plt.subplot(gs[1,0])
    data2 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files('cineromycin_B/J_NOE/*.J')])
    ax2 = data2["model"].plot.hist(alpha=0.5, bins=100,
        edgecolor='black', linewidth=1.2, color="b", label="Model")
    data2["exp"].plot.hist(alpha=0.5, bins=100,
        edgecolor='black', linewidth=1.2, color="orange", label="Experiment", ax=ax2)
    ax2.set_xlabel(r"J coupling (Hz)", size=16)
    ax2.set_ylabel("")
    fig.tight_layout()




#:}}}

# Append to Database:{{{
def append_to_database(A, dbName="database_Nd.pkl", verbose=False, **kwargs):
    n_lambdas = A.K
    pops = A.P_dP[:,n_lambdas-1]
    pops = np.nan_to_num(pops, nan=0.0)
    BS = A.f_df
    prior_micro_populations = kwargs.get("prior microstate populations")
    prior_macro_populations = kwargs.get("prior macrostate populations")
    macro_populations = kwargs.get("macrostate populations")

    data = pd.DataFrame()
    data["FF"] = [kwargs.get("FF")]
    data["System"] = [kwargs.get("System")]
    data["nsteps"] = [kwargs.get("nsteps")]
    data["nstates"] = [kwargs.get("nStates")]
    data["nlambda"] = [kwargs.get("n_lambdas")]
    data["nreplica"] = [kwargs.get("nreplicas")]
    data["lambda_swap_every"] = [kwargs.get("lambda_swap_every")]
    data["Nd"] = [kwargs.get("Nd")]
    data["uncertainties"] = [kwargs.get("data_uncertainty")]
    data["stat_model"] = [kwargs.get("stat_model")]

    #model_scores = A.get_model_scores(verbose=False)# True)
    #for i,lam in enumerate(kwargs.get("lambda_values")):
    #    lam = "%0.2g"%lam
    #    data["BIC Score lam=%s"%lam] = [np.float(model_scores[i]["BIC score"])]

    for i,lam in enumerate(kwargs.get("lambda_values")):
        lam = "%0.2g"%lam
        data["BICePs Score lam=%s"%lam] = [BS[i,0]]
        data["BICePs Score Std lam=%s "%lam] = [2*BS[i,1]] # at 95% C


    try:                    data["k"] = [A.get_restraint_intensity()]
    except(Exception) as e: data["k"] = [np.nan]

    data["micro pops"] = [pops]
    data["macro pops"] = [macro_populations]
    data["prior micro pops"] = [prior_micro_populations]
    data["prior macro pops"] = [prior_macro_populations]
    data["D_KL"] = [np.nansum([pops[i]*np.log(pops[i]/prior_micro_populations[i]) for i in range(len(pops))])]
    data["RMSE (prior micro pops)"] = [np.sqrt(metrics.mean_squared_error(pops, prior_micro_populations))]
    try:
        data["RMSE (prior macro pops)"] = [np.sqrt(metrics.mean_squared_error(macro_populations, prior_macro_populations))]
    except(Exception) as e:
        pass
    data["prior"] = [kwargs.get("prior")]

    acceptance_info = kwargs.get("acceptance_info")
    columns = acceptance_info.columns.to_list()
    for i,lam in enumerate(acceptance_info["lambda"].to_list()):
        row = acceptance_info.iloc[[i]]
        for k,col in enumerate(row.columns[1:]):
            data[f"{col}_lam={lam}"] = row[col].to_numpy()

    data.to_pickle(dbName)
    gc.collect()

# }}}

# MAIN:{{{
save_obj = 1
testing = 1
if testing: FF = 0
else: FF = int(sys.argv[2])
if testing: nstates = 500
else: nstates = [5,10,50,75,100,500][int(sys.argv[1])]
# [ 0,   , 1     , 2               , 3           , 4         , 5        , 6     , 7      , 8 ]
#['A14SB', 'CM36', 'A99SBnmr1-ildn', 'A99SB-ildn', 'CM22star', 'OPLS-aa', 'A99', 'A99SB', 'CM27']

n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 8, 10000000, 5000, 0

n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 8, 10000000, 10000, 0
n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 8, 10000000, 0, 0

#n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 8, 10000000, 500, 0

#print(round(nsteps/10.9))
#print(round(nsteps/11))
#print(round(nsteps/13))
#exit()

#stat_model, data_uncertainty = "Bayesian_intermediate", "single"
#n_lambdas, nreplicas, nsteps, swap_every, change_Nr_every = 2, 1, 10000000, 0, 0

stat_model, data_uncertainty = "Students_intermediate", "single"
#stat_model, data_uncertainty = "GB_intermediate", "single"
#stat_model, data_uncertainty = "GaussianSP_intermediate", "single"
#stat_model, data_uncertainty = "Gaussian_intermediate", "multiple"

analysis_dir = "../analysis_all_data"
all_data = 0
#analysis_dir = "../analysis_noe_only"
noe_only = 0
#analysis_dir = "../analysis_j_only"
J_only   = 0
analysis_dir = "../analysis_cs_only"
cs_only  = 1


#if stat_model == "Gaussian_intermediate": burn = 1000000
#else: burn = 0

burn = 0

#ref_file_outdir = f"avg_FF_energies_and_scores"
#file = f"{ref_file_outdir}/{stat_model.replace('_intermediate','')}_nclusters_{nstates}_{analysis_dir.split('analysis_')[-1].split('_')[0]}.csv"
#print(file)
#df = pd.read_csv(file)
#print(df)
##exit()

ff_dict = {'A14SB':0, 'CM36':1, 'A99SBnmr1-ildn':2, 'A99SB-ildn':3, 'CM22star':4,
           'OPLS-aa':5, 'A99':6, 'A99SB':7, 'CM27':8}
#arr = df["FF"].to_numpy()


fwd_model_mixture = 1
swap_forward_model = 0

karplus_key = "Bax2007"
#karplus_key = "Habeck"

scale_energies = 0
find_optimal_nreplicas = 0
write_every = 1000
#write_every = 100
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
data_likelihood = "gaussian" #"log normal" # "gaussian"

continuous_space=0
dsigma=0.05 # good
move_sigma_std=1.0

attempt_move_state_every = 5
attempt_move_sigma_every = 2

#attempt_move_state_every = 50
#attempt_move_sigma_every = 20





scale_and_offset = 0
move_ftilde_every = 0
dftilde = 1.10 #1.0 #0.1
ftilde_sigma = 1.0 #2.0 #1.0

verbose = 0#False
if verbose:
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

walk_in_all_dim=continuous_space
multiprocess=1
plottype= "step"
####### Data and Output Directories #######
sys_name = "CLN001"
forcefield_dirs = next(os.walk(sys_name))[1]
skip_dirs = ['indices']
forcefield_dirs = [ff for ff in forcefield_dirs if ff not in skip_dirs]

print(forcefield_dirs)
#exit()
forcefield = f"{forcefield_dirs[FF]}"
print(forcefield)
#exit()
data_dir = f"{sys_name}/{forcefield}"
data_dirs = next(os.walk(data_dir))[1]
index = np.where(np.array(data_dirs) == f'nclusters_{nstates}')[0]
data_dir = f"{sys_name}/{forcefield}/{data_dirs[int(index)]}"
dir = f"{analysis_dir}/{forcefield}/{data_dirs[int(index)]}"

outdir = f'{dir}/{stat_model}_{data_uncertainty}_sigma'
biceps.toolbox.mkdir(outdir)

# NOTE: IMPORTANT: Create multiple trials to average over and check the deviation of results
check_dirs = next(os.walk(f"{dir}/{stat_model}_{data_uncertainty}_sigma/"))[1]
trial = len(check_dirs)

outdir = f'{dir}/{stat_model}_{data_uncertainty}_sigma/{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam__swap_every_{swap_every}_change_Nr_every_{change_Nr_every}_trial_{trial}'
#outdir = f'{dir}/{stat_model}_{data_uncertainty}_sigma/{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam__swap_every_{swap_every}_change_Nr_every_{change_Nr_every}_{FF}'
print(f"data_dir: {data_dir}")
print(f"outdir: {outdir}")
biceps.toolbox.mkdir(outdir)

file = f'{data_dir}/inverse_distances_k{nstates}_msm_energies.csv'
#print(file)
energies = pd.read_csv(file, index_col=0)
prior = energies.to_numpy()/4184.*(6.022e23)  # Joules to kcal/mol
prior = prior/0.5959   # convert to reduced free energies F = f/kT
energies = prior - prior.min()  # set ground state to zero, just in case
#print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")

if all_data: input_data = biceps.toolbox.sort_data(f'{data_dir}/CS_J_NOE_{karplus_key}')
if J_only: input_data = [[file] for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE_{karplus_key}/*.J')]
if noe_only: input_data = [[file] for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE_{karplus_key}/*.noe')]
if cs_only: input_data = [[file] for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE_{karplus_key}/*.cs*')]


#print(input_data)
#print(pd.read_pickle(input_data[135][-1]))
#exit()
dfs  = [pd.read_pickle(file) for file in input_data[0]]
Nd  = sum([len(df) for df in dfs])
#print(input_data[0])
#exit()

## to filter out J coupling data for testing
#input_data = np.transpose(input_data)[1]
#input_data = input_data.reshape(len(input_data),1)

print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
print(f"nSteps of sampling: {nsteps}\nnReplica: {nreplicas}")

sigMin, sigMax, dsig = 0.001, 200, 1.02
sigMin, sigMax, dsig = 0.001, 100, 1.02
arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
l = len(arr)
sigma_index = round(l*0.73)
#sigma_index = round(l*0.63)
#sigma=(sigMin, sigMax, dsig)
#sigma_index=sigma_index
#print(arr[sigma_index])


if stat_model == "Students":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"Students","single"
    beta,beta_index,stat_model,data_uncertainty=(1.0, 100.0, 10000),0,"Students","single"

elif stat_model == "GB":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 100.0, 1000),0,"GB","single"
    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"

elif stat_model == "GB_intermediate":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 100.0, 1000),0,"GB","single"
    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB","single"

elif stat_model == "GaussianSP_intermediate":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB",data_uncertainty
    beta,beta_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"GB",data_uncertainty

elif stat_model == "Students_intermediate":
    phi,phi_index,stat_model,data_uncertainty=(1.0, 2.0, 1),0,"Students","single"
    #beta,beta_index,stat_model,data_uncertainty=(1.0, 100.0, 10000),0,"Students","single"

    # NOTE: Need to limit the beta parameter to be within 0 and 10 for TI. Otherwise,
    # beta can get wayyy too large when xi gets close to zero.
    beta,beta_index,stat_model,data_uncertainty=(1.0, 10.0, 10000),0,"Students","single"

elif stat_model == "Bayesian_intermediate":
    phi,phi_index,data_uncertainty,stat_model=(1.0, 2.0, 1),0,data_uncertainty,"Bayesian"
    beta,beta_index,data_uncertainty,stat_model=(1.0, 2.0, 1),0,data_uncertainty,"Bayesian"

elif stat_model == "Gaussian_intermediate":
    phi,phi_index,data_uncertainty,stat_model=(1.0, 2.0, 1),0,data_uncertainty,"Gaussian"
    beta,beta_index,data_uncertainty,stat_model=(1.0, 2.0, 1),0,data_uncertainty,"Gaussian"

else:
    phi,phi_index,data_uncertainty=(1.0, 2.0, 1),0,data_uncertainty
    beta,beta_index,data_uncertainty=(1.0, 2.0, 1),0,data_uncertainty


options = biceps.get_restraint_options(input_data)
print(pd.DataFrame(options))
if all_data:
    options[0].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))
    options[1].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))
    options[2].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))
else:
    options[0].update(dict(ref="uniform", sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))


print(pd.DataFrame(options))
energies = np.concatenate(energies)
# [ 0,   , 1     , 2               , 3           , 4         , 5        , 6     , 7      , 8 ]
#['A14SB', 'CM36', 'A99SBnmr1-ildn', 'A99SB-ildn', 'CM22star', 'OPLS-aa', 'A99', 'A99SB', 'CM27']
#ff_dirs = forcefield_dirs
#print(ff_dirs)

from scipy import interpolate, integrate

ff_dirs = [forcefield_dirs[FF]]
lambda_values = [0.0]

ensemble = biceps.ExpandedEnsemble(energies, lambda_values=lambda_values)
#ensemble = biceps.ExpandedEnsemble(energies, expanded_values=expanded_values)
lambda_values = ensemble.lambda_values
ensemble.initialize_restraints(input_data, options)
sampler = biceps.PosteriorSampler(ensemble, nreplicas,
        write_every=write_every, change_Nr_every=change_Nr_every,
        continuous_space=continuous_space, dsigma=dsigma, move_sigma_std=move_sigma_std,
        dftilde=dftilde, move_ftilde_every=move_ftilde_every,
        ftilde_sigma=ftilde_sigma, scale_and_offset=scale_and_offset,
#        change_xi_every=round(nsteps/11.75), dXi=0.09, xi_integration=True) # GB, GaussianSP, Students (for all data and NOE)

        change_xi_every=round(nsteps/11), dXi=0.1, xi_integration=True) # Bayesian &&& GB, GaussianSP, Students (for J and cs)
        #xi_integration=1) # Bayesian (same as above)
sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1, swap_forward_model=swap_forward_model,
        find_optimal_nreplicas=find_optimal_nreplicas,
        attempt_move_state_every=attempt_move_state_every,
        attempt_move_sigma_every=attempt_move_sigma_every,
        burn=burn, print_freq=100, walk_in_all_dim=walk_in_all_dim,
        verbose=0, progress=1, multiprocess=multiprocess, capture_stdout=0)

ti_info = sampler.ti_info

mbar = sampler.get_mbar_obj_for_TI(multiprocess=1)

overlap = mbar.compute_overlap()
overlap_matrix = overlap["matrix"]
#matrices.append(overlap_matrix)
_results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
Deltaf_ij, dDeltaf_ij, Theta_ij = _results["Delta_f"], _results["dDelta_f"], _results["Theta"]
#print(Deltaf_ij)
f_df = np.zeros( (len(overlap_matrix), 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
f_df[:,0] = Deltaf_ij[0,:]  # NOTE: biceps score
f_df[:,1] = dDeltaf_ij[0,:] # NOTE: biceps score std
BS = -f_df[:,0]/sampler.nreplicas
#print(BS)
print("Integral from MBAR:", BS[-1])

force_constants = [r"$\xi$=%0.2g"%val for val in ti_info[0]]
fig, ax = plt.subplots(figsize=(14, 10))  # Adjust the figsize as desired
im = ax.pcolor(overlap_matrix, edgecolors='k', linewidths=2)

# Add annotations
for i in range(len(overlap_matrix)):
    for j in range(len(overlap_matrix[i])):
        value = overlap_matrix[i][j]
        if value > 0.01:
            text_color = 'white' if value < 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', color=text_color,
                    fontsize=12)  # Adjust fontsize as desired
ax.set_xticks(np.array(list(range(len(force_constants)))))
ax.set_yticks(np.array(list(range(len(force_constants)))))
ax.tick_params(axis='x', direction='inout')
ax.tick_params(axis='y', direction='inout')
ax.grid()

try:
    ax.set_xticklabels([str(tick) for tick in force_constants], rotation=90, size=16)
    ax.set_yticklabels([str(tick) for tick in force_constants], size=16)
except Exception as e:
    print(e)

ax.set_xticklabels(ax.get_xticklabels(), ha='left')
ax.set_yticklabels(ax.get_yticklabels(), va='bottom')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label("Overlap probability between states", size=16)  # Set the colorbar label
fig.tight_layout()
fig.savefig(f"{outdir}/contour.png")


integral = sampler.get_score_using_TI()
print("integral: ",integral)
####### store data #######
expanded_values = sampler.expanded_values
A = biceps.Analysis(sampler, outdir=outdir, scale_energies=0, MBAR=False, multiprocess=1)
A.plot_energy_trace()
sampler.save_trajectories(outdir)
if save_obj: biceps.toolbox.save_object(sampler, f"{outdir}/sampler_obj.pkl")


exit()

print(sampler.acceptance_info)
sampler.ensemble_names = names
sampler.save_trajectories(outdir)
#sampler.plot_exchange_info(xlim=(-100, nsteps), figsize=(12,6), figname=f"{outdir}/lambda_swaps.png")

if save_obj: biceps.toolbox.save_object(sampler, f"{outdir}/sampler_obj.pkl")

if swap_every != 0: sampler.plot_exchange_info(xlim=(-100, nsteps), figsize=(12,6), figname=f"{outdir}/lambda_swaps.png")
A = biceps.Analysis(sampler, outdir=outdir, nstates=len(energies), MBAR=0, scale_energies=scale_energies)
A.plot_acceptance_trace()
A.plot_energy_trace()
exit()

print(sampler.acceptance_info)
#print(sampler.exchange_info)

expanded_values = sampler.expanded_values

####### Posterior Analysis #######
A = biceps.Analysis(sampler, outdir=outdir, nstates=len(energies), MBAR=True,
                    scale_energies=scale_energies)

A.plot_acceptance_trace()
try:
    A.plot_energy_trace()
except(Exception) as e:
    print(e)





BS, pops = A.f_df, A.P_dP[:,n_lambdas-1]
pops = np.nan_to_num(pops, nan=0.0)
K = n_lambdas-1
dpops = A.P_dP[:,2*K]
dpops = np.nan_to_num(dpops, nan=0.0)

if scale_energies == False:
    BS /= sampler.nreplicas

print(f"BICePs Score = {BS[:,0]}")
if data_uncertainty == "single":
    #try:
    #    A.plot(plottype=plottype, figsize=(12,6), figname=f"BICePs.pdf", hspace=0.85, plot_all_distributions=1)
    #except(Exception) as e:
    #    print(e)
    A.plot(plottype=plottype, figsize=(12,6), figname=f"BICePs.pdf", hspace=0.85, plot_all_distributions=1)

else:
    A.plot(plottype=plottype, figsize=(12,14), figname=f"BICePs.pdf",hspace=0.85, plot_all_distributions=0)

#NOTE: Save analysis object?
#biceps.toolbox.save_object(A, f"{outdir}/Analysis_obj_{stat_model}_{data_uncertainty}_sigma.pkl")
mlp = pd.concat([A.get_max_likelihood_parameters(model=i) for i in range(len(lambda_values))])
mlp.reset_index(inplace=True, drop=True)
mlp.to_pickle(outdir+"/mlp.pkl")
#biceps.toolbox.save_object(A, filename=outdir+"/analysis.pkl")

#traj = sampler.traj[-1].__dict__
##init, frac = biceps.find_all_state_sampled_time(traj['state_trace'], len(energies))
#C = biceps.Convergence(traj, outdir=outdir)
##biceps.toolbox.save_object(C, filename=outdir+"/convergence.pkl")
#C.plot_traces(figname="traces.pdf", xlim=(0, nsteps))
#C.get_autocorrelation_curves(method="block-avg", maxtau=5000)
#C.process()

#A.plot_population_evolution()
#A.plot_convergence()


#exit()

dbName = f"{outdir}/results.pkl"
if os.path.exists(dbName): os.remove(dbName)


try:
    #ntop = int(len(energies)/10.)
    ntop = len(energies)
    topN_pops = pops[np.argsort(pops)[-ntop:]]
    print(topN_pops)
    topN_labels = [np.where(topN_pops[i]==pops)[0][0] for i in range(len(topN_pops))]
    print(topN_labels)
    #print(pops)
    #exit()
except(Exception) as e:
    print(e)


# NOTE: Get reweighted populations
df = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_assignments.csv", index_col=0, skiprows=0, comment='#')
df_pops = df.copy()
df_pops.columns = ["macrostate"]
df_pops["microstate"] = df_pops.index.to_numpy()
df_pops["population"] = pops
df_pops.to_csv(f"{outdir}/{stat_model}_{data_uncertainty}_sigma__reweighted_populations.csv")
grouped = df_pops.groupby(["macrostate"])
reweighted_macro_pops = grouped.sum().drop("microstate", axis=1)
reweighted_macro_pops["FF"] = [forcefield for i in range(len(reweighted_macro_pops.index.to_list()))]
reweighted_macro_pops["nclusters"] = [nstates for i in range(len(reweighted_macro_pops.index.to_list()))]
reweighted_macro_pops["System"] = [sys_name.split("/")[-1] for i in range(len(reweighted_macro_pops.index.to_list()))]
print(reweighted_macro_pops)
#print(df)

# NOTE: Get Prior MSM populations
df = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_assignments.csv", index_col=0, skiprows=0, comment='#')
msm_pops = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_pops.csv", index_col=0, skiprows=0, comment='#')
df_pops = df.copy()
df_pops.columns = ["macrostate"]
df_pops["microstate"] = df_pops.index.to_numpy()
df_pops["population"] = msm_pops
#df_pops.to_csv(f"{outdir}/{stat_model}_{data_uncertainty}_sigma__reweighted_populations.csv")
grouped = df_pops.groupby(["macrostate"])
macrostate_populations = grouped.sum().drop("microstate", axis=1)
macrostate_populations["FF"] = [forcefield for i in range(len(macrostate_populations.index.to_list()))]
macrostate_populations["nclusters"] = [nstates for i in range(len(macrostate_populations.index.to_list()))]
macrostate_populations["System"] = [sys_name.split("/")[-1] for i in range(len(macrostate_populations.index.to_list()))]
print(macrostate_populations)
#exit()

###############################################################################
###############################################################################
###############################################################################

#input_data = biceps.toolbox.sort_data(f'{data_dir}/CS_J_NOE')

# NOTE: Get Prior MSM populations
df = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_assignments.csv", index_col=0, skiprows=0, comment='#')
msm_pops = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_pops.csv", index_col=0, skiprows=0, comment='#')

if all_data or noe_only:
    noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/CS_J_NOE_{karplus_key}/*.noe")]
    #  Get the ensemble average observable
    noe_Exp = noe[0]["exp"].to_numpy()
    noe_model = [i["model"].to_numpy() for i in noe]

    # FIXME: add reweighted uncertainty that comes from predicted populations
    noe_prior = np.array([w*noe_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    noe_reweighted = np.nansum(np.array([u.ufloat(w,dpops[i])*noe_model[i] for i,w in enumerate(pops)]), axis=0) #.sum(axis=0)

    noe_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}-{three2one(row[1]['res2'])}.{row[1]['atom_name2']}" for row in noe[0].iterrows()]
    noe_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2']] for row in noe[0].iterrows()])


if all_data or J_only:
    J = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE_{karplus_key}/*.J')]
    #  Get the ensemble average observable
    J_Exp = J[0]["exp"].to_numpy()
    J_model = [i["model"].to_numpy() for i in J]

    J_prior = np.array([w*J_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    #J_reweighted = np.array([w*J_model[i] for i,w in enumerate(pops)]).sum(axis=0)
    J_reweighted = np.nansum(np.array([u.ufloat(w,dpops[i])*J_model[i] for i,w in enumerate(pops)]), axis=0) #.sum(axis=0)

    #J_labels = [f"{row[1]['atom_name1']}-{row[1]['atom_name2']}-{row[1]['atom_name3']}-{row[1]['atom_name4']}" for row in J[0].iterrows()]
    #J_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}-{three2one(row[1]['res2'])}.{row[1]['atom_name2']}-{three2one(row[1]['res3'])}.{row[1]['atom_name3']}-{three2one(row[1]['res4'])}.{row[1]['atom_name4']}" for row in J[0].iterrows()]
    J_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}\n{three2one(row[1]['res2'])}.{row[1]['atom_name2']}\n{three2one(row[1]['res3'])}.{row[1]['atom_name3']}\n{three2one(row[1]['res4'])}.{row[1]['atom_name4']}" for row in J[0].iterrows()]
    J_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2'], row[1]['atom_index3'], row[1]['atom_index4']] for row in J[0].iterrows()])


if all_data or cs_only:
    cs = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE_{karplus_key}/*.cs*')]
    #  Get the ensemble average observable
    cs_Exp = cs[0]["exp"].to_numpy()
    cs_model = [i["model"].to_numpy() for i in cs]

    cs_prior = np.array([w*cs_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    #cs_reweighted = np.array([w*cs_model[i] for i,w in enumerate(pops)]).sum(axis=0)
    cs_reweighted = np.nansum(np.array([u.ufloat(w,dpops[i])*cs_model[i] for i,w in enumerate(pops)]), axis=0) #.sum(axis=0)
    cs_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}" for row in cs[0].iterrows()]
    cs_label_indices = np.array([[row[1]['atom_index1']] for row in cs[0].iterrows()])


# NOTE: make plot of reweighted data points overlaid with experimental and prior
fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(3, 1)


ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])
#data1 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files('cineromycin_B/J_NOE/*.noe')])
data = []

if all_data or noe_only:
    for i in range(len(noe_reweighted)):
        try:
            data.append({"index":i,
                "reweighted noe":noe_reweighted[i].nominal_value, "reweighted noe err":noe_reweighted[i].std_dev,
                "prior noe":noe_prior[i], "exp noe":noe_Exp[i], "label":noe_labels[i]
                })
        except(Exception) as e:
            data.append({"index":i,
                "reweighted noe":noe_reweighted[i], "reweighted noe err":0.0,
                "prior noe":noe_prior[i], "exp noe":noe_Exp[i], "label":noe_labels[i]
                })


    data1 = pd.DataFrame(data)
    #print(data1)
    #exit()

    _data1 = data1.sort_values(["exp noe"])
    _data1 = _data1.reset_index()
    print(_data1)
    #exit()

    #ax1 = data1.plot.scatter(x='index', y="reweighted noe", s=5, edgecolor='black', color="b", label="BICePs")
    ax1.scatter(x=_data1.index.to_numpy(), y=_data1["prior noe"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
    ax1.scatter(x=_data1.index.to_numpy(), y=_data1["exp noe"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
    ax1.errorbar(x=_data1.index.to_numpy(), y=_data1["reweighted noe"].to_numpy(),
                 yerr=_data1["reweighted noe err"].to_numpy(),
                 #capsize=25,
                 ecolor="b", label="BICePs", fmt="o",
                 mec='k', mfc='b')
    ax1.set_xlim(-1, 140)
    #ax1.scatter(x=_data1.index.to_numpy(), y=_data1["reweighted noe"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
    ax1.legend(fontsize=14)
    ax1.set_xlabel(r"Index", size=16)
    ax1.set_ylabel(r"NOE distance ($\AA$)", size=16)


if all_data or J_only:
    data = []
    for i in range(len(J_reweighted)):
        try:
            data.append({"index":i,
                "reweighted J":J_reweighted[i].nominal_value, "reweighted J err":J_reweighted[i].std_dev,
                "prior J":J_prior[i], "exp J":J_Exp[i], "label":J_labels[i]
                })
        except(Exception) as e:
            data.append({"index":i,
                "reweighted J":J_reweighted[i], "reweighted J err":0.0,
                "prior J":J_prior[i], "exp J":J_Exp[i], "label":J_labels[i]
                })

    data1 = pd.DataFrame(data)

    ax3.scatter(x=data1['label'].to_numpy(), y=data1["prior J"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
    ax3.scatter(x=data1['label'].to_numpy(), y=data1["exp J"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
    #ax3.scatter(x=data1['label'].to_numpy(), y=data1["reweighted J"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
    ax3.errorbar(x=data1['label'].to_numpy(), y=data1["reweighted J"].to_numpy(),
                 yerr=data1["reweighted J err"].to_numpy(),
                 #capsize=25,
                 ecolor="b", label="BICePs", fmt="o",
                 mec='k', mfc='b')

    #ax3.legend(fontsize=14)
    #ax3.set_xlabel(r"Index", size=16)
    ax3.set_ylabel(r"J-coupling (Hz)", size=16)

if all_data or cs_only:
    data = []
    for i in range(len(cs_reweighted)):
        try:
            data.append({"index":i,
                "reweighted cs":cs_reweighted[i].nominal_value, "reweighted cs err":cs_reweighted[i].std_dev,
                "prior cs":cs_prior[i], "exp cs":cs_Exp[i], "label":cs_labels[i]
                })
        except(Exception) as e:
            data.append({"index":i,
                "reweighted cs":cs_reweighted[i], "reweighted cs err":0.0,
                "prior cs":cs_prior[i], "exp cs":cs_Exp[i], "label":cs_labels[i]
                })

    data1 = pd.DataFrame(data)

    ax2.scatter(x=data1['label'].to_numpy(), y=data1["prior cs"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
    ax2.scatter(x=data1['label'].to_numpy(), y=data1["exp cs"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
    #ax2.scatter(x=data1['label'].to_numpy(), y=data1["reweighted cs"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
    ax2.errorbar(x=data1["label"].to_numpy(), y=data1["reweighted cs"].to_numpy(),
                 yerr=data1["reweighted cs err"].to_numpy(),
                 #capsize=25,
                 ecolor="b", label="BICePs", fmt="o",
                 mec='k', mfc='b')


    #ax2.legend(fontsize=14)
    #ax2.set_xlabel(r"Index", size=16)
    ax2.set_ylabel(r"Chemical Shift (ppm)", size=16)

axs = [ax1,ax2,ax3]
#rotations = [0,0,65]
rotations = [0,0,0]
for n, ax in enumerate(axs):

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),]
    xmarks = [ax.get_xticklabels()]
    ymarks = [ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(14)
    for k in range(0,len(xmarks)):
        for mark in xmarks[k]:
            mark.set_size(fontsize=14)
            mark.set_rotation(s=rotations[n])
    for k in range(0,len(ymarks)):
        for mark in ymarks[k]:
            mark.set_size(fontsize=14)
            mark.set_rotation(s=0)

    ax.text(-0.1, 1.0, string.ascii_lowercase[n], transform=ax.transAxes,
            size=20, weight='bold')
fig.tight_layout()
fig.savefig(f"{outdir}/reweighted_observables.png")

print(mlp)

try:

    if all_data or noe_only:
        #print("# BICePs Posterior average x=%5.3f" % (biceps_avg_post[0]))
        #print("BICePs chix %5.3f" % (np.abs((biceps_avg_post[0]-avgx_exp))/exp_sigma[0]))
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        chi2_exp = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(noe_reweighted), noe_Exp], axis=1))
        chi2_prior = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(noe_reweighted), noe_prior], axis=1))
        print("BICePs Vs Exp.  chi^2  %5.3f" % (chi2_exp[0]))
        print("BICePs Vs Prior chi^2  %5.3f" % (chi2_prior[0]))
        #sse = np.array([(noe_reweighted[i] - noe_Exp[i])**2/np.std([noe_reweighted[i],noe_Exp[i]]) for i in range(len(noe_Exp))]).sum()
        #sse = np.array([(noe_reweighted[i] - noe_Exp[i])**2/(mlp.iloc[0]["sigma_noe"]**2 + mlp.iloc[-1]["sigma_noe"]**2) for i in range(len(noe_Exp))]).sum()
        #print(sse)
        #exit()
        columns = [col for col in mlp.columns.to_list() if "sigma_noe" in col]
        sse = []
        for k,col in enumerate(columns):
            sse.append(np.array([(noe_reweighted[i] - noe_Exp[i])**2/(mlp.iloc[0][col]**2 + mlp.iloc[-1][col]**2) for i in range(len(noe_Exp))]).sum())
        print(sse)


    if all_data or J_only:
        chi2_exp = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(J_reweighted), J_Exp], axis=1))
        chi2_prior = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(J_reweighted), J_prior], axis=1))
        print("BICePs Vs Exp.  chi^2  %5.3f" % (chi2_exp[0]))
        print("BICePs Vs Prior chi^2  %5.3f" % (chi2_prior[0]))
        #sse = np.array([(J_reweighted[i] - J_Exp[i])**2/np.std([J_reweighted[i],J_Exp[i]]) for i in range(len(J_Exp))]).sum()
        #sse = np.array([(J_reweighted[i] - J_Exp[i])**2/(mlp.iloc[0]["sigma_J"]**2 + mlp.iloc[-1]["sigma_J"]**2) for i in range(len(J_Exp))]).sum()
        #print(sse)
        columns = [col for col in mlp.columns.to_list() if "sigma_J" in col]
        sse = []
        for k,col in enumerate(columns):
            sse.append(np.array([(J_reweighted[i] - J_Exp[i])**2/(mlp.iloc[0][col]**2 + mlp.iloc[-1][col]**2) for i in range(len(J_Exp))]).sum())
        print(sse)


    if all_data or cs_only:
        chi2_exp = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(cs_reweighted), cs_Exp], axis=1))
        chi2_prior = scipy.stats.chi2_contingency(np.stack([unumpy.nominal_values(cs_reweighted), cs_prior], axis=1))
        print("BICePs Vs Exp.  chi^2  %5.3f" % (chi2_exp[0]))
        print("BICePs Vs Prior chi^2  %5.3f" % (chi2_prior[0]))
        #sse = np.array([(cs_reweighted[i] - cs_Exp[i])**2/np.std([cs_reweighted[i],cs_Exp[i]]) for i in range(len(cs_Exp))]).sum()
        #sse = np.array([(cs_reweighted[i] - cs_Exp[i])**2/(mlp.iloc[0]["sigma_H"]**2 + mlp.iloc[-1]["sigma_H"]**2) for i in range(len(cs_Exp))]).sum()
        #print(sse)
        columns = [col for col in mlp.columns.to_list() if "sigma_H" in col]
        sse = []
        for k,col in enumerate(columns):
            sse.append(np.array([(cs_reweighted[i] - cs_Exp[i])**2/(mlp.iloc[0][col]**2 + mlp.iloc[-1][col]**2) for i in range(len(cs_Exp))]).sum())
        print(sse)

except(Exception) as e:
    print(e)



kwargs = {"nsteps": nsteps, "nStates": nstates, "n_lambdas": n_lambdas,
    "nreplicas": nreplicas, "Nd": Nd, "data_uncertainty": data_uncertainty,
    "stat_model": stat_model,"lambda_values": lambda_values, "dir": dir,
    "prior microstate populations": msm_pops[msm_pops.columns[0]].to_numpy(),
    "prior macrostate populations": macrostate_populations[macrostate_populations.columns[0]].to_numpy(),
    "microstate populations": msm_pops[msm_pops.columns[0]].to_numpy(),
    "macrostate populations": reweighted_macro_pops[reweighted_macro_pops.columns[0]].to_numpy(),
    "lambda_swap_every": swap_every,
    "FF": forcefield, "System": sys_name.split("/")[-1],
    "prior": prior,
    "acceptance_info": sampler.acceptance_info,
    "mlp": mlp
    }

append_to_database(A, dbName=dbName, verbose=False, **kwargs)


try:
    approx_scores = A.approximate_scores(burn=1000)
except(Exception) as e:
    print(e)
    approx_scores = A.approximate_scores(burn=100)
print(approx_scores)
approx_scores.to_pickle(f"{outdir}/approx_scores.pkl")





exit()

#:}}}




