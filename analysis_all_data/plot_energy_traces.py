import biceps
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

mpl_colors = matplotlib.colors.get_named_colors_mapping()
mpl_colors = list(mpl_colors.values())[::5]
extra_colors = mpl_colors.copy()
#mpl_colors = ["k","lime","b","brown","c","green",
mpl_colors = ["b","brown","c","green",
              "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
              #'#acc2d9', "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
        '#e78ea5', '#983fb2', '#b7e1a1', '#430541', '#507b9c', '#c9d179',
            '#2cfa1f', '#fd8d49', '#b75203', '#b1fc99']+extra_colors[::2]+ ["k","grey"]

remove_xi0 = 0
energies = 500
burn = 1000

_outdir = "energy_traces"
biceps.toolbox.mkdir(_outdir)

#stat_model,nreplicas,data_uncertainty = "Bayesian", 1, "single"
#stat_model,nreplicas,data_uncertainty = "GB", 8, "single"
#stat_model,nreplicas,data_uncertainty = "Students",8, "single"
stat_model,nreplicas,data_uncertainty = "Gaussian", 8, "multiple"

outdir = f"*/nclusters_{energies}/{stat_model}_{data_uncertainty}_sigma/10000000_steps_{nreplicas}_replicas_*_trial_0"

# NOTE: the pickle file is just too large for GuassianModel (takes about 1 min to 3 min to load in object)
#path = f"{outdir}/sampler_obj.pkl"

# NOTE: the npz file takes a few seconds to load for GaussianModel
path = f"{outdir}/traj_lambda(0.0, 1.0).npz"

#files = biceps.toolbox.get_files(path)[::4]
files = biceps.toolbox.get_files(path)
#print(files)
#exit()
_files = []
_FF = ""
for j,file in enumerate(files):
    print("\n\n\n",file,"\n\n\n")
    FF = file.split("/")[0]
    FF = FF.replace('AMBER','A').replace('CHARM','C')

    # NOTE: Look at only a single within each force field
    if (j != 0):
        if (FF == _FF):
            _files[-1] = file
            continue
        else:
            _FF = FF
    _files.append(file)
files = _files.copy()



figsize = (10, 5)
#figsize=(8,4)
label_fontsize=12
legend_fontsize=10
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 2, width_ratios=(4, 1), wspace=0.02)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
axs = [ax1,ax2][::-1]
ax12 = fig.add_subplot(gs[0,1], sharey=ax1)
ax22 = fig.add_subplot(gs[1,1], sharey=ax2)
axs_ = [ax12,ax22][::-1]


previous_FF = ""
for j,file in enumerate(files):
    print("\n\n\n",file,"\n\n\n")
    FF = file.split("/")[0]
    FF = FF.replace('AMBER','A').replace('CHARM','C')

    # NOTE: Look at only a single within each force field
    if (j != 0) and (FF == previous_FF): continue
    else: previous_FF = FF

    label = FF

    if file.endswith("pkl"):
        outdir = file.split("sampler_obj.pkl")[0]
        sampler = biceps.toolbox.load_object(file)
        expanded_values = sampler.expanded_values
        A = biceps.Analysis(sampler, outdir=outdir, nstates=energies, MBAR=0, verbose=0)
        figs,steps,dists = A.plot_energy_trace()

    if file.endswith("npz"):
        traj_files = biceps.toolbox.get_files(f"{file.split('/traj_lambda')[0]}/traj_lambda(*).npz")
        steps,dists = [],[]
        expanded_values = []
        for traj_file in traj_files:
            vals = traj_file.split("traj_lambda(")[-1].split(").npz")[0].split(",")
            lam,xi = vals
            expanded_values.append((float(lam), float(xi)))

            npz = np.load(traj_file, allow_pickle=True)["arr_0"].item()
            traj = np.array(npz['trajectory']).T
            steps.append(traj[0].astype(int))
            dists.append(traj[1].astype(float))
        print(len(dists))
        print(expanded_values)


    x = steps[0].copy()
    for i in range(len(dists)):
        energy_dist = dists[i].copy()
        energy_dist /= nreplicas
        c = axs[i].plot(x, energy_dist, color=mpl_colors[j], label=label)
        # NOTE: burn-in a certain number of samples
        energy_dist = energy_dist[burn:]
        # NOTE: plot the distribution of the subsampled energy trace
        #axs_[i].hist(energy_dist, bins="auto", color=mpl_colors[j], alpha=0.5, orientation='horizontal') # edgecolor="k",
        axs_[i].hist(energy_dist, bins=25, color=mpl_colors[j], alpha=0.5, orientation='horizontal') # edgecolor="k",


ax2.set_xlim(left=np.min(x), right=np.max(x))
ax2.set_xlabel("steps", fontsize=14)

handles, labels = ax1.get_legend_handles_labels()
order = list(range(len(handles)))
#ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.25, 0.5), fontsize=label_fontsize)
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.30, 1.15), fontsize=label_fontsize)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
xticks = np.array(ax1.get_xticks())
xticks = np.linspace(0, x[-1], len(xticks), dtype=int)
ax1.set_xticklabels([f"{tick}" for tick in xticks])



for ax in axs_:
    #ax.set_yticks([])
    #ax.set_xticks([])
    for tick_label in ax.get_yticklabels():
        tick_label.set_visible(False)
    for tick_label in ax.get_xticklabels():
        tick_label.set_visible(False)
    ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
    ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
    ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
    ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(label_fontsize)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=label_fontsize-2)

for i,ax in enumerate(axs):
    lam,xi = expanded_values[i]
    vals_str = r"$\lambda$=%0.1f, $\xi$=%0.1f"%(lam, xi)
    ax.set_ylabel("Energy (kT)\n"+vals_str, fontsize=14)
    ax.grid()
    ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
    ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
    ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
    ax.tick_params(which="minor", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(label_fontsize)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=label_fontsize-2)
    if stat_model == "Students":
        ax.set_ylim(top=210)
    if stat_model == "Gaussian":
        ax.set_ylim(top=200)




if remove_xi0:
    ax1.set_ylim(top=200)
    ax2.set_ylim(top=200)
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.30, 1.25), fontsize=label_fontsize)
    fig.delaxes(ax2)
    fig.delaxes(ax22)
    ax2.set_xlabel("steps", fontsize=14)



#fig.tight_layout()
fig.subplots_adjust(left=0.087, bottom=0.125, top=0.95, right=0.765, wspace=0.20, hspace=0.5)
fig.savefig(f"{_outdir}/{stat_model}_iZ_distributions_{energies}_states.png", dpi=600)










