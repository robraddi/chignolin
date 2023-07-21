
#import libraries:# {{{
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import biceps
from biceps.PosteriorSampler import u_kln_and_states_kn
from biceps.PosteriorSampler import u_kln_by_mapping_fwd_models
from biceps.convergence import get_autocorrelation_time,exp_average
from pymbar import MBAR
import sys, os, math, string
import seaborn as sb
from tqdm import tqdm
# }}}

# functions:{{{

def matprint(mat, fmt="g"):
    """
    https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def plot_contour(data, force_constants):

    fig, ax = plt.subplots()
    im = ax.contourf(data)
    ax.set_xticks(range(len(force_constants[0])))
    ax.set_xticklabels([r"$k_{f}$ = "+str("%0.2f"%tick) for tick in force_constants[0][::-1]])
    ax.set_yticks(range(len(force_constants[1])))
    ax.set_yticklabels([r"$k_{f}$ = "+str("%0.2f"%tick) for tick in force_constants[1][::-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()
    return fig


def get_avg_model_energies_and_scores(path_to_ref_traj, burn=1000, maxtau=500):
    files = biceps.toolbox.get_files(path_to_ref_traj)
    _files = []
    _FF = ""
    for j,file in enumerate(files):
        #print("\n\n\n",file,"\n\n\n")
        FF = file.split("/ncluster")[0].split("/")[-1]
        FF = FF.replace('AMBER','A').replace('CHARM','C')
        _files.append(file)
    files = _files.copy()
    previous_FF = ""
    results = []

    for j,file in tqdm(enumerate(files), total=len(files)):
        #print("\n\n\n",file,"\n\n\n")
        FF = file.split("/ncluster")[0].split("/")[-1]
        FF = FF.replace('AMBER','A').replace('CHARM','C')
        label = FF
        if file.endswith("npz"):
            traj_files = biceps.toolbox.get_files(f"{file.split('/traj_lambda')[0]}/traj_lambda(0.0, 1.0).npz")
            steps,dists = [],[]
            expanded_values = []
            for traj_file in traj_files:
                vals = traj_file.split("traj_lambda(")[-1].split(").npz")[0].split(",")
                lam,xi = vals
                expanded_values.append((float(lam), float(xi)))

                npz = np.load(traj_file, allow_pickle=True)["arr_0"].item()
                traj = np.array(npz['trajectory'], dtype=object).T
                steps.append(traj[0].astype(int))
                dists.append(traj[1].astype(float))
            #print(len(dists))
            #print(expanded_values)
        x = steps[0].copy()
        for i in range(len(dists)):
            energy_dist = dists[i].copy()
            energy_dist /= nreplicas
            minima = np.array(energy_dist.min())
            # NOTE: burn-in a certain number of samples
            _energy_dist = energy_dist[burn:] - minima
            try:
                tau_c = get_autocorrelation_time([_energy_dist], method="auto", maxtau=maxtau)
                tau_c = np.float(tau_c)
                if np.isnan(tau_c):
                    tau_c = 0
                elif math.isnan(tau_c):
                    tau_c = 0
                else:
                    tau_c = int(round(tau_c))
            except(Exception) as e:
                print(e)
                tau_c = 0
            if tau_c < 0.0:
                print("\n\n\n\n Negative autocorrelation_time! \n\n\n")
                exit()
            avg = minima -np.log(exp_average(_energy_dist[tau_c:]))
            results.append({"FF":FF, "avg":avg})
    results = pd.DataFrame(results)
    mean = results.groupby(["FF"]).agg("mean")
    sem = results.groupby(["FF"]).agg("sem")
    #minimum = mean.iloc[np.where(mean == mean.min())[0]]
    #ref = minimum.index.to_list()[0]
    return mean, sem
# }}}

# plot_u_kln:{{{
def plot_u_kln(u_kln, columns, outdir):
    diag = np.diagonal(u_kln, axis1=0, axis2=1)
    N, M, _ = u_kln.shape
    subfig_rows = 5
    subfig_cols = 5
    bins = 50
    counter = 0

    for i in range(N):
        for j in range(M):
            if counter % (subfig_rows * subfig_cols) == 0:
                if counter > 0:
                    fig.tight_layout()
                    fig.text(0.5, 0.02, 'Energy (kT)', ha='center', fontsize=16)
                    fig.subplots_adjust(bottom=0.05)
                    fig.savefig(f"{outdir}/distributions_{counter//(subfig_rows*subfig_cols)-1}.png")
                    plt.close(fig)
                fig, axes = plt.subplots(subfig_rows, subfig_cols, figsize=(20, 20))

            ax = axes[(counter // subfig_cols) % subfig_rows, counter % subfig_cols]

            # Subtract the diagonal elements from the corresponding column
            data = u_kln[i, j] - diag[:, j]

            ax.hist(data, bins=bins, edgecolor='black')
            label = f"{columns[i]} — {columns[j]}"
            ax.set_title(label, fontsize=10)
            counter += 1

    # Save last figure
    fig.tight_layout()
    fig.savefig(f"{outdir}/distributions_{counter//(subfig_rows*subfig_cols)}.png")
    plt.close(fig)

# }}}


nstates = 500
#nreplicas,stat_model,data_uncertainty = 1,"Bayesian","single"
nreplicas,stat_model,data_uncertainty = 8,"Students","single"
#nreplicas,stat_model,data_uncertainty = 8,"GB","single"
#nreplicas,stat_model,data_uncertainty = 8,"GaussianSP","single"
#nreplicas,stat_model,data_uncertainty = 8,"Gaussian","multiple"
use_intermediates = 1
scale_energies = 0
pull_random_trials = 0
use_only_FF_pairs = 1
multiprocess = 1

#analysis_dir = "../analysis_all_data"
#analysis_dir = "../analysis_noe_only"
#analysis_dir = "../analysis_j_only"
analysis_dir = "../analysis_cs_only"

sys_name = "./"
outdir = "final_scores"
biceps.toolbox.mkdir(outdir)
outdir = f"{outdir}/{analysis_dir.replace('../','')}"
biceps.toolbox.mkdir(outdir)
outdir = f"{outdir}/{stat_model}"
biceps.toolbox.mkdir(outdir)

ref_file_outdir = f"avg_FF_energies_and_scores"
biceps.toolbox.mkdir(ref_file_outdir)


if (stat_model == "GB") or (stat_model == "GaussianSP"): burn = 1000
else: burn = 2000

burn = 1000

ff_dict = {'A14SB':0, 'CM36':1, 'A99SBnmr1-ildn':2, 'A99SB-ildn':3, 'CM22star':4,
           'OPLS-aa':5, 'A99':6, 'A99SB':7, 'CM27':8}



get_reference = 0
if get_reference:
    outdir = f"{analysis_dir}/*/nclusters_{nstates}/{stat_model}_{data_uncertainty}_sigma/10000000_steps_{nreplicas}_replicas_*_trial_*"
    path_to_ref_traj = f"{outdir}/traj_lambda(0.0, 1.0).npz"
    #print(biceps.toolbox.get_files(path_to_ref_traj))
    #exit()
    mean, sem = get_avg_model_energies_and_scores(path_to_ref_traj, burn=burn, maxtau=500)
    minimum = mean.iloc[np.where(mean == mean.min())[0]]
    ref = minimum.index.to_list()[0]
    sorted_mean = mean.sort_values('avg', ascending=1)
    file = f"{ref_file_outdir}/{stat_model}_nclusters_{nstates}_{analysis_dir.split('analysis_')[-1].split('_')[0]}.csv"
    sorted_mean.to_csv(file)
    print(sorted_mean)

    exit()


# TODO: FIXME: keep for now, but need to remove after you have fixed make_plot

reference_models = [{"stat_model": "Students", "ref": "A99", "nstates": 500, "data":"all"},
                    {"stat_model": "Students", "ref": "A99SBnmr1-ildn", "nstates": 500, "data":"noe"},
                    {"stat_model": "Students", "ref": "A99", "nstates": 500, "data":"J"},
                    {"stat_model": "Students", "ref": "CM27", "nstates": 500, "data":"cs"},
                    {"stat_model": "Gaussian", "ref": "A99SB", "nstates": 500, "data":"all"},
                    {"stat_model": "GB", "ref": "A99", "nstates": 500, "data":"all"},
                    {"stat_model": "Bayesian", "ref": "A99SB", "nstates": 500, "data":"all"}]
reference_models = pd.DataFrame(reference_models)


# NOTE: Ref for Gaussian model:{{{
"""
FF              exp avg
A14SB           76.681930±2.206804
A99             63.472147±2.310510
A99SB           52.653463±2.272101
A99SB-ildn      66.183231±3.264134
A99SBnmr1-ildn  65.965468±1.610371
CM22star        93.830542±3.971661
CM27            76.827942±0.759844
CM36            89.186756±0.610089
OPLS-aa         73.944902±3.038064
"""
# }}}


make_plot = 0
print_only = 0
figname="BICePs_Scores.pdf"
annot_fontsize=13
# make_plot:{{{
if make_plot:
    analysis_dir = "../analysis*"
    cs,J,noe,students,gb,bayes,gaussian = [],[],[],[],[],[],[]
    for stat_model in ["Bayesian","Students","GB","Gaussian"]:
        outdir = f"final_scores/{analysis_dir.replace('../','')}"
        outdir = f"{outdir}/{stat_model}"
        files = biceps.toolbox.get_files(f"{outdir}/*_{nstates}_scores_trial_*.csv")
        #print(files)
        #exit()
        for file in files:
            print(file)
            _ = pd.read_csv(file)
            #try: _.drop(np.where(_["FF"].to_numpy()=="intermediate")[0], axis=0, inplace=True)
            intermediates = [i for i,inter in enumerate(_["FF"].to_numpy()) if "inter" in inter]
            try:
                _.drop(intermediates, axis=0, inplace=True)
            except(Exception) as e: pass
            print(_)
            _ref = _.iloc[[0]]["FF"].to_numpy()[0]
            _ref = _ref.split("/ncluster")[0].split("/")[-1]
            _ref = _ref.replace('AMBER','A').replace('CHARM','C')
            _.columns = _.columns[:-1].to_list()+["Rel BS"]
            _["ref"] = [_ref for i in range(len(_))]

            ## TODO: FIXME: need to use
            #file = f"{ref_file_outdir}/{stat_model}_nclusters_{nstates}_{analysis_dir.split('analysis_')[-1].split('_')[0]}.csv"
            #sorted_mean = pd.read_csv(file)



            ref = reference_models.iloc[np.where(reference_models["stat_model"].to_numpy() == stat_model)[0]]

            if "analysis_J_only" in file:
                ref = ref.iloc[np.where(ref["data"].to_numpy() == "J")[0]]["ref"].to_numpy()[0]
                if _ref == ref: J.append(_)
            elif "analysis_cs_only" in file:
                ref = ref.iloc[np.where(ref["data"].to_numpy() == "cs")[0]]["ref"].to_numpy()[0]
                if _ref == ref: cs.append(_)
            elif "analysis_noe_only" in file:
                ref = ref.iloc[np.where(ref["data"].to_numpy() == "noe")[0]]["ref"].to_numpy()[0]
                if _ref == ref: noe.append(_)
            else:
                if _ref == ref["ref"].to_numpy()[0]:
                    if "Students" in file: students.append(_)
                    if "Bayesian" in file: bayes.append(_)
                    if "GB" in file: gb.append(_)
                    if "Gaussian" in file: gaussian.append(_)


    students = pd.concat(students)
    bayes = pd.concat(bayes)
    gb = pd.concat(gb)
    gaussian = pd.concat(gaussian)
    noe = pd.concat(noe)
    cs = pd.concat(cs)
    J = pd.concat(J)
    #print(noe)
    #print(len(noe))
    #exit()



    # Create the figure and gridspec
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(3, 8, height_ratios=[1, 1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.2])

    # Define the subplots
    subplots = []
    for i in range(7):
        subplots.append(fig.add_subplot(gs[0, i]))

    for i in range(7):
        subplots.append(fig.add_subplot(gs[1, i]))

    #cbar_subplots = []
    #for i in range(7):
    #    cbar_subplots.append(fig.add_subplot(gs[2, 0:6]))


    _cols_ = [J, cs, noe, students, bayes, gb, gaussian]
    x_labels = ["J", "CS", "NOE", "all", "all", "all", "all"]
    titles = ["Student's","Student's","Student's","Student's", "Bayesian", "Good-Bad", "Gaussain"]

    pos = 0
    for j,_col_ in enumerate(_cols_):
        _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
        _col_ = _col_.iloc[np.where(_col_["lambda"].to_numpy() == 1.0)[0]]
        ref = list(set(_col_["ref"].to_numpy()))[0]
        _ref_index = np.where(_col_["FF"].to_numpy() == ref)[0][0]
        _col_.drop(["Trial","Model", "exp_avg", "approx_score", "lambda", "xi", "ref"], axis=1, inplace=True)

        if print_only:
            mean = _col_.groupby(["FF"]).agg("mean")
            sem = _col_.groupby(["FF"]).agg("sem")
            column = [col for col in mean.columns if "Rel" in col][0]
            # NOTE: just for printing the table
            arr = []
            for item in zip(mean[column].to_numpy(),sem[column].to_numpy()):
                arr.append("%0.3g"%item[0]+"±"+"%0.3g"%item[1])
            mean[column] = arr
            print(mean)
            #arr = []
            #for item in zip(mean["BS"].to_numpy(),sem["BS"].to_numpy()):
            #    arr.append("%0.3g"%item[0]+"±"+"%0.3g"%item[1])
            #print(mean)
            continue



        ax = subplots[j]
        data = _col_.groupby(["FF"]).agg("mean")
        column = "Rel BS"
        vmin, vmax = data[column].min(), data[column].max() #cbar_ax.get_data_limits()
        data = data.reset_index()
        data["nstates"] = [nstates for val in range(len(data))]
        data = data.pivot("FF", "nstates", column)
        if j == 0:
            # Create a new axes for the colorbar
            cbar_ax = fig.add_subplot(gs[2, :7])
            # left, bottom, width, height
            heatmap = sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                       cbar_kws=dict(use_gridspec=True), cbar_ax=cbar_ax)
            # Add the colorbar to the figure
            cbar = fig.colorbar(heatmap.get_children()[0], cax=cbar_ax, orientation='horizontal')
            ticks = cbar.get_ticks()
            # Update the colorbar limits
            cbar.set_ticks([vmin, vmax])
            tick_labels = ['Stronger','Weaker']
            cbar.set_ticklabels(tick_labels, fontsize=12)

        else:
            sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},cbar=False)

##############################

        # find the model with the lowest score and put a black box around it
        data_array  = data.to_numpy()
        # Find the indices of the minimum value in the data array
        highlighted_cells = [np.unravel_index(data_array.argmin(), data_array.shape)]
        # Iterate over the highlighted cells and draw rectangles around them
        for cell in highlighted_cells:
            rect = patches.Rectangle((cell[1]+0.90, cell[0]), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='gold', linestyle="-")
            ax.add_patch(rect)

        for cell in [(_ref_index,0)]:
            #rect = patches.Rectangle((cell[1], cell[0]), width=1.00, height=1.00, linewidth=2., edgecolor='white', facecolor='none', linestyle="--")
            rect = patches.Rectangle((cell[1]+0.90, cell[0]+0.5), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='white', linestyle="-")
            ax.add_patch(rect)



        if j == 0:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=1, top=1, left=0, right=0)

        #ax.set_xlabel(f"{x_labels[j]}")
        ax.set_xlabel(f"")
        ax.set_ylabel(f"")
        title = ax.set_title(f"{titles[j]}\n({x_labels[j]})", size=12, rotation=25, loc="center", pad=5)
        title.set_position([0.75, 1.0])
        if j == 0:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                labelright=False, bottom=True, top=1, left=True, right=0)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                labelright=False, bottom=True, top=1, left=0, right=0)
        pos += 1


    for j,_col_ in enumerate(_cols_):
        k = pos + j
        _col_ = _col_.iloc[np.where(_col_["lambda"].to_numpy() == 1.0)[0]]
        _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
        data = _col_.groupby(["FF"]).agg("mean")
        ax = subplots[k]
        column = "BS"
        vmin, vmax = data[column].min(), data[column].max() #cbar_ax.get_data_limits()
        data = data.reset_index()
        #print(data)
        data["nstates"] = [nstates for val in range(len(data))]
        data = data.pivot("FF", "nstates", column)
        sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},cbar=False)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                         labelright=False, bottom=True, top=True, left=True, right=True)
        ax.set_xlabel(f"")
        ax.set_ylabel(f"")
        if j == 0:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                labelright=False, bottom=True, top=1, left=True, right=0)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                labelright=False, bottom=True, top=1, left=0, right=0)


    # Customize the spacing between subplots
    gs.update(wspace=0.1, hspace=0.2)

    ## Add content to the subplots (example)
    for i, ax in enumerate(subplots):
        x,y = -0.1, 1.02
        ax.text(x,y, string.ascii_lowercase[i], transform=ax.transAxes,
                size=14, weight='bold')

        # Setting the ticks and tick marks
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks()]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(14)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=14)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)


#    # Add the bigg arrow
#    arrow_props = dict(arrowstyle='<-', linewidth=3)
#    ax_arrow = fig.add_subplot(gs[:, 7])
#    ax_arrow.annotate('', xy=(0.25, 0.1), xytext=(0.25, 1), arrowprops=arrow_props)
#    ax_arrow.set_axis_off()
#
#    # NOTE: add the xi 0 to 1
#    arrow_props = dict(arrowstyle='<-', linewidth=2)
#    ax_arrow = fig.add_subplot(gs[:, 7])
#    ax_arrow.annotate('', xy=(0.95, 0.25), xytext=(0.95, 0.35), arrowprops=arrow_props)
#    ax_arrow.text(0.50, 0.20, r'$\xi=0$', fontsize=12)
#    ax_arrow.text(0.50, 0.37, r'$\xi=1$', fontsize=12)
#    ax_arrow.set_axis_off()
#
#
#    # NOTE: add the lambda 0 to 1
#    arrow_props = dict(arrowstyle='<-', linewidth=2)
#    ax_arrow = fig.add_subplot(gs[:, 7])
#    ax_arrow.annotate('', xy=(0.95, 0.75), xytext=(0.95, 0.85), arrowprops=arrow_props)
#    #ax_arrow.annotate(r'$\lambda=0$', xy=(0.50, 0.70), xytext=(0.50, 0.70))
#    ax_arrow.text(0.50, 0.70, r'$\lambda=0$', fontsize=12)
#    ax_arrow.text(0.50, 0.87, r'$\lambda=1$', fontsize=12)
#    ax_arrow.set_axis_off()



    fig.subplots_adjust(left=0.225, bottom=0.0705, top=0.850, right=0.925)#, wspace=0.20, hspace=0.5)
    fig.savefig(f"{figname}", dpi=600)



    figname="BICePs_Scores_SEM.pdf"

    # Create the figure and gridspec
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(3, 8, height_ratios=[1, 1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.2])

    # Define the subplots
    subplots = []
    for i in range(7):
        subplots.append(fig.add_subplot(gs[0, i]))

    for i in range(7):
        subplots.append(fig.add_subplot(gs[1, i]))

    _cols_ = [J, cs, noe, students, bayes, gb, gaussian]
    x_labels = ["J", "CS", "NOE", "all", "all", "all", "all"]
    titles = ["Student's","Student's","Student's","Student's", "Bayesian", "Good-Bad", "Gaussain"]

    pos = 0
    for j,_col_ in enumerate(_cols_):
        _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
        _col_ = _col_.iloc[np.where(_col_["lambda"].to_numpy() == 1.0)[0]]
        ref = list(set(_col_["ref"].to_numpy()))[0]
        _ref_index = np.where(_col_["FF"].to_numpy() == ref)[0][0]
        _col_.drop(["Trial","Model", "exp_avg", "approx_score", "lambda", "xi", "ref"], axis=1, inplace=True)

        ax = subplots[j]
        data = _col_.groupby(["FF"]).agg("sem")
        column = "Rel BS"
        vmin, vmax = data[column].min(), data[column].max() #cbar_ax.get_data_limits()
        data = data.reset_index()
        data["nstates"] = [nstates for val in range(len(data))]
        data = data.pivot("FF", "nstates", column)
        if j == 0:
            # Create a new axes for the colorbar
            cbar_ax = fig.add_subplot(gs[2, :7])
            # left, bottom, width, height
            heatmap = sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                       cbar_kws=dict(use_gridspec=True), cbar_ax=cbar_ax)
            # Add the colorbar to the figure
            cbar = fig.colorbar(heatmap.get_children()[0], cax=cbar_ax, orientation='horizontal')
            ticks = cbar.get_ticks()
            # Update the colorbar limits
            cbar.set_ticks([vmin, vmax])
            tick_labels = ['Stronger','Weaker']
            cbar.set_ticklabels(tick_labels, fontsize=12)

        else:
            sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},cbar=False)

        # find the model with the lowest score and put a black box around it
        data_array  = data.to_numpy()
        # Find the indices of the minimum value in the data array
        highlighted_cells = [np.unravel_index(data_array.argmin(), data_array.shape)]
        # Iterate over the highlighted cells and draw rectangles around them
        for cell in highlighted_cells:
            rect = patches.Rectangle((cell[1]+0.90, cell[0]), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='gold', linestyle="-")
            ax.add_patch(rect)

        for cell in [(_ref_index,0)]:
            #rect = patches.Rectangle((cell[1], cell[0]), width=1.00, height=1.00, linewidth=2., edgecolor='white', facecolor='none', linestyle="--")
            rect = patches.Rectangle((cell[1]+0.90, cell[0]+0.5), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='white', linestyle="-")
            ax.add_patch(rect)



        if j == 0:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=1, top=1, left=0, right=0)

        #ax.set_xlabel(f"{x_labels[j]}")
        ax.set_xlabel(f"")
        ax.set_ylabel(f"")
        title = ax.set_title(f"{titles[j]}\n({x_labels[j]})", size=12, rotation=25, loc="center", pad=5)
        title.set_position([0.75, 1.0])
        if j == 0:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                labelright=False, bottom=True, top=1, left=True, right=0)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                labelright=False, bottom=True, top=1, left=0, right=0)
        pos += 1


    for j,_col_ in enumerate(_cols_):
        k = pos + j
        _col_ = _col_.iloc[np.where(_col_["lambda"].to_numpy() == 1.0)[0]]
        _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
        data = _col_.groupby(["FF"]).agg("sem")
        ax = subplots[k]
        column = "BS"
        vmin, vmax = data[column].min(), data[column].max() #cbar_ax.get_data_limits()
        data = data.reset_index()
        #print(data)
        data["nstates"] = [nstates for val in range(len(data))]
        data = data.pivot("FF", "nstates", column)
        sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},cbar=False)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                         labelright=False, bottom=True, top=True, left=True, right=True)
        ax.set_xlabel(f"")
        ax.set_ylabel(f"")
        if j == 0:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                labelright=False, bottom=True, top=1, left=True, right=0)
        else:
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                labelright=False, bottom=True, top=1, left=0, right=0)


    # Customize the spacing between subplots
    gs.update(wspace=0.1, hspace=0.2)

    ## Add content to the subplots (example)
    for i, ax in enumerate(subplots):
        x,y = -0.1, 1.02
        ax.text(x,y, string.ascii_lowercase[i], transform=ax.transAxes,
                size=14, weight='bold')

        # Setting the ticks and tick marks
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks()]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(14)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=14)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)


    fig.subplots_adjust(left=0.225, bottom=0.0705, top=0.850, right=0.925)#, wspace=0.20, hspace=0.5)
    fig.savefig(f"{figname}", dpi=600)







    exit()


# }}}


#ntrials = 30
ntrials = 5

# realtive biceps score calculations:{{{

file = f"{ref_file_outdir}/{stat_model}_nclusters_{nstates}_{analysis_dir.split('analysis_')[-1].split('_')[0]}.csv"
ref_table = pd.read_csv(file)
reference_model = ref_table["FF"].to_numpy()[0]
#print(ref_table)

#_reference_model = reference_models.iloc[np.where(reference_models["stat_model"].to_numpy() == stat_model)[0]]


for nstates in [500]:
    ensembles,trajs,logZs,results = [],[],[],[]
    expanded_values = []
    _FFs_ = ref_table["FF"].to_numpy()
    for ff in _FFs_:
        ff = ff.replace('A','AMBER').replace('C','CHARM')
        # get the sampler object that performed thermodynamic integration over xi
        path = f"{analysis_dir}/{sys_name}/{ff}/nclusters_{nstates}/{stat_model}_inter*_{data_uncertainty}_sigma/*/sampler_obj.pkl"
        files = biceps.toolbox.get_files(path)
        _results = []
        for file in files:
            print(file)
            sampler = biceps.toolbox.load_object(file)
            ti_info = sampler.ti_info

            mbar = sampler.get_mbar_obj_for_TI(multiprocess=1)

            # TODO: need to check if the xi=0 state has enough overlap with
            # adjacent states, otherwise the free energy calculation should
            #not be considered reliable
            overlap = mbar.compute_overlap()
            overlap_matrix = overlap["matrix"]

            mbar_results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
            Deltaf_ij, dDeltaf_ij, Theta_ij = mbar_results["Delta_f"], mbar_results["dDelta_f"], mbar_results["Theta"]
            f_df = np.zeros( (len(overlap_matrix), 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
            f_df[:,0] = Deltaf_ij[0,:]  # NOTE: biceps score
            f_df[:,1] = dDeltaf_ij[0,:] # NOTE: biceps score std
            BS = -f_df[:,0]/sampler.nreplicas
            print(BS)
            _results.append({"FF": ff, "nstates":nstates, "BS": BS[-1],
                             })

            # NOTE: if you want to compare how estimating the free energy using
            # the integral over the average derivative of the energy with respect to xi
            # compares to using MBAR
            integral = sampler.get_score_using_TI()
            print(integral)
            #exit()
        df = pd.DataFrame(_results)
        mean = df.groupby(["FF", "nstates"]).agg("mean")
        sem = df.groupby(["FF", "nstates"]).agg("sem")
        std = df.groupby(["FF", "nstates"]).agg("std")
        #print(df)
        #print(mean.to_numpy())
        #print(std.to_numpy())
        results.append({"FF": df["FF"].to_numpy()[0], "nstates": int(df["nstates"].to_numpy()[0]),
                        "BS": float(mean["BS"].to_numpy()[0]), "BS std": float(std["BS"].to_numpy()[0]),
                        "BS sem": float(sem["BS"].to_numpy()[0])})


        #    print(file)
        #    df = pd.read_csv(file)
        #exit()

    outfile = f"{outdir}/{stat_model}_{nstates}_scores_from_TI.csv"
    results = pd.DataFrame(results)
    results.to_csv(outfile)
    print(results)
    exit()






exit()






model_weights = []

for i in range(ntrials):
    trial = i
    print(f"Trial: {trial}")
    #for nstates in [5, 10, 50, 75, 100, 500]:
    for nstates in [500]:
        ensembles,trajs,logZs,results = [],[],[],[]
        expanded_values = []
        files = []
        if use_only_FF_pairs:
            num = 1
            _FFs_ = [ref_table["FF"].to_numpy()[0], ref_table["FF"].to_numpy()[num]]
            pair = [ff_dict[key] for key in _FFs_]
        else:
            _FFs_ = ref_table["FF"].to_numpy()
            #_FFs_ = ref_table["FF"].to_numpy()[:3]



        for ff in _FFs_:
            ff = ff.replace('A','AMBER').replace('C','CHARM')
            path = f"{analysis_dir}/{sys_name}/{ff}/nclusters_{nstates}/{stat_model}_{data_uncertainty}_sigma/*_trial_{trial}/sampler_obj.pkl"
            files.append(biceps.toolbox.get_files(path))

            # get the sampler object that performed thermodynamic integration over xi
            path = f"{analysis_dir}/{sys_name}/{ff}/nclusters_{nstates}/{stat_model}_inter*_{data_uncertainty}_sigma/*/sampler_obj.pkl"
            files.append(biceps.toolbox.get_files(path))


        files = np.concatenate(files)

        #print(files)
        if use_only_FF_pairs:
            to_keep = []
            current_ff = ""
            for f,file in enumerate(files):
                ff = file.split("/nclusters")[0].split("/")[-1]
                if ff != current_ff:
                    to_keep.append(f)
                    current_ff = ff
                    continue
                if file.split("/")[-2].endswith(f"{pair[0]}_{pair[1]}"):
                    to_keep.append(f)
            files = files[to_keep]
        #print(files)
        #exit()


        outfile = f"{outdir}/{stat_model}_{nstates}_scores_trial_{trial}.csv"
        # conditional to check if the calculation has already been perfomed
        if os.path.exists(outfile): continue

 #["A14SB","A99","A99SB-ildn","A99SB","A99SBnmr1-ildn","CM22star","CM27","CM36","OPLS-aa",]

#        files = [files[1], files[3], files[2], files[4]]#, files[4]]#, files[6]]#, files[8]]
        for file in files:
            print(file)

            FF = file.replace(f"{analysis_dir}/{sys_name}","").split("/")[0]
            FF = file.split("/ncluster")[0].split("/")[-1]
            FF = FF.replace('AMBER','A').replace('CHARM','C')

            ###################################################################
            if pull_random_trials:
                _ = biceps.toolbox.get_files(file.replace(f"trial_0", "trial_*"))
                num = np.random.randint(len(_))
                file = file.replace(f"trial_0", f"trial_{num}")
            else:
                try:
                    num = int(file.split("trial_")[-1].split("/")[0])
                except(Exception) as e:
                    num = np.nan
                    FF = "inter-"+FF
            ###################################################################


            sampler = biceps.toolbox.load_object(file)
            sampler.fwd_model_mixture = False
            if hasattr(sampler, "fwd_model_weights"):
                model_weights.append(sampler.fwd_model_weights)
            else:
                sampler.fwd_model_weights = [[1.0, 0.0],[0.0, 1.0]]
                model_weights.append(sampler.fwd_model_weights)

            ####### store data #######
            expanded_values.append(sampler.expanded_values)
            # get ensemble and normalization
            for k in range(len(sampler.ensembles)):
                ensembles.append(sampler.ensembles[k])
                logZs.append(sampler.logZs[k])
            # get trajectories
            trajs.append([sampler.traj[k].__dict__ for k in range(len(sampler.traj))])

            ####### Calculate BICePs scores #######
            A = biceps.Analysis(sampler, outdir=outdir, multiprocess=1, MBAR=0)
            # get BICePs scores w/ MBAR
#            BS, pops = A.f_df, A.P_dP[:,len(expanded_values[-1])-1]
#            BS /= sampler.nreplicas
            K = len(expanded_values[-1])-1
#            pops_std = A.P_dP[:,2*K]
            # approximate scores w/ exponential averaging
            try: approx_scores = A.approximate_scores(burn)
            except(Exception) as e: print(e);print(file);exit()
            approx_scores["exp_avg"] /= sampler.nreplicas
#            approx_scores["BS"] = BS[:,0] # add BICePs scores to DataFrame
            approx_scores["FF"] = [FF for f in range(len(approx_scores))]
            approx_scores["Trial"] = [num for f in range(len(approx_scores))]
            results.append(approx_scores)


        exit()
        nreplicas = sampler.nreplicas
        trajectories = list(np.concatenate(trajs))

        df = pd.concat(results)
        print(df)
        ref_model = 0

        # plot distributions:{{{
        label_fontsize=12
        legend_fontsize=8
        # NOTE: plot energy distributions
        figsize = (10, 5)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1, wspace=0.02)
        ax1 = fig.add_subplot(gs[0,0])
        axs = [ax1]

        n_trajs = 0
        for i in range(len(trajs)):
            for k in range(len(expanded_values[i])):
                model_id = results[i]["FF"].to_numpy()[k]
                x = np.array(trajs[i][k]['trajectory'], dtype=object).T[0]
                energy_dist = np.array(trajs[i][k]['trajectory'], dtype=object).T[1]
                energy_dist /= nreplicas
                energy_dist = energy_dist[burn:]
#                print(expanded_values[i][k])
                #label = r"Model$_{%s}$: $(\lambda, \xi)$ = "%(i)+str(expanded_values[i][k])
                label = r"%s: $(\lambda, \xi)$ = "%(model_id)+str(expanded_values[i][k])
                if "inter" in model_id:
                    label = '_nolegend_'
                    color = "gray"
                else:
                    color = biceps.toolbox.mpl_colors[::2][n_trajs]
                    n_trajs += 1
                #print(len(expanded_values[k]))
                if len(expanded_values[i]) > 2:
                    edge_colors = ["k" for i in range(100)]
                else:
                    edge_colors = ["r", "b", "k"]

                counts, bins = np.histogram(energy_dist, bins="auto")
                x,y = np.array(bins[:-1], dtype=float), np.array(counts, dtype=int)
                axs[0].fill_between(x, y, color=color, step="pre", alpha=0.5,
                                    label=label, edgecolor=edge_colors[k], linewidth=2.)

        axs[0].set_xlabel("Energy (kT)", fontsize=14)
        axs[0].set_ylabel("Counts", fontsize=14)

        handles, labels = axs[0].get_legend_handles_labels()
        order = list(range(len(handles)))
        axs[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                      loc='center left', bbox_to_anchor=(1.025, 0.5), fontsize=label_fontsize-1)
        xticks = np.array(axs[0].get_xticks())
        fig.subplots_adjust(left=0.09, bottom=0.125, top=0.95, right=0.60, wspace=0.20, hspace=0.5)
        fig.savefig(f"{outdir}/{stat_model}_energy_distributions_{nstates}_states_trial_{trial}.pdf", dpi=400)

        # }}}
        #exit()

        labels = [f"{ff} "+r"$\lambda$=%s"%lam for (ff,lam) in zip(df["FF"].to_numpy(), df["lambda"].to_numpy())]
        _outdir = outdir+f"/{stat_model}_{nstates}_scores_trial_{trial}"

        # TODO: FIXME: col_name and reference model needs to be defined more clearly
        col_name = "BS w/ ref=Model"
        #df = pd.concat(results)
        df[col_name+f"{ref_model}"] = f_df[:,0][:K]
        print(df)
        df.to_csv(f"{outfile}", index=False)





# }}}




exit()
















