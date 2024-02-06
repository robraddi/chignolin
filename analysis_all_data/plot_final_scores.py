# Libraries:{{{
import string,re
import biceps
from biceps.toolbox import three2one
import itertools
import pandas as pd
import numpy as np
import pandas as pd
import scipy
from scipy.stats import sem
from sklearn import metrics
import matplotlib.pyplot as plt
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

import seaborn as sb
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import uncertainties as u ################ Error Prop. Library
#from tqdm import tqdm
# }}}



dpi=600
sys_name = "CLN001"
pub_images = 1
error_statistic = "sem"


analysis_dir = "../analysis_all_data"
suffix = analysis_dir.split("analysis_")[-1]
all_data = pd.read_pickle(f"{analysis_dir}/biceps_score_results/biceps_score_results_{suffix}.pkl")
all_data = all_data.iloc[np.where(all_data["nstates"] ==500)[0]]

J = pd.read_pickle(f"{analysis_dir}/biceps_score_results/biceps_score_results_j_only.pkl")
J = J.iloc[np.where(J["nstates"]==500)[0]]

cs = pd.read_pickle(f"{analysis_dir}/biceps_score_results/biceps_score_results_cs_only.pkl")
cs = cs.iloc[np.where(cs["nstates"]==500)[0]]

noe = pd.read_pickle(f"{analysis_dir}/biceps_score_results/biceps_score_results_noe_only.pkl")
noe = noe.iloc[np.where(noe["nstates"]==500)[0]]


students = all_data.iloc[np.where(all_data["stat_model"]=="Student's")[0]]
bayes = all_data.iloc[np.where(all_data["stat_model"]=="Bayesian")[0]]
#gaussian = all_data.iloc[np.where(all_data["stat_model"]=="Gaussian")[0]]
gaussianSP = all_data.iloc[np.where(all_data["stat_model"]=="GaussianSP")[0]]
gb = all_data.iloc[np.where(all_data["stat_model"]=="Good-Bad")[0]]


bayes_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_all_data/Bayesian/Bayesian_500_scores_from_TI.csv", index_col=0)
students_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_all_data/Students/Students_500_scores_from_TI.csv", index_col=0)
gaussianSP_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_all_data/GaussianSP/GaussianSP_500_scores_from_TI.csv", index_col=0)
#gaussian_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_all_data/Gaussian/Gaussian_500_scores_from_TI.csv", index_col=0)
gb_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_all_data/GB/GB_500_scores_from_TI.csv", index_col=0)
J_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_j_only/Students/Students_500_scores_from_TI.csv", index_col=0)
cs_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_cs_only/Students/Students_500_scores_from_TI.csv", index_col=0)
noe_xi_leg = pd.read_csv(f"{analysis_dir}/final_scores/analysis_noe_only/Students/Students_500_scores_from_TI.csv", index_col=0)


################################################################################
################################################################################
################################################################################




figname="extended_BICePs_Scores_PNAS.pdf"
title_fontsize=20
title_position=(0.5,0.98)
annot_fontsize=13
save_tables=0
cbar_loc="top" # "right"
grid=(2, 2)
positions=[(0,0), (0,1), (1,0), (1,1)]
figsize=(12, 12)


# Create the figure and gridspec
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 8, height_ratios=[1, 1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.2])

# Define the subplots
subplots = []
for i in range(7):
    subplots.append(fig.add_subplot(gs[0, i]))

for i in range(7):
    subplots.append(fig.add_subplot(gs[1, i]))


# Create the figure and gridspec
fig1 = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(3, 8, height_ratios=[1, 1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.2])

# Define the subplots
subplots1 = []
for i in range(7):
    subplots1.append(fig1.add_subplot(gs1[0, i]))

for i in range(7):
    subplots1.append(fig1.add_subplot(gs1[1, i]))



#cbar_subplots = []
#for i in range(7):
#    cbar_subplots.append(fig.add_subplot(gs[2, 0:6]))


#_cols_ = [J, cs, noe, students, bayes, gb, gaussian]
#_cols_xi_ = [J_xi_leg, cs_xi_leg, noe_xi_leg, students_xi_leg, bayes_xi_leg, gb_xi_leg, gaussianSP_xi_leg]
#x_labels = ["J", "CS", "NOE", "all", "all", "all", "all"]
#titles = ["Student's","Student's","Student's","Student's", "Bayesian", "Good-Bad", "Gaussain"]


_cols_ = [bayes, gaussianSP, gb, students, noe, J, cs]
_cols_xi_ = [bayes_xi_leg, gaussianSP_xi_leg, gb_xi_leg, students_xi_leg, noe_xi_leg, J_xi_leg, cs_xi_leg]
x_labels = ["all", "all", "all", "all", "NOE", "J", "CS", ]
titles = ["Single replica", "GaussainSP", "Good-Bad", "Student's","Student's","Student's","Student's"]


# IMPORTANT: for lambda=0 to lambda=1
# NOTE: add J-coupling
pos = 0
for j,_col_ in enumerate(_cols_):
    print(titles[j])
    _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
    results_df = _col_.groupby(["FF", "nstates", "uncertainties"]).agg("mean")
    results_df_error = _col_.groupby(["FF", "nstates", "uncertainties"]).agg(error_statistic)
    ax = subplots[j]
    ax1 = subplots1[j]
    biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in str(x)]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    grouped = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"])
    data = grouped.agg("mean")
    grouped = results_df_error[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"])
    data_error = grouped.agg("mean")
    data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()
    data_error["BICePs Score lam=1"] = data_error["BICePs Score lam=1"].to_numpy()
    vmin, vmax = data["BICePs Score lam=1"].min(), data["BICePs Score lam=1"].max() #cbar_ax.get_data_limits()
    vmin_error, vmax_error = data_error["BICePs Score lam=1"].min(), data_error["BICePs Score lam=1"].max() #cbar_ax.get_data_limits()
    data = data.reset_index()
    data_error = data_error.reset_index()
    data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
    data_error["nstates"] = [int(val) for val in data_error["nstates"].to_numpy()]
    data = data.pivot("FF", "nstates", "BICePs Score lam=1")
    data_error = data_error.pivot("FF", "nstates", "BICePs Score lam=1")
    print(data_error)
    if j == 0:
        # Create a new axes for the colorbar
        cbar_ax = fig.add_subplot(gs[2, :7])
        cbar_ax1 = fig1.add_subplot(gs1[2, :7])
        # left, bottom, width, height
        heatmap = sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True), cbar_ax=cbar_ax)
        heatmap1 = sb.heatmap(data_error, ax=ax1, linewidths=.5, annot=True, fmt="0.3f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True), cbar_ax=cbar_ax1)

        # Add the colorbar to the figure
        cbar = fig.colorbar(heatmap.get_children()[0], cax=cbar_ax, orientation='horizontal')
        ticks = cbar.get_ticks()
        # Update the colorbar limits
        #cbar.set_ticks([vmin, vmax])
        cbar.set_ticks([vmin+vmax*0.1, vmax-vmax*0.1])

        tick_labels = ['Stronger evidence','Weaker evidence']
        cbar.set_ticklabels(tick_labels, fontsize=12)

        # Add the colorbar to the figure
        cbar1 = fig1.colorbar(heatmap1.get_children()[0], cax=cbar_ax1, orientation='horizontal')
        ticks = cbar1.get_ticks()
        # Update the colorbar limits
        #cbar1.set_ticks([vmin_error, vmax_error])
        cbar1.set_ticks([vmin_error+vmax_error*0.1, vmax_error-vmax_error*0.1])
        tick_labels = ['Rel. lower error','Rel. higher error']
        cbar1.set_ticklabels(tick_labels, fontsize=12)

    else:
        sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},cbar=False)
        sb.heatmap(data_error, ax=ax1, linewidths=.5, annot=True, fmt="0.3f", annot_kws={"fontsize":annot_fontsize},cbar=False)

    # find the model with the lowest score and put a black box around it
    data_array  = data.to_numpy()
    # Find the indices of the minimum value in the data array
    highlighted_cells = [np.unravel_index(data_array.argmin(), data_array.shape)]
    # Iterate over the highlighted cells and draw rectangles around them
    for cell in highlighted_cells:
        rect = patches.Rectangle((cell[1]+0.90, cell[0]), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='gold', linestyle="-")
        ax.add_patch(rect)

    #for cell in [(_ref_index,0)]:
    #    #rect = patches.Rectangle((cell[1], cell[0]), width=1.00, height=1.00, linewidth=2., edgecolor='white', facecolor='none', linestyle="--")
    #    rect = patches.Rectangle((cell[1]+0.90, cell[0]+0.5), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='white', linestyle="-")
    #    ax.add_patch(rect)

    for axes in [ax,ax1]:
        axes.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                         labelright=False, bottom=True, top=True, left=True, right=True)
        #axes.set_xlabel(f"{x_labels[j]}")
        axes.set_xlabel(f"")
        axes.set_ylabel(f"")
        title = axes.set_title(f"{titles[j]}\n({x_labels[j]})", size=12, rotation=25, loc="center", pad=5)
        title.set_position([0.75, 1.0])
        if j == 0:
            axes.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                labelright=False, bottom=0, top=1, left=True, right=0)
        else:
            axes.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                labelright=False, bottom=0, top=1, left=0, right=0)

    pos += 1


for j,_col_ in enumerate(_cols_xi_):
    print(titles[j])
    k = pos + j
    _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
    data = _col_
    ax = subplots[k]
    ax1 = subplots1[k]
    data["nstates"] = np.ones(len(data))*500
    print(data)
    drop_cols = [col for col in data.columns if ("BS" in col)]
    drop_cols = [col for col in drop_cols if (error_statistic not in col)]
    print(drop_cols)

    _data = data.pivot("FF", "nstates")
    data = _data.drop(["BS std", "BS sem"], axis=1)
    data_error = _data.drop(drop_cols, axis=1)
    print(data_error)
    print(data)
    sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},cbar=False)
    sb.heatmap(data_error, ax=ax1, linewidths=.5, annot=True, fmt="0.3f", annot_kws={"fontsize":annot_fontsize},cbar=False)

    # find the model with the lowest score and put a black box around it
    data_array  = data.to_numpy()
    # Find the indices of the minimum value in the data array
    highlighted_cells = [np.unravel_index(data_array.argmin(), data_array.shape)]
    # Iterate over the highlighted cells and draw rectangles around them
    for cell in highlighted_cells:
        rect = patches.Rectangle((cell[1]+0.90, cell[0]), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='gold', linestyle="-")
        ax.add_patch(rect)

    #for cell in [(_ref_index,0)]:
    #    #rect = patches.Rectangle((cell[1], cell[0]), width=1.00, height=1.00, linewidth=2., edgecolor='white', facecolor='none', linestyle="--")
    #    rect = patches.Rectangle((cell[1]+0.90, cell[0]+0.5), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='white', linestyle="-")
    #    ax.add_patch(rect)

    for axes in [ax,ax1]:
        axes.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                         labelright=False, bottom=True, top=True, left=True, right=True)
        axes.set_xlabel(f"")
        axes.set_ylabel(f"")
        if j == 0:
            axes.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                labelright=False, bottom=0, top=1, left=True, right=0)
        else:
            axes.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                labelright=False, bottom=0, top=1, left=0, right=0)

# Customize the spacing between subplots
gs.update(wspace=0.1, hspace=0.2)
gs1.update(wspace=0.1, hspace=0.2)

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

## Add content to the subplots (example)
for i, ax in enumerate(subplots1):
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


for i,_gs in enumerate([gs,gs1]):
    figs = [fig,fig1]
    # Add the bigg arrow
    arrow_props = dict(arrowstyle='<-', linewidth=3)
    ax_arrow = figs[i].add_subplot(_gs[:, 7])
    ax_arrow.annotate('', xy=(0.25, 0.1), xytext=(0.25, 1), arrowprops=arrow_props)
    ax_arrow.set_axis_off()

    # NOTE: add the xi 0 to 1
    arrow_props = dict(arrowstyle='<-', linewidth=2)
    ax_arrow = figs[i].add_subplot(_gs[:, 7])
    ax_arrow.annotate('', xy=(0.95, 0.25), xytext=(0.95, 0.35), arrowprops=arrow_props)
    ax_arrow.text(0.50, 0.20, r'$\xi=0$', fontsize=12)
    ax_arrow.text(0.50, 0.37, r'$\xi=1$', fontsize=12)
    ax_arrow.set_axis_off()


    # NOTE: add the lambda 0 to 1
    arrow_props = dict(arrowstyle='<-', linewidth=2)
    ax_arrow = figs[i].add_subplot(_gs[:, 7])
    ax_arrow.annotate('', xy=(0.95, 0.75), xytext=(0.95, 0.85), arrowprops=arrow_props)
    #ax_arrow.annotate(r'$\lambda=0$', xy=(0.50, 0.70), xytext=(0.50, 0.70))
    ax_arrow.text(0.50, 0.70, r'$\lambda=0$', fontsize=12)
    ax_arrow.text(0.50, 0.87, r'$\lambda=1$', fontsize=12)
    ax_arrow.set_axis_off()


#if title_position != None:
#    fig.suptitle(f"BICePs Scores", fontweight="bold",
#                 x=title_position[0], y=title_position[1], size=title_fontsize)


#fig.tight_layout()
fig.subplots_adjust(left=0.225, bottom=0.0705, top=0.850, right=0.925)#, wspace=0.20, hspace=0.5)
fig.savefig(f"{figname}", dpi=600)
fig1.subplots_adjust(left=0.225, bottom=0.0705, top=0.850, right=0.925)#, wspace=0.20, hspace=0.5)
fig1.savefig(f"{figname.replace('.pdf','_error.pdf')}", dpi=600)




################################################################################
################################################################################
################################################################################



figname="total_BICePs_Scores.pdf"
title_fontsize=20
title_position=(0.5,0.98)
annot_fontsize=14
save_tables=0
cbar_loc="top" # "right"
grid=(2, 2)
positions=[(0,0), (0,1), (1,0), (1,1)]
figsize=(12, 12)


# Create the figure and gridspec
fig = plt.figure(figsize=(8, 4))
#gs = gridspec.GridSpec(2, 8, height_ratios=[1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.2])
gs = gridspec.GridSpec(2, 7, height_ratios=[1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1, 1])

# Define the subplots
subplots = []
for i in range(7):
    subplots.append(fig.add_subplot(gs[0, i]))




_cols_ = [bayes, gaussianSP, gb, students, noe, J, cs]
_cols_xi_ = [bayes_xi_leg, gaussianSP_xi_leg, gb_xi_leg, students_xi_leg, noe_xi_leg, J_xi_leg, cs_xi_leg]
x_labels = ["all", "all", "all", "all", "NOE", "J", "CS", ]
titles = ["Single replica", "GaussainSP", "Good-Bad", "Student's","Student's","Student's","Student's"]


pos = 0
for j,_col_ in enumerate(_cols_):
    print(titles[j])
    _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
    results_df = _col_.groupby(["FF", "nstates", "uncertainties"]).agg("mean")
    ax = subplots[j]
    biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in str(x)]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
    data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()
    vmin, vmax = data["BICePs Score lam=1"].min(), data["BICePs Score lam=1"].max() #cbar_ax.get_data_limits()
    data = data.reset_index()
    data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]

    data = data.pivot("FF", "nstates")


    _cols_xi_[j]["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _cols_xi_[j]["FF"].to_numpy()]
    _cols_xi_[j] = _cols_xi_[j].drop("BS std", axis=1)
    _cols_xi_[j] = _cols_xi_[j].pivot("FF", "nstates")
    arr = _cols_xi_[j]["BS"].to_numpy()
    data["BS"] = _cols_xi_[j]["BS"].to_numpy()
    index = np.where(data["BS"].to_numpy() == np.min(data["BS"].to_numpy()))[0]
    data["BS"] -= data["BS"].min()
    data["BS"] = data["BS"].to_numpy() + np.concatenate(data["BICePs Score lam=1"].to_numpy())

    vmin, vmax = data["BS"].min(), data["BS"].max() #cbar_ax.get_data_limits()
    data = data.reset_index()
    data["nstates"] = [500 for val in range(len(data))]
    data = data.drop("BICePs Score lam=1", axis=1)
    data = data.pivot("FF", "nstates", "BS")
#    data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
    print(data)


    if j == 0:
        # Create a new axes for the colorbar
        cbar_ax = fig.add_subplot(gs[1, :7])
        # left, bottom, width, height
        heatmap = sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True), cbar_ax=cbar_ax)
        # Add the colorbar to the figure
        cbar = fig.colorbar(heatmap.get_children()[0], cax=cbar_ax, orientation='horizontal')
        ticks = cbar.get_ticks()
        # Update the colorbar limits
        cbar.set_ticks([vmin+vmax*0.1, vmax-vmax*0.1])
        tick_labels = ['Stronger evidence','Weaker evidence']
        cbar.set_ticklabels(tick_labels, fontsize=14)

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

    #for cell in [(_ref_index,0)]:
    #    #rect = patches.Rectangle((cell[1], cell[0]), width=1.00, height=1.00, linewidth=2., edgecolor='white', facecolor='none', linestyle="--")
    #    rect = patches.Rectangle((cell[1]+0.90, cell[0]+0.5), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='white', linestyle="-")
    #    ax.add_patch(rect)


    ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                     labelright=False, bottom=True, top=True, left=True, right=True)
    #ax.set_xlabel(f"{x_labels[j]}")
    ax.set_xlabel(f"")
    ax.set_ylabel(f"")
    title = ax.set_title(f"{titles[j]}\n({x_labels[j]})", size=14, rotation=25, loc="center", pad=5)
    title.set_position([0.75, 1.0])
    if j == 0:
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
            labelright=False, bottom=0, top=1, left=True, right=0)
    else:
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
            labelright=False, bottom=0, top=1, left=0, right=0)
    pos += 1


# Customize the spacing between subplots
gs.update(wspace=0.1, hspace=0.15)

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
            if k == 1:
                mark.set_rotation(s=0)
                mark.set_position([0., 0.0])
                #mark.set_rotation(s=0)



## arrows:{{{
#
## Add the bigg arrow
#arrow_props = dict(arrowstyle='<-', linewidth=3)
#ax_arrow = fig.add_subplot(gs[:, 7])
#ax_arrow.annotate('', xy=(0.25, 0.1), xytext=(0.25, 1), arrowprops=arrow_props)
#ax_arrow.set_axis_off()
#
## NOTE: add the xi 0 to 1
#arrow_props = dict(arrowstyle='<-', linewidth=2)
#ax_arrow = fig.add_subplot(gs[:, 7])
#ax_arrow.annotate('', xy=(0.95, 0.4), xytext=(0.95, 0.7), arrowprops=arrow_props)
#ax_arrow.text(0.50, 0.25, r'$(\lambda=0,$'+'\n'+r' $\xi=0)$', fontsize=12)
#ax_arrow.set_axis_off()
#
#
## NOTE: add the lambda 0 to 1
##arrow_props = dict(arrowstyle='<-', linewidth=2)
##ax_arrow = fig.add_subplot(gs[:, 7])
##ax_arrow.annotate('', xy=(0.95, 0.75), xytext=(0.95, 0.85), arrowprops=arrow_props)
##ax_arrow.annotate(r'$\lambda=0$', xy=(0.50, 0.70), xytext=(0.50, 0.70))
#ax_arrow.text(0.50, 0.75, r'$(\lambda=1,$'+'\n'+r' $\xi=1)$', fontsize=12)
#ax_arrow.set_axis_off()
## }}}
#


#if title_position != None:
#    fig.suptitle(f"BICePs Scores", fontweight="bold",
#                 x=title_position[0], y=title_position[1], size=title_fontsize)


#fig.tight_layout()
fig.subplots_adjust(left=0.225, bottom=0.075, top=0.75, right=0.955)#, wspace=0.20, hspace=0.5)
fig.savefig(f"{figname}", dpi=600)





figname="total_BICePs_Scores_PNAS.pdf"
title_fontsize=20
title_position=(0.5,0.98)
annot_fontsize=14
save_tables=0
cbar_loc="top" # "right"
grid=(2, 2)
positions=[(0,0), (0,1), (1,0), (1,1)]
figsize=(12, 12)


# Create the figure and gridspec
fig = plt.figure(figsize=(8, 4))
#gs = gridspec.GridSpec(2, 8, height_ratios=[1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 0.2])
gs = gridspec.GridSpec(2, 6, height_ratios=[1, 0.1], width_ratios=[1, 1, 1, 1, 1, 1])

# Define the subplots
subplots = []
for i in range(6):
    subplots.append(fig.add_subplot(gs[0, i]))




_cols_ = [bayes, gb, students, noe, J, cs]
_cols_xi_ = [bayes_xi_leg, gb_xi_leg, students_xi_leg, noe_xi_leg, J_xi_leg, cs_xi_leg]
x_labels = ["all", "all", "all", "NOE", "J", "CS", ]
titles = ["Single replica", "Good-Bad", "Student's","Student's","Student's","Student's"]


pos = 0
for j,_col_ in enumerate(_cols_):
    print(titles[j])
    _col_["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _col_["FF"].to_numpy()]
    results_df = _col_.groupby(["FF", "nstates", "uncertainties"]).agg("mean")
    ax = subplots[j]
    biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in str(x)]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
    data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()
    vmin, vmax = data["BICePs Score lam=1"].min(), data["BICePs Score lam=1"].max() #cbar_ax.get_data_limits()
    data = data.reset_index()
    data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]

    data = data.pivot("FF", "nstates")


    _cols_xi_[j]["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in _cols_xi_[j]["FF"].to_numpy()]
    _cols_xi_[j] = _cols_xi_[j].drop("BS std", axis=1)
    _cols_xi_[j] = _cols_xi_[j].pivot("FF", "nstates")
    arr = _cols_xi_[j]["BS"].to_numpy()
    data["BS"] = _cols_xi_[j]["BS"].to_numpy()
    index = np.where(data["BS"].to_numpy() == np.min(data["BS"].to_numpy()))[0]
    #data["BS"] -= data["BS"].min()
    data["BS"] = data["BS"].to_numpy() + np.concatenate(data["BICePs Score lam=1"].to_numpy())
    data["BS"] -= data["BS"].min()

    vmin, vmax = data["BS"].min(), data["BS"].max() #cbar_ax.get_data_limits()
    data = data.reset_index()
    data["nstates"] = [500 for val in range(len(data))]
    data = data.drop("BICePs Score lam=1", axis=1)
    data = data.pivot("FF", "nstates", "BS")
#    data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
    print(data)


    if j == 0:
        # Create a new axes for the colorbar
        cbar_ax = fig.add_subplot(gs[1, :7])
        # left, bottom, width, height
        heatmap = sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True), cbar_ax=cbar_ax)
        # Add the colorbar to the figure
        cbar = fig.colorbar(heatmap.get_children()[0], cax=cbar_ax, orientation='horizontal')
        ticks = cbar.get_ticks()
        # Update the colorbar limits
        cbar.set_ticks([vmin+vmax*0.1, vmax-vmax*0.1])
        tick_labels = ['Stronger evidence','Weaker evidence']
        cbar.set_ticklabels(tick_labels, fontsize=14)

    else:
        sb.heatmap(data, ax=ax, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},cbar=False)

    ## find the model with the lowest score and put a black box around it
    #data_array  = data.to_numpy()
    ## Find the indices of the minimum value in the data array
    #highlighted_cells = [np.unravel_index(data_array.argmin(), data_array.shape)]
    ## Iterate over the highlighted cells and draw rectangles around them
    #for cell in highlighted_cells:
    #    rect = patches.Rectangle((cell[1]+0.90, cell[0]), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='gold', linestyle="-")
    #    ax.add_patch(rect)

    #for cell in [(_ref_index,0)]:
    #    #rect = patches.Rectangle((cell[1], cell[0]), width=1.00, height=1.00, linewidth=2., edgecolor='white', facecolor='none', linestyle="--")
    #    rect = patches.Rectangle((cell[1]+0.90, cell[0]+0.5), width=0.08, height=0.50, linewidth=0.5, edgecolor='k', facecolor='white', linestyle="-")
    #    ax.add_patch(rect)


    ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                     labelright=False, bottom=True, top=True, left=True, right=True)
    #ax.set_xlabel(f"{x_labels[j]}")
    ax.set_xlabel(f"")
    ax.set_ylabel(f"")
    title = ax.set_title(f"{titles[j]}\n({x_labels[j]})", size=14, rotation=25, loc="center", pad=5)
    title.set_position([0.75, 1.0])
    if j == 0:
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True,
            labelright=False, bottom=0, top=1, left=True, right=0)
    else:
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
            labelright=False, bottom=0, top=1, left=0, right=0)
    pos += 1


# Customize the spacing between subplots
gs.update(wspace=0.1, hspace=0.15)

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
            if k == 1:
                mark.set_rotation(s=0)
                mark.set_position([0., 0.0])
                #mark.set_rotation(s=0)



## arrows:{{{
#
## Add the bigg arrow
#arrow_props = dict(arrowstyle='<-', linewidth=3)
#ax_arrow = fig.add_subplot(gs[:, 7])
#ax_arrow.annotate('', xy=(0.25, 0.1), xytext=(0.25, 1), arrowprops=arrow_props)
#ax_arrow.set_axis_off()
#
## NOTE: add the xi 0 to 1
#arrow_props = dict(arrowstyle='<-', linewidth=2)
#ax_arrow = fig.add_subplot(gs[:, 7])
#ax_arrow.annotate('', xy=(0.95, 0.4), xytext=(0.95, 0.7), arrowprops=arrow_props)
#ax_arrow.text(0.50, 0.25, r'$(\lambda=0,$'+'\n'+r' $\xi=0)$', fontsize=12)
#ax_arrow.set_axis_off()
#
#
## NOTE: add the lambda 0 to 1
##arrow_props = dict(arrowstyle='<-', linewidth=2)
##ax_arrow = fig.add_subplot(gs[:, 7])
##ax_arrow.annotate('', xy=(0.95, 0.75), xytext=(0.95, 0.85), arrowprops=arrow_props)
##ax_arrow.annotate(r'$\lambda=0$', xy=(0.50, 0.70), xytext=(0.50, 0.70))
#ax_arrow.text(0.50, 0.75, r'$(\lambda=1,$'+'\n'+r' $\xi=1)$', fontsize=12)
#ax_arrow.set_axis_off()
## }}}
#


#if title_position != None:
#    fig.suptitle(f"BICePs Scores", fontweight="bold",
#                 x=title_position[0], y=title_position[1], size=title_fontsize)


#fig.tight_layout()
fig.subplots_adjust(left=0.225, bottom=0.075, top=0.75, right=0.955)#, wspace=0.20, hspace=0.5)
fig.savefig(f"{figname}", dpi=600)








exit()













