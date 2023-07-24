# Libraries:{{{
import string,re,copy
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import uncertainties as u ################ Error Prop. Library
#from tqdm import tqdm
# }}}

## Pairplot_of_biceps_scores:{{{
#def pairplot_of_biceps_scores(df, stat_models, figname="cross_correlations.png"):
#    """
#    1. plot biceps scores against scores of different statistical models
#    2. plots chi^2 against Kyle Beauchamp's "Are forcefields getting better"
#    3.
#    """
#
#    results = df.copy()
#
#    fig = plt.figure(figsize=(12, 12))
#    gs = gridspec.GridSpec(2, 2)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
##    print(results['stat_model'].to_numpy())
##    exit()
#
#    positions = [(0,0), (0,1), (1,0), (1,1)]
#    combos = itertools.combinations(stat_models, 2)
#    for i,pair in enumerate(combos):
#        ax4 = plt.subplot(gs[positions[i]])
#
#
#        results_df = results.where((results["stat_model"] == pair[0]) | (results["stat_model"] == pair[1]))
#        results_df = results.groupby(["stat_model", "FF", "nstates", "uncertainties"]).agg("mean")
#
#        biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in x]
#        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
#        columns = ["nsteps","nreplica","nlambda"]
#        colors = ["grey", "orange", "g", "r"]
#
#        #print(results_df)
#        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1)#.groupby(["FF","nstates"]).agg("mean")
#        data = data.groupby(["stat_model","FF","nstates","BICePs Score lam=1"]).agg("mean")
#        print(data)
##        data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()#/nreplica
#
#        #NOTE: over a certain number of states...
#
#        data = data.reset_index().pivot("FF", "nstates", "BICePs Score lam=1")
#        print(data)
#        #for column in data.columns.to_list():
#        data = data[data.columns[0]]
#        print(data)
#        exit()
#
#
#        x, y = data.where(data.index.to_numpy()==data[pair[0]]), data.where(data.index.to_numpy()==data[pair[1]])
#        #data = np.array([FF, nstates])
##        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f") #, label=col, vmin=-20, vmax=0)
#        ax4.scatter(x, y, ax=ax4)
#
#        for row in ax4.get_xticklabels():
#            row.set_text(row.get_text().split(",")[0].split("(")[-1])
#
#        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
#        ax4.set_ylabel("")
#        ax4.set_xlabel("States", fontsize=16)
#
#        if positions[i] == (0,0):
#            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
#                             labelright=False, bottom=True, top=True, left=True, right=True)
#            ax4.set_xlabel("")
#
#        if positions[i] == (0,1):
#            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
#                             labelright=False, bottom=True, top=True, left=True, right=True)
#            ax4.set_xlabel("")
#
#        if positions[i] == (1,1):
#            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
#                             labelright=False, bottom=True, top=True, left=True, right=True)
#
#        #ax4.set_ylabel("BICePs Score", fontsize=16)
#        ax4.set_title(f"BICePs Scores using {stat_model} model", size=16)
#        #ax4.xaxis.set_visible(False)
#
#
#        # Setting the ticks and tick marks
#        ticks = [ax4.xaxis.get_minor_ticks(),
#                 ax4.xaxis.get_major_ticks()]
#        marks = [ax4.get_xticklabels(),
#                ax4.get_yticklabels()]
#        for k in range(0,len(ticks)):
#            for tick in ticks[k]:
#                tick.label.set_fontsize(16)
#        for k in range(0,len(marks)):
#            for mark in marks[k]:
#                mark.set_size(fontsize=16)
#                #mark.set_rotation(s=65)
#
#    fig.tight_layout()
#    fig.savefig(f"{figname}", dpi=600)
#
## }}}
#
# Plot:{{{


def compute_chi2(f_exp, fX, sigma, use_reduced=False):
    """Compute the chi-squared values using uncertainties that come from prior.
    This is meant to follow Kyle Beauchamp in the paper 'Are Force Fields getting better?'
    simga_noe = std(snapshots in microstate clusters)
    simga_cs = shiftx2 tabulated uncerainty values
    sigma_J = Karplus relation constants have uncertainties associated with them

    http://www.shiftx2.ca/performance.html

    """
    chi2 = np.array([(f_exp[i] - fX[i])**2/sigma[i]**2 for i in range(len(f_exp))]).sum()
    if use_reduced: chi2 /= len(f_exp)
    return chi2

# plot_heatmaps_of_chi2_prior:{{{
def plot_heatmaps_of_chi2_prior(df, reduced=False, figname="chi_squared.pdf",
                      save_tables=True, cols=None, Nds=None, all_data=0,
                      title_fontsize=20, title_position=(0.5,0.98), annot_fontsize=13,
                      cbar_loc="right", grid=(2, 2), plot_titles=None,
                      positions=[(0,0), (0,1), (1,0), (1,1)],
                      figsize=(12, 12)):

    df["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in df["FF"].to_numpy()]
    results = df.copy()
    #print(results)
    #print(results[cols[0]])

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid[0], grid[1])#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    left_cols, bottom_cols, other_cols = biceps.toolbox.get_seperate_columns(positions)

    for i, col in enumerate(cols):
        if reduced: prefix = r"$\chi^{2}/N$"
        else:       prefix = r"$\chi^{2}$"

        if Nds != None: Nd = Nds[i]
        ax4 = plt.subplot(gs[positions[i]])
        biceps_score_cols = [col]
        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
        results_df = results.where(results["stat_model"]=="Single replica")
        results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        print(results_df)
        results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")
        #results_df = results_df.groupby(["FF", "nstates"]).agg("mean")

        if not all_data: prefix = prefix + " for %s"%col.split("_")[-1]

        if Nds != None:
            if Nd: ax4.set_title(r"Prior %s ($N$=%s)"%(prefix,Nd), size=16)
            else:  ax4.set_title(r"Prior %s"%prefix, size=16)
        if plot_titles:
            ax4.set_title(r"%s"%plot_titles[i], size=16)

        columns = ["nsteps","nreplica","nlambda"]
        colors = ["grey", "orange", "g", "r"]
        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        #data = results[col].groupby(["FF", "nstates", "uncertainties"]).agg("mean").reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        #print(data)
        data = data.reset_index()
        data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
        print(data)
        data = data.pivot("FF", "nstates", col)

        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True,location=cbar_loc)) #, label=col, vmin=-20, vmax=0)
        cbar = ax4.collections[0].colorbar
        # here set the labelsize by 14
        cbar.ax.tick_params(labelsize=14)


        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)
        if np.where(positions[i] == np.array(left_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(other_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(bottom_cols))[0] != []:
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)

        if positions[i] == (grid[0]-1, 0):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)



        #ax4.set_title(f"BICePs Scores ({stat_model})", size=16)
        #ax4.set_title(f"{stat_model}", size=16)
        #ax4.xaxis.set_visible(False)


        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)
        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')
    #if title_position != None:
    #    fig.suptitle(f"BICePs Scores", fontweight="bold",
    #                 x=title_position[0], y=title_position[1], size=title_fontsize)
    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)

# }}}


# plot_heatmap_of_chi2_prior:{{{
def plot_heatmap_of_chi2_prior(df, reduced=False, Nd=None, figname="chi_squared.pdf", save_tables=True, datatype="all"):

    if datatype == "all": datatype = ""

    if reduced: _chi2_col_ = "reduced chi-squared_prior"
    else:       _chi2_col_ = "chi-squared_prior"


    results = df.copy()
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    ax4 = plt.subplot(gs[(0,0)])
    biceps_score_cols = [_chi2_col_]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    results_df = results.where(results["stat_model"]=="Single replica")
    results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
    results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")

    columns = ["nsteps","nreplica","nlambda"]
    colors = ["grey", "orange", "g", "r"]
    data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
    data = data.reset_index().pivot("FF", "nstates", _chi2_col_)
    #if save_tables: data.to_latex(f"{figname.split('.')[0]}_prior.tex")
    if save_tables: data.to_html(f"{figname.split('.')[0]}_prior.html")
    sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f") #, label=col, vmin=-20, vmax=0)

    for row in ax4.get_xticklabels():
        row.set_text(row.get_text().split(",")[0].split("(")[-1])

    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.set_ylabel("")
    ax4.set_xlabel("States", fontsize=16)
    ax4.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                     labelright=False, bottom=True, top=False, left=True, right=False)
    #ax4.set_xlabel("")
    if reduced: prefix = r"$\chi^{2}/N$"
    else:       prefix = r"$\chi^{2}$"

    if datatype != "":
        prefix = prefix + " for %s"%datatype

    if Nd: ax4.set_title(r"%s from prior ($N$=%s)"%(prefix,Nd), size=16)
    else:  ax4.set_title(r"%s from prior"%prefix, size=16)

    # Setting the ticks and tick marks
    ticks = [ax4.xaxis.get_minor_ticks(),
             ax4.xaxis.get_major_ticks()]
    marks = [ax4.get_xticklabels(),
            ax4.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            if k == 0:
                #mark.set_rotation(s=25)
                mark.set_rotation(s=0)

            #mark.set_rotation(s=65)

    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)

# }}}


# plot_heatmap_of_chi2:{{{
def plot_heatmap_of_chi2(df, stat_models, reduced=False, Nd=None, figname="chi_squared.pdf", save_tables=True):


    if reduced:
        _chi2_col_ = "reduced chi-squared"
    else:
        _chi2_col_ = "chi-squared"


    results = df.copy()
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    #stat_models = ["Bayesian","Outliers","GaussianSP","Gaussian"]
    #stat_models = ["Bayesian","OutliersSP","Outliers","Gaussian"]
    #stat_models = ["Bayesian","SinglePrior","Outliers","Gaussian"]
    positions = [(0,0), (0,1), (1,0), (1,1)]
    for i,stat_model in enumerate(stat_models):
        ax4 = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)
        if stat_model == "Bayesian":
            results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        nreplica = results_df["nreplica"].to_numpy(dtype=int)
        nreplica = np.bincount(nreplica).argmax()
        if nreplica == 0:
            nreplica = results_df["nreplica"].to_numpy(dtype=int).max()

        results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")

        #micro_pops = df['micro pops'].to_numpy()[0]
        #prior_pops = df['prior micro pops'].to_numpy()[0]

        biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if _chi2_col_==x]
        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
        columns = ["nsteps","nreplica","nlambda"]
        colors = ["grey", "orange", "g", "r"]

        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        data = data.reset_index().pivot("FF", "nstates", _chi2_col_)

        #if save_tables: data.to_latex(f"{figname.split('.')[0]}_{stat_model}.tex")
        if save_tables: data.to_html(f"{figname.split('.')[0]}_{stat_model}.html")
        #print(data)

        #data = np.array([FF, nstates])
        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f") #, label=col, vmin=-20, vmax=0)

        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)

        if positions[i] == (0,0):
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if positions[i] == (0,1):
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if positions[i] == (1,1):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)

        if Nd:
            ax4.set_title(fr"$\chi^{2}$ ({stat_model})", size=16)
        else:
            ax4.set_title(fr"$\chi^{2}$ ({stat_model})", size=16)

        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    mark.set_rotation(s=25)
        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')




                #mark.set_rotation(s=65)

    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)
# }}}

# plot_heatmap_of_chi_squared:{{{
def plot_heatmap_of_chi_squared(df, stat_models, figname="chi_squared.pdf"):

    results = df.copy()
    fig = plt.figure(figsize=(13, 16))
    gs = gridspec.GridSpec(3, 2)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    ax4 = plt.subplot(gs[(0,0)])
    biceps_score_cols = ["chi-squared_prior"]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    results_df = results.where(results["stat_model"]=="Bayesian")
    results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
    results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")

    columns = ["nsteps","nreplica","nlambda"]
    colors = ["grey", "orange", "g", "r"]
    data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
    data = data.reset_index().pivot("FF", "nstates", "chi-squared_prior")
    sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f") #, label=col, vmin=-20, vmax=0)

    for row in ax4.get_xticklabels():
        row.set_text(row.get_text().split(",")[0].split("(")[-1])

    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.set_ylabel("")
    ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                     labelright=False, bottom=True, top=True, left=True, right=True)
    ax4.set_xlabel("States", fontsize=16)
    ax4.set_title(fr"$\chi^{2}$ from prior", size=16)

    # Setting the ticks and tick marks
    ticks = [ax4.xaxis.get_minor_ticks(),
             ax4.xaxis.get_major_ticks()]
    marks = [ax4.get_xticklabels(),
            ax4.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            #mark.set_rotation(s=65)


    #stat_models = ["Bayesian","Outliers","GaussianSP","Gaussian"]
#    stat_models = ["Bayesian","OutliersSP","SinglePrior","Outliers","Gaussian"]

    positions = [(0,1), (1,0), (1,1), (2,0), (2,1)]
    for i,stat_model in enumerate(stat_models):
        ax4 = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)
        if stat_model == "Bayesian":
            results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        nreplica = results_df["nreplica"].to_numpy(dtype=int)
        nreplica = np.bincount(nreplica).argmax()
        if nreplica == 0:
            nreplica = results_df["nreplica"].to_numpy(dtype=int).max()

        results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")

        biceps_score_cols = ["chi-squared"]
        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
        columns = ["nsteps","nreplica","nlambda"]
        colors = ["grey", "orange", "g", "r"]

        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        data = data.reset_index().pivot("FF", "nstates", "chi-squared")
        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f") #, label=col, vmin=-20, vmax=0)

        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)

        if (positions[i] == (0,0)) or (positions[i] == (1,0)):
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if (positions[i] == (0,1)) or (positions[i] == (1,1)):
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        #if positions[i] == (1,1):
        if positions[i] == (2,1):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)

        ax4.set_title(fr"$\chi^{2}$ ({stat_model})", size=16)
        #ax4.xaxis.set_visible(False)


        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                #mark.set_rotation(s=65)
        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')




    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)
# }}}

# plot_heatmap_of_DKL:{{{
def plot_heatmap_of_DKL(df, stat_models, figname="BICePs_Scores.pdf"):
    results = df.copy()
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    positions = [(0,0), (0,1), (1,0), (1,1)]
    for i,stat_model in enumerate(stat_models):
        ax4 = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)
        if stat_model == "Bayesian":
            results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        nreplica = results_df["nreplica"].to_numpy(dtype=int)
        results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")
        biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "D_KL" in str(x)]
        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
        columns = ["nsteps","nreplica","nlambda"]
        colors = ["grey", "orange", "g", "r"]
        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        data = data.reset_index()
        data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
        data = data.pivot("FF", "nstates", "D_KL")

        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f") #, label=col, vmin=-20, vmax=0)

        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)

        if positions[i] == (0,0):
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if positions[i] == (0,1):
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if positions[i] == (1,1):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)

        #ax4.set_ylabel("BICePs Score", fontsize=16)
        ax4.set_title(r"$D_{KL}$ (%s)"%(stat_model), size=16)
        #ax4.xaxis.set_visible(False)


        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)
        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')



    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)
# }}}


# plot_heatmaps_of_column:{{{
def plot_heatmaps_of_columns(df, stat_models, columns, figname="heatmaps.pdf", title=None,
                      title_fontsize=20, title_position=(0.5,0.98), annot_fontsize=13,
                      save_tables=True, cbar_loc="right", grid=(2, 2),
                      positions=[(0,0), (0,1), (1,0), (1,1)],
                      figsize=(12, 12)):

    df["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in df["FF"].to_numpy()]
    results = df.copy()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid[0], grid[1])#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    left_cols, bottom_cols, other_cols = biceps.toolbox.get_seperate_columns(positions)
    for i,stat_model in enumerate(stat_models):
        print(stat_model)
        ax4 = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)
        if stat_model == "Bayesian":
            results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        nreplica = results_df["nreplica"].to_numpy(dtype=int)

        results_df = results_df.groupby(["FF", "nstates"]).agg("mean")
        data = results_df[columns[i]].reset_index().groupby(["FF","nstates"]).agg("mean")
        data = data.reset_index()
        data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
        data = data.pivot("FF", "nstates", columns[i])

        #if save_tables: data.to_html(f"{figname.split('.')[0]}_{stat_model}.html")
        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True,location=cbar_loc)) #, label=col, vmin=-20, vmax=0)
        cbar = ax4.collections[0].colorbar
        # here set the labelsize by 14
        cbar.ax.tick_params(labelsize=14)


        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)
        if np.where(positions[i] == np.array(left_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(other_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(bottom_cols))[0] != []:
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)

        if positions[i] == (grid[0]-1, 0):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)



        #ax4.set_title(f"BICePs Scores ({stat_model})", size=16)
        ax4.set_title(f"{stat_model}", size=16)
        #ax4.xaxis.set_visible(False)


        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)

        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')
    if title_position != None:
        if title != None:
            fig.suptitle(title, fontweight="bold",
                     x=title_position[0], y=title_position[1], size=title_fontsize)
    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)
# }}}

# plot_heatmap_of_biceps_scores:{{{
def plot_heatmap_of_biceps_scores(df, stat_models, figname="BICePs_Scores.pdf",
                      title_fontsize=20, title_position=(0.5,0.98), annot_fontsize=13,
                      save_tables=True, cbar_loc="right", grid=(2, 2),
                      positions=[(0,0), (0,1), (1,0), (1,1)],
                      figsize=(12, 12)):

    df["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in df["FF"].to_numpy()]
    results = df.copy()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid[0], grid[1])#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    left_cols, bottom_cols, other_cols = biceps.toolbox.get_seperate_columns(positions)
    for i,stat_model in enumerate(stat_models):
        print(stat_model)
        ax4 = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)
        if stat_model == "Bayesian":
            results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        nreplica = results_df["nreplica"].to_numpy(dtype=int)
        #nreplica = np.bincount(nreplica).argmax()
        #if nreplica == 0:
        #    nreplica = results_df["nreplica"].to_numpy(dtype=int).max()

        results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")
        #print(results_df.columns.to_list())

        biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in str(x)]
        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
        columns = ["nsteps","nreplica","nlambda"]
        colors = ["grey", "orange", "g", "r"]

        #print(results_df)
        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        #print(data)
        data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()#/nreplica
        data = data.reset_index()
        data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
        data = data.pivot("FF", "nstates", "BICePs Score lam=1")

        #if save_tables: data.to_latex(f"{figname.split('.')[0]}_{stat_model}.tex")
        if save_tables: data.to_html(f"{figname.split('.')[0]}_{stat_model}.html")

        #data = np.array([FF, nstates])
        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True,location=cbar_loc)) #, label=col, vmin=-20, vmax=0)
        cbar = ax4.collections[0].colorbar
        # here set the labelsize by 14
        cbar.ax.tick_params(labelsize=14)


        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)
        if np.where(positions[i] == np.array(left_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(other_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(bottom_cols))[0] != []:
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)

        if positions[i] == (grid[0]-1, 0):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)



        #ax4.set_title(f"BICePs Scores ({stat_model})", size=16)
        ax4.set_title(f"{stat_model}", size=16)
        #ax4.xaxis.set_visible(False)


        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)

        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')
    if title_position != None:
        fig.suptitle(r"BICePs Scores ($\lambda=0 \rightarrow \lambda=1$)", fontweight="bold",
                     x=title_position[0], y=title_position[1], size=title_fontsize)
    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)
# }}}


# plot_heatmap_of_biceps_scores_std:{{{
def plot_heatmap_of_biceps_scores_std(df, stat_models, figname="BICePs_Scores.pdf",
                      title_fontsize=20, title_position=(0.5,0.98), annot_fontsize=13,
                      save_tables=True, cbar_loc="right", grid=(2, 2), use_SEM=False,
                      positions=[(0,0), (0,1), (1,0), (1,1)],
                      figsize=(12, 12)):

    df["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in df["FF"].to_numpy()]
    results = df.copy()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid[0], grid[1])#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    left_cols, bottom_cols, other_cols = biceps.toolbox.get_seperate_columns(positions)
    for i,stat_model in enumerate(stat_models):
        #print(stat_model)
        ax4 = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)
        if stat_model == "Bayesian":
            results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        nreplica = results_df["nreplica"].to_numpy(dtype=int)
        #nreplica = np.bincount(nreplica).argmax()
        #if nreplica == 0:
        #    nreplica = results_df["nreplica"].to_numpy(dtype=int).max()

        if use_SEM:
            results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("sem")
        else:
            results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("std")
        #print(results_df.columns.to_list())

        biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in str(x)]
        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
        columns = ["nsteps","nreplica","nlambda"]
        colors = ["grey", "orange", "g", "r"]

        #print(results_df)
        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        #print(data)
        data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()#/nreplica
        data = data.reset_index()
        data["nstates"] = [int(val) for val in data["nstates"].to_numpy()]
        data = data.pivot("FF", "nstates", "BICePs Score lam=1")
        #if save_tables: data.to_latex(f"{figname.split('.')[0]}_{stat_model}.tex")
        if save_tables: data.to_html(f"{figname.split('.')[0]}_{stat_model}.html")

        #data = np.array([FF, nstates])
        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":annot_fontsize},
                   cbar_kws=dict(use_gridspec=True,location=cbar_loc)) #, label=col, vmin=-20, vmax=0)
        cbar = ax4.collections[0].colorbar
        # here set the labelsize by 14
        cbar.ax.tick_params(labelsize=14)

        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)

        if np.where(positions[i] == np.array(left_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(other_cols))[0] != []:
            ax4.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("")

        if np.where(positions[i] == np.array(bottom_cols))[0] != []:
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)

        if positions[i] == (grid[0]-1, 0):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)
            ax4.set_xlabel("States", fontsize=16)


        #ax4.set_title(f"BICePs Scores ({stat_model})", size=16)
        ax4.set_title(f"{stat_model}", size=16)
        #ax4.xaxis.set_visible(False)


        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    #mark.set_rotation(s=25)
                    mark.set_rotation(s=0)

        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')

    if title_position != None:
        fig.suptitle(r"BICePs Scores SEM ($\lambda=0 \rightarrow \lambda=1$)", fontweight="bold",
        #fig.suptitle(f"BICePs Scores Std", fontweight="bold",
                     x=title_position[0], y=title_position[1], size=title_fontsize)
    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)

# }}}


# plot_heatmap_of_biceps_scores_horizontal:{{{
def plot_heatmap_of_biceps_scores_horizontal(df, stat_models, figname="BICePs_Scores.pdf", save_tables=True):


    df["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in df["FF"].to_numpy()]
    results = df.copy()

    fig = plt.figure(figsize=(20,7))
    gs = gridspec.GridSpec(1, 4)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    positions = [(0,0), (0,1), (0,2), (0,3)]
    for i,stat_model in enumerate(stat_models):
        print(stat_model)
        ax4 = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)


        if stat_model == "Bayesian":
            results_df = results_df.iloc[np.where((results_df["nreplica"]==1))[0]]
        nreplica = results_df["nreplica"].to_numpy(dtype=int)
        results_df = results_df.groupby(["FF", "nstates", "uncertainties"]).agg("mean")
        biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in str(x)]
        score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
        columns = ["nsteps","nreplica","nlambda"]
        colors = ["grey", "orange", "g", "r"]

        print(results_df)
        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        print(data)
        data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()#/nreplica
        data = data.reset_index().pivot("FF", "nstates", "BICePs Score lam=1")
        #if save_tables: data.to_latex(f"{figname.split('.')[0]}_{stat_model}.tex")
        if save_tables: data.to_html(f"{figname.split('.')[0]}_{stat_model}.html")

        #data = np.array([FF, nstates])
        sb.heatmap(data, ax=ax4, linewidths=.5, annot=True, fmt="0.2f", annot_kws={"fontsize":13},
                   cbar_kws=dict(use_gridspec=True,location="top")) #, label=col, vmin=-20, vmax=0)
        cbar = ax4.collections[0].colorbar
        # here set the labelsize by 14
        cbar.ax.tick_params(labelsize=14)

        for row in ax4.get_xticklabels():
            row.set_text(row.get_text().split(",")[0].split("(")[-1])

        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.set_ylabel("")
        ax4.set_xlabel("States", fontsize=16)

        if positions[i] == (0,0):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                             labelright=False, bottom=True, top=True, left=True, right=True)

        if positions[i] == (0,1):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)

        if positions[i] == (0,2):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)

        if positions[i] == (0,3):
            ax4.tick_params(labelbottom=True, labeltop=False, labelleft=False,
                             labelright=False, bottom=True, top=True, left=True, right=True)



        #ax4.set_ylabel("BICePs Score", fontsize=16)
        ax4.set_title(f"BICePs Scores ({stat_model})", size=16)
        #ax4.xaxis.set_visible(False)


        # Setting the ticks and tick marks
        ticks = [ax4.xaxis.get_minor_ticks(),
                 ax4.xaxis.get_major_ticks()]
        marks = [ax4.get_xticklabels(),
                ax4.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    mark.set_rotation(s=25)
        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')


    #fig.suptitle(f"BICePs Scores", size=16)
    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)
# }}}


# Simple Plot:{{{

def simple_plot(x,y,xlabel='x',ylabel='y',name=None,size=111,Type='scatter',
        color=False,fig_size=(12,10),invert_x_axis=False,fit=False,order=None,
        xLine=None,yLine=None,
        annotate_text=None,text_x=0,text_y=0,
        annotate_x=0,annotate_y=0,
        arrow='->', plot_ref_line=True):
    '''
    Returns a plot and saves it to the working directory
    unless stated otherwise.
    x = numpy array
    y = numpy array
    xlabel,ylabel,name = strings
    size = axis size
    color = color of line
    fig_size = (x,y)
    '''
    marks = ['o','D','2','>','*',',',"4","8","s",
             "p","P","*","h","H","+","x","X","D","d"]
    colors = ['k','b','g','r','c','m','y',
              'k','b','g','r','c','m','y']

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(size)

    if plot_ref_line:
        _min = np.min([x.min(), y.min()])
        _max = np.max([x.max(), y.max()])
        ax.plot([_min, _max], [_min, _max], 'k-', label="_nolegend_")

    if Type=='scatter':
        if color==False:
            ax.scatter(x,y,color='k')
        else:
            ax.scatter(x,y,color=color)

    if Type=='line':
        if color==False:
            ax.plot(x,y,'k')
        else:
            ax.plot(x,y,color)
    if fit==True:
        #ax.plot(x,y,label="_nolegend_")
        z = np.polyfit(x, y, order)
        n_coeff = len(z)
        #################################################
        p = np.poly1d(z)
        ax.plot(x,p(x),"k:", label="_nolegend_")
        # the line equation:
        print('LINEST data:')
        if order==1:
            print("y=%.6f*x+(%.6f)"%(z[0],z[1]))
            print(scipy.stats.linregress(x,y))
        elif order==2:
            print("y=%.6f*x**2.+(%.6f)*x+(%.6f)"%(z[0],z[1],z[2]))
        elif order==3:
            print("y=%.6f*x**3.+(%.6f)*x**2.+(%.6f)*x+(%.6f)"%(z[0],z[1],z[2],z[3]))
        elif order==4:
            print("y=%.6fx**4.+(%.6f)*x**3.+(%.6f)*x**2.\
                    +(%.6f)*x+(%.6f)"%(z[0],z[1],z[2],z[3],z[4]))
        else:
            print('You need to add a greater order of polynomials to the script')

        print(scipy.stats.chi2(y))
        print(scipy.stats.ttest_ind(x, y, axis=0, equal_var=True))
        #eq = "y=%.6fx+(%.6f)"%(z[0],z[1])

    ax.set_xlabel('%s'%xlabel, fontsize=16)
    ax.set_ylabel('%s'%ylabel, fontsize=16)
    # Does the x-axis need to be reverse?
    if invert_x_axis==True:
        plt.gca().invert_xaxis()
    # Setting the ticks and tick marks
    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    if annotate_text != None:
        ax.annotate(r'%s'%annotate_text,
                xy=(annotate_x,annotate_y),xytext=(text_x,text_y),
                #arrowprops=dict(facecolor='black', arrowstyle=arrow),
                fontsize=16)
    if xLine!=None:
        ax.axhline(xLine)
    if yLine!=None:
        ax.axhline(yLine)
    fig.tight_layout()
    if name==None:
        pass
    else:
        fig.savefig('%s'%name)
    fig.show()


# }}}

#:}}}



experimental_folded = u.ufloat(0.61029, 0.03375)
val = experimental_folded.nominal_value
dev = experimental_folded.std_dev
experimental_lower, experimental_upper = val-dev, val+dev


dpi=600
#sys_name = "CLN001" # "CLN025"
sys_name = "./"
data_dir = "./CLN001"
karplus_key = "Bax2007"
#karplus_key = "Habeck"
pub_images = 1


analysis_dir = "../analysis_all_data"
#analysis_dir = "../analysis_noe_only"
#analysis_dir = "../analysis_j_only"
#analysis_dir = "../analysis_cs_only"



FF_list = ["AMBER14SB","AMBER99SB-ildn","CHARMM27","AMBER99",
    "AMBER99SBnmr1-ildn","CHARMM36","AMBER99SB","CHARMM22star","OPLS-aa"]

stat_models = ["Bayesian", "GB", "Students", "Gaussian"]
new_sm = ["Single replica", "Good-Bad", "Student's", "Gaussian"]

#stat_models = ["Bayesian", "GaussianSP", "GB", "Students"]
#new_sm = ["Single replica", "GaussianSP", "Good-Bad", "Student's"]


_stat_models = stat_models.copy()
# NOTE:
plot_population_bar_charts = 0
if plot_population_bar_charts:
    stat_models = ["Bayesian", "GB", "Students", "Gaussian"]
    new_sm = ["Single replica", "Good-Bad", "Student's", "Gaussian"]

use_SEM = 1
if plot_population_bar_charts:
    # NOTE: Plot Bar charts of Populations across models: {{{
    for FF in FF_list:
        for clusters in [5,10,50,75,100,500]:
            files = biceps.toolbox.get_files(f"{analysis_dir}/{FF}/nclusters_{clusters}/*/*/*__reweighted_populations.csv")
            # Get the files for prior pops and assignments (microstate --> macrostate)
            # ..for each FF/microstate clustering
            assignment_files = biceps.toolbox.get_files(f"{data_dir}/{sys_name}/{FF}/nclusters_{clusters}/inverse_distances_k*_msm_assignments.csv")
            prior_pops_files = biceps.toolbox.get_files(f"{data_dir}/{sys_name}/{FF}/nclusters_{clusters}/inverse_distances_k*_msm_pops.csv")
            assignment_df = pd.read_csv(assignment_files[0], index_col=0)
            prior_df =  pd.read_csv(prior_pops_files[0], index_col=0)
            # Get a dataframe of prior macrostate populations (from MSM)
            microstates = assignment_df.index.to_numpy()
            macrostates = assignment_df[assignment_df.columns[0]].to_numpy()
            prior_pops = prior_df[prior_df.columns[0]].to_numpy()
            ntop = len(prior_pops)
            topN_pops = prior_pops[np.argsort(prior_pops)[-ntop:]]
            topN_labels = [np.where(topN_pops[i]==prior_pops)[0][0] for i in range(len(topN_pops))]
            macro_asignment = []
            for i,state in enumerate(topN_labels):
                index = np.where(microstates == state)[0]
                macrostate = int(macrostates[int(index)])
                macro_asignment.append({"macrostate": macrostate, "microstate":state, "population":topN_pops[i]})
            df = pd.DataFrame(macro_asignment)
            grouped = df.groupby(["macrostate"])
            df = df.set_index("macrostate")
            macrostate_populations = grouped.sum().drop("microstate", axis=1)
            print(macrostate_populations)


            #exit()

            #energies = np.loadtxt(f'{sys_name}/{sys_name}_energy_microstate.txt')/4184.*(6.022e23)  # Joules to kcal/mol
            #energies = energies/0.5959   # convert to reduced free energies F = f/kT
            #energies -= energies.min()  # set ground state to zero, just in case

            ## CLN001 MSM
            #State1 = 9.2*0.01   #random
            #State2 = 5.1*0.01   #helix
            #State3 = 9.3*0.01   #misfolded
            #State4 = 76.5*0.01  #folded

            results = []
            results.append({
                "stat_model": "Experiment", "nreplicas": 0, "nlambdas": 0, "nsteps": 0, "data_uncertainty": "none",
                #0: 0.83, 1:0, 2:0
                0: experimental_folded.nominal_value, 1:0, 2:0
                })
            #results.append({
            #    "stat_model": "MSM", "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
            #    4: 0.765, 3:0.093, 2:0.051, 1:0.092
            #    })
            results.append({
                "stat_model": "MSM", "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
                **macrostate_populations.to_dict()[macrostate_populations.columns[0]]
                })
            print(results)
            #exit()

            for file in files:
                print(file)
                stat_model = file.split("/")[-1].split("_single_sigma")[0].split("_multiple_sigma")[0]
                # NOTE: Removing SinglePrior model from the plots
                if stat_model not in stat_models: continue
                data_uncertainty = file.split("/")[-1].split("_sigma")[0].split("_")[-1]
                nreplicas = int(file.split("/")[-2].split("_replicas")[0].split("_")[-1])
                nsteps = int(file.split("/")[-2].split("_steps")[0].split("_")[-1])
                nlambdas = int(file.split("/")[-2].split("_lam")[0].split("_")[-1])
                df = pd.read_csv(file)
                grouped = df.groupby(["macrostate"])
                df = df.set_index("macrostate")
                pop_dict = grouped.sum()["population"].to_dict()
                results.append({
                    "FF": FF,
                    "stat_model": stat_model,
                    "nreplicas": nreplicas,
                    "nlambdas": nlambdas,
                    "nsteps": nsteps,
                    "data_uncertainty": data_uncertainty,
                    **pop_dict
                    })

            df = pd.DataFrame(results)

            #grouped = df.groupby([4,3,2,1])
            grouped = df.groupby(["stat_model","nreplicas","data_uncertainty", "nsteps"]).agg("mean")
            if use_SEM:
                grouped_err = df.groupby(["stat_model","nreplicas","data_uncertainty", "nsteps"]).agg("sem")
            else:
                grouped_err = df.groupby(["stat_model","nreplicas","data_uncertainty", "nsteps"]).agg("std")


            model_order = ["Experiment", "MSM"]+stat_models #, "BayesianModel", "Good-Bad", "StudentsModel", "GaussianModel"]
            model_labels = ["Exp.", "MSM"]+new_sm

            #model_order = ["Experiment", "MSM", "Bayesian", "GaussianGB","Gaussian","Students_new","Outliers"]
            #model_labels = ["Exp", "MSM", "Bayesian", "G-B","Gaussian","Student's","Outliers"]
            indices = grouped.index
            new_idx = []
            for model in model_order:
                for i,idx in enumerate(indices):
                    if idx[0] == model:
                        new_idx.append(i)
                        break

            grouped = pd.concat([grouped.iloc[[i]] for i in new_idx])
            grouped_err = pd.concat([grouped_err.iloc[[i]] for i in new_idx])
            print(grouped)

            df = grouped#.reset_index()
            df_err = grouped_err.reset_index()
            df_err[0] = np.concatenate([np.array([dev]),df_err[0].to_numpy()[1:]])
            #print(df_err)
            #exit()
            print(df)
            #exit()
            yerr = df_err[[0,1,2]].to_numpy().T
            print(yerr)
            y = df[[0,1,2]].to_numpy()
            #labels = df["stat_model"].to_numpy()
            #print(y)
            #exit()

            #try:
            #ax = grouped.bar([4,3,2,1], alpha=0.5, edgecolor='black', linewidth=1.2, color="b", figsize=(14, 6))

#            fig = plt.figure(figsize=(14, 6))
#            ax = fig.add_subplot(111)

            df = df.fillna(0.0)
            yerr = np.nan_to_num(yerr, nan=0.0)
            if pub_images:
                ax = df[[0,1,2]].plot.bar(yerr=yerr, figsize=(10, 4), rot=65, ecolor='black', capsize=10)
            else:
                ax = df[[0,1,2]].plot.bar(yerr=yerr,  figsize=(10, 4), rot=65, ecolor='black', capsize=10)
            df = df.reset_index()
            print(df.iloc[np.where(df["stat_model"].to_numpy() == "Experiment")[0]].to_dict()[0])
            ax.axhline(y=df.iloc[np.where(df["stat_model"].to_numpy() == "Experiment")[0]].to_dict()[0][0], color='k', linestyle="--", label="_nolegend_")
            #ax.axhline(y=val, color='k', linestyle="--", label="_nolegend_")
            ax.axhspan(experimental_lower, experimental_upper, facecolor='orange', alpha=0.4, label="_nolegend_")
            xticklabels = []
            #model_labels = df["stat_model"].to_numpy()
            replica_labels = df["nreplicas"].to_numpy()
            step_labels = df["nsteps"].to_numpy()
            #print(results)
            for i in range(len(model_labels)):
                if replica_labels[i] == 0:
                    xticklabels.append(model_labels[i])
                if replica_labels[i] != 0:
                    if pub_images:
                        xticklabels.append(str(model_labels[i]))
                    else:
                        xticklabels.append(str(str(model_labels[i])+"\n"+str(step_labels[i])+" steps & "+str(replica_labels[i])+" replica"))
            ax.set_xticklabels([f"{x}" for x in xticklabels])
            ax.set_xlabel(r"", fontsize=16)
            ax.set_ylabel(r"Populations", fontsize=16)
            ax.set_ylim(0,1.1)


            # Setting the ticks and tick marks
            ticks = [ax.xaxis.get_minor_ticks(),
                     ax.xaxis.get_major_ticks()]
            marks = [ax.get_xticklabels(),
                    ax.get_yticklabels()]
            for k in range(0,len(ticks)):
                for tick in ticks[k]:
                    tick.label.set_fontsize(16)
            for k in range(0,len(marks)):
                for mark in marks[k]:
                    mark.set_size(fontsize=16)
                    if k == 0:
                        mark.set_rotation(s=15)

            #if sys_name == "CLN001": ax.legend(["0: folded","1: misfolded", "2: unfolded"], loc="best", fontsize=16)
            #if sys_name == "CLN025": ax.legend(["Exp", "4: folded","3: misfolded", "2: unfolded", "1: helix"], loc="best")#, fontsize=16)
            ax.legend(["Folded","Misfolded", "Unfolded"], loc="best", fontsize=16)
            fig = ax.get_figure()
            fig.tight_layout()
            biceps.toolbox.mkdir(f"{clusters}_microstates")
            fig.savefig(f"{clusters}_microstates/{sys_name}_{FF}_populations.pdf", dpi=dpi)
            #print(f"RMSE = {df['error'].mean()} ± {df['error'].std()}")
            #except(Exception) as e:
            #    print(e)
            #    exit()
            #    continue
    # }}}
    exit()




#files = biceps.toolbox.get_files(f"{analysis_dir}/*/nclusters_*/*/*/results.pkl")
#
#for file in files:
#    if "GaussianSP" in file:
#        #print(list(set(results["stat_model"].to_numpy())))
#        df = pd.read_pickle(file)
#        #print(df["stat_model"].to_numpy())
#        df["stat_model"] = ["GaussianSP"]
#        #print(df["stat_model"].to_numpy())
#        df.to_pickle(file)
#        #exit()
#
#exit()


# get files for all force fields and plot biceps score comparison
files = biceps.toolbox.get_files(f"{analysis_dir}/*/nclusters_*/*/*/results.pkl")
#files = biceps.toolbox.get_files(f"{sys_name}/*/nclusters_*/*/*/results.pkl")
#files = biceps.toolbox.get_files(f"{sys_name}/*/nclusters_*/*/*/results.pkl")[::4] # for testing...
results = []
for file in files:
#for file in files[::10]:
    for stat_model in stat_models:
        if stat_model in file:
            add_file = True
            break
        else:
            add_file = False

    if add_file:
        # NOTE: there is a lot of hardcoding in this script... here is just one example of it..
        if len(file.split("trial_")) >= 2:
            trial = int(file.split("trial_")[-1].split("/results.pkl")[0])
        else:
            trial = 0
        print(file)
        df = pd.read_pickle(file)
        df["trial"] = trial
        results.append(df)
results = pd.concat(results)
#print(results)
#print(list(set(results["stat_model"].to_numpy())))
#exit()

results["nstates"] = [int(n) for n in results["nstates"].to_numpy()]

print(results.columns.to_list())

for i in range(len(new_sm)):
    results = biceps.toolbox.change_stat_model_name(results, sm=stat_models[i], new_sm=new_sm[i])

stat_models = new_sm.copy()


df = results.copy()
df["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in df["FF"].to_numpy()]

for i,stat_model in enumerate(stat_models):
    print("\n\n\n\n\n")
    print(stat_model)
    #results_df = results.where(results["stat_model"]==stat_model)
    results_df = results.iloc[np.where(stat_model == results["stat_model"].to_numpy())[0]]
    _results_df = results_df.groupby(["FF", "nstates"]).agg("mean")
    biceps_score_cols = [x for i,x in enumerate(_results_df.columns.to_list()) if "BICePs Score" in str(x)]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]

    __results_df = results_df.groupby(["FF", "nstates"]).agg("std")
    biceps_score_cols = [x for i,x in enumerate(__results_df.columns.to_list()) if "BICePs Score" in str(x)]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    std_error = __results_df[score_cols[-1]].to_numpy()
    _results_df[score_cols[-1]+" std error"] = std_error
    print(_results_df[[score_cols[-1],score_cols[-1]+" std error"]])
    print("Max std: ", np.nanmax(_results_df[score_cols[-1]+" std error"].to_numpy()))
    print("Mean std: ", np.nanmean(_results_df[score_cols[-1]+" std error"].to_numpy()))



print(list(set(results["FF"].to_numpy())))

#exit()
#print(results)
#results.to_csv("table.csv")
#exit()
biceps.toolbox.mkdir("figures")


suffix = analysis_dir.split("analysis_")[-1]

outdir = "biceps_score_results"
biceps.toolbox.mkdir(outdir)

#results.to_pickle(f"{outdir}/biceps_score_results_{suffix}.pkl")
#exit()


plot_BS_heatmap = 0
if plot_BS_heatmap:
    plot_heatmap_of_biceps_scores(results, stat_models, figname=f"figures/biceps_scores.pdf",
                          grid=(2, 2), positions=[(0,0), (0,1), (1,0), (1,1)], figsize=(12, 12))
                          #grid=(3, 2), positions=[(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],figsize=(12, 12))
                          #grid=(4, 2), positions=[(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)],figsize=(12, 14))

    plot_heatmap_of_biceps_scores(results, stat_models, figname=f"figures/biceps_scores_horizontial.pdf",
                          title_fontsize=20, title_position=(0.5, 0.96), annot_fontsize=14,
                          #title_fontsize=20, title_position=None, annot_fontsize=14,
                          grid=(1, 4), positions=[(0,0), (0,1), (0,2), (0,3)], figsize=(20,7), cbar_loc="top")

    plot_heatmap_of_biceps_scores_std(results, stat_models, figname=f"figures/biceps_scores_std.pdf",  use_SEM=True,
                          grid=(2, 2), positions=[(0,0), (0,1), (1,0), (1,1)], figsize=(12, 12))
                          #grid=(3, 2), positions=[(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],figsize=(12, 12))

    plot_heatmap_of_biceps_scores_std(results, stat_models, figname=f"figures/biceps_scores_std_horizontial.pdf",
                          title_fontsize=20, title_position=(0.5, 0.96), annot_fontsize=14,
                          #title_fontsize=20, title_position=None, annot_fontsize=14,  use_SEM=True,
                          grid=(1, 4), positions=[(0,0), (0,1), (0,2), (0,3)], figsize=(20,7), cbar_loc="top")


#    #plot_heatmap_of_biceps_scores(results, stat_models=["Bayesian", "OutliersSP"], figname=f"figures/biceps_scores_Bayes_OutliersSP.pdf",
#    plot_heatmap_of_biceps_scores(results, stat_models=["Bayesian"], figname=f"figures/biceps_scores_Bayes.pdf",
#                          title_fontsize=20, title_position=None, annot_fontsize=14,
#                          #grid=(1, 2), positions=[(0,0), (0,1)], figsize=(20,7), cbar_loc="top")
#                          grid=(1, 1), positions=[(0,0)], figsize=(7,7), cbar_loc="top")
#
#    plot_heatmap_of_biceps_scores_std(results, stat_models=["Bayesian"], figname=f"figures/biceps_scores_Bayes_std.pdf",  use_SEM=True,
#                          grid=(1, 1), positions=[(0,0)], annot_fontsize=14, title_position=None, figsize=(7, 7), cbar_loc="top")
#                          #grid=(3, 2), positions=[(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],figsize=(12, 12))




    exit()

#plot_heatmap_of_biceps_scores_horizontal(results, stat_models, figname=f"figures/biceps_scores_pnas.pdf")
#plot_heatmap_of_DKL(results, stat_models, figname=f"figures/KL_divergence.pdf")
#exit()

rankings_table = []
rankings_table_columns = []

#########################################
plot_chi_squared = 1 #True #False

pro_and_gly_only = 0
without_pro_and_gly = 0
without_pro = 0
without_gly = 0

if pro_and_gly_only: ext = "_pro_and_gly_only"
elif without_pro_and_gly: ext = "_without_pro_and_gly"
elif without_pro: ext = "_without_pro"
elif without_gly: ext = "_without_gly"
else: ext = ""


if plot_chi_squared:

    # TODO: put the prior forward model uncertainties inside the results dataframe
    df = pd.read_pickle(f"{data_dir}/prior_std_error_in_forward_model.pkl")

    sigma_H = 0.2351 # comes from shiftx2 performance
    sigma_HA = 0.1081 # comes from shiftx2 performance
#    sigma_J = 0.34   # comes from uncertainty in karplus parameters (Habeck2005)
    sigma_J = 0.36   # comes from uncertainty in karplus parameters (Bax2007)
    sigma_NOE = 1.0  # set to 1 due to experimental upper and lower limit
    sigma_J_column, sigma_noe_column, sigma_cs_column = [], [], []
    chi2, chi2_prior = [], []
    biceps_chi2 = []
    reduced_prior, reduced, reduced_biceps = [],[],[]
    #print(results.columns.to_list())
    #exit()
    for row in results.iterrows():
        #if row[1][9] != "Bayesian": continue
        #print(row[1])
        #exit()
        FF = row[1][0] # ff
        nstates = row[1][3] # nstates
        system = row[1][1] # system
        # search for row
        print(f"{system}/{FF}/{nstates}")
        sel_row = df.iloc[np.where( (df["sys_name"] == system) & (df["forcefield"] == FF) & (df["nstates"] == nstates))[0]]

        #print(sel_row)
        sigma_noe = float(sel_row["std"].to_numpy()) # FIXME: if no longer a float
        #sigma_noe = sel_row["std"].to_numpy()[0]
        #sigma_noe = sigma_NOE # FIXME

        # append to lists
        sigma_J_column.append(sigma_J)
        sigma_cs_column.append(sigma_H)
        sigma_noe_column.append(sigma_noe)
        fm_data_dir = f"{data_dir}/{sys_name}/{FF}/nclusters_{nstates}" # FIXME: this is user controlled & might be different in the future
        print(fm_data_dir)
        ############################################################################
        ############################################################################
        ############################################################################
        #pops = row[1][19] # reweighted pops
        pops = row[1][np.where('micro pops' == results.columns.to_numpy())[0]].values[0]
#        print(pops)
#        prior_pops = row[1][21] # prior pops
        prior_pops = row[1][np.where('prior micro pops' == results.columns.to_numpy())[0]].values[0]

        noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{fm_data_dir}/CS_J_NOE_{karplus_key}/*.noe")]

        if pro_and_gly_only:
            dfs = []
            cols = ["res1", "res2"]
            for _df in noe:
                dfs.append(_df.loc[_df[cols].apply(lambda x: x.str.contains(r'PRO4|GLY1|GLY10|GLY7')).any(axis=1)])
            noe = dfs.copy()

        elif without_pro_and_gly:
            dfs = []
            cols = ["res1", "res2"]
            for _df in noe:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'PRO4|GLY1|GLY10|GLY7')).any(axis=1)])
            noe = dfs.copy()

        elif without_pro:
            dfs = []
            cols = ["res1", "res2"]
            for _df in noe:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'PRO4')).any(axis=1)])
            noe = dfs.copy()

        elif without_gly:
            dfs = []
            cols = ["res1", "res2"]
            for _df in noe:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'GLY1|GLY10|GLY7')).any(axis=1)])
            noe = dfs.copy()
        else:
            pass



        #  Get the ensemble average observable
        noe_Exp = noe[0]["exp"].to_numpy()
        noe_model = [i["model"].to_numpy() for i in noe]
#        print(prior_pops)
        try:
            noe_prior = np.array([w*noe_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
        except(Exception) as e:
            continue
        noe_reweighted = np.array([w*noe_model[i] for i,w in enumerate(pops)]).sum(axis=0)

        noe_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}-{three2one(row[1]['res2'])}.{row[1]['atom_name2']}" for row in noe[0].iterrows()]
        noe_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2']] for row in noe[0].iterrows()])


        J = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{fm_data_dir}/CS_J_NOE_{karplus_key}/*.J')]
        if pro_and_gly_only:
            dfs = []
            cols = ["res1", "res2", "res3", "res4"]
            for _df in J:
                dfs.append(_df.loc[_df[cols].apply(lambda x: x.str.contains(r'PRO4|GLY1|GLY10|GLY7')).any(axis=1)])
            J = dfs.copy()

        elif without_pro_and_gly:
            dfs = []
            cols = ["res1", "res2", "res3", "res4"]
            for _df in J:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'PRO4|GLY1|GLY10|GLY7')).any(axis=1)])
            J = dfs.copy()


        elif without_pro:
            dfs = []
            cols = ["res1", "res2", "res3", "res4"]
            for _df in J:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'PRO4')).any(axis=1)])
            J = dfs.copy()

        elif without_gly:
            dfs = []
            cols = ["res1", "res2", "res3", "res4"]
            for _df in J:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'GLY1|GLY10|GLY7')).any(axis=1)])
            J = dfs.copy()
        else:
            pass


        #  Get the ensemble average observable
        J_Exp = J[0]["exp"].to_numpy()
        J_model = [i["model"].to_numpy() for i in J]
        J_prior = np.array([w*J_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
        J_reweighted = np.array([w*J_model[i] for i,w in enumerate(pops)]).sum(axis=0)
        J_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}\n{three2one(row[1]['res2'])}.{row[1]['atom_name2']}\n{three2one(row[1]['res3'])}.{row[1]['atom_name3']}\n{three2one(row[1]['res4'])}.{row[1]['atom_name4']}" for row in J[0].iterrows()]
        J_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2'], row[1]['atom_index3'], row[1]['atom_index4']] for row in J[0].iterrows()])


        cs = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{fm_data_dir}/CS_J_NOE_{karplus_key}/*.cs*')]
        if pro_and_gly_only:
            dfs = []
            cols = ["res1"]
            for _df in cs:
                dfs.append(_df.loc[_df[cols].apply(lambda x: x.str.contains(r'PRO4|GLY1|GLY10|GLY7')).any(axis=1)])
            cs = dfs.copy()

        elif without_pro_and_gly:
            dfs = []
            cols = ["res1"]
            for _df in cs:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'PRO4|GLY1|GLY10|GLY7')).any(axis=1)])
            cs = dfs.copy()


        elif without_pro:
            dfs = []
            cols = ["res1"]
            for _df in cs:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'PRO4')).any(axis=1)])
            cs = dfs.copy()
        elif without_gly:
            dfs = []
            cols = ["res1"]
            for _df in cs:
                dfs.append(_df.loc[~_df[cols].apply(lambda x: x.str.contains(r'GLY1|GLY10|GLY7')).any(axis=1)])
            cs = dfs.copy()
        else:
            pass
        print(f"len(noe[0]) = {len(noe[0])}")
        print(f"len(J[0]) = {len(J[0])}")
        print(f"len(cs[0]) = {len(cs[0])}")
        #exit()


        #  Get the ensemble average observable
        cs_Exp = cs[0]["exp"].to_numpy()
        cs_model = [i["model"].to_numpy() for i in cs]
        cs_atom_name1 = cs[0]["atom_name1"].to_numpy()

        cs_prior = np.array([w*cs_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
        cs_reweighted = np.array([w*cs_model[i] for i,w in enumerate(pops)]).sum(axis=0)

        cs_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}" for row in cs[0].iterrows()]
        cs_label_indices = np.array([[row[1]['atom_index1']] for row in cs[0].iterrows()])


        # NOTE: get chi-squared
        _sigma_H = np.ones(len(cs_Exp))*sigma_H
        _sigma_J = np.ones(len(J_Exp))*sigma_J
        #_sigma_noe = np.ones(len(noe_Exp))*sigma_noe # FIXME: if no longer a float
        #_sigma_noe = sigma_noe
        _sigma_noe = np.ones(len(noe_Exp))*sigma_NOE

        data = []
        noe_chi2_prior = compute_chi2(noe_Exp, noe_prior, _sigma_noe)
        noe_chi2 = compute_chi2(noe_Exp, noe_reweighted, _sigma_noe)
        noe_chi2_biceps = compute_chi2(noe_Exp, noe_reweighted, np.ones(len(noe_Exp)))

        for i in range(len(noe_reweighted)):
            data.append({"index":i,
                "reweighted obs":noe_reweighted[i], "prior obs":noe_prior[i], "exp obs":noe_Exp[i],
                         "sigma": _sigma_noe[i], "label":noe_labels[i]
                })

        J_chi2_prior = compute_chi2(J_Exp, J_prior, _sigma_J)
        J_chi2 = compute_chi2(J_Exp, J_reweighted, _sigma_J)
        J_chi2_biceps = compute_chi2(J_Exp, J_reweighted, np.ones(len(_sigma_J)))
        for i in range(len(J_reweighted)):
            data.append({"index":i,
                "reweighted obs":J_reweighted[i], "prior obs":J_prior[i], "exp obs":J_Exp[i],
                         "sigma": _sigma_J[i], "label":J_labels[i]
                })

        HA_indices = []
        H_indices = []
        for i in range(len(cs_reweighted)):
            #print(cs_atom_name1[i])
            if cs_atom_name1[i] == "HA":
                data.append({"index":i,
                    "reweighted obs":cs_reweighted[i], "prior obs":cs_prior[i], "exp obs":cs_Exp[i],
                             "sigma": sigma_HA, "label":cs_labels[i]
                    })
                HA_indices.append(i)
            elif cs_atom_name1[i] == "H":
                data.append({"index":i,
                    "reweighted obs":cs_reweighted[i], "prior obs":cs_prior[i], "exp obs":cs_Exp[i],
                             "sigma": sigma_H, "label":cs_labels[i]
                    })
                H_indices.append(i)
            else:
                print(f"Error: Atom name {cs_atom_name1[i]} is not one of HA or H!")

        _sigma_HA = np.ones(len(HA_indices))*sigma_HA
        _sigma_H = np.ones(len(H_indices))*sigma_H
        cs_HA_chi2_prior = compute_chi2(cs_Exp[HA_indices], cs_prior[HA_indices], _sigma_HA)
        cs_HA_chi2 = compute_chi2(cs_Exp[HA_indices], cs_reweighted[HA_indices], _sigma_HA)
        cs_HA_chi2_biceps = compute_chi2(cs_Exp[HA_indices], cs_reweighted[HA_indices], np.ones(len(_sigma_HA)))
        cs_H_chi2_prior = compute_chi2(cs_Exp[H_indices], cs_prior[H_indices], _sigma_H)
        cs_H_chi2 = compute_chi2(cs_Exp[H_indices], cs_reweighted[H_indices], _sigma_H)
        cs_H_chi2_biceps = compute_chi2(cs_Exp[H_indices], cs_reweighted[H_indices], np.ones(len(_sigma_H)))

        data1 = pd.DataFrame(data)
        #NOTE: TODO: If you want to condition the  chi-squared such that you
        # look at the differences in chi-squard when the data involves Gly and Pro,
        # and when the data does not involve Gly and Pro
        # can you see a clear contribution to issues from Gly and Pro in certain force fields?



        #data1 = data1.iloc[139:] # NOTE: uncomment to test using only CS and J data
        f_exp, fX, fX_prior, sigma = data1["exp obs"].to_numpy(), data1["reweighted obs"].to_numpy(), data1["prior obs"].to_numpy(), data1["sigma"].to_numpy()
        chi2.append(compute_chi2(f_exp, fX, sigma))
        chi2_prior.append(compute_chi2(f_exp, fX_prior, sigma))
        #Nd = np.sum([noe[0].index.max()+1, cs[0].index.max()+1, J[0].index.max()+1])
        _Nd = {"NOE": len(noe_prior), "CS_Ha":len(HA_indices), "CS_H":len(H_indices), "J":len(J_prior)}
        Nd = np.sum(list(_Nd.values()))

        #biceps_chi2.append(compute_chi2(f_exp, fX, np.ones(len(f_exp))))

        #reduced_prior = np.sum([noe_chi2_prior/(noe[0].index.max()+1), cs_chi2_prior/(cs[0].index.max()+1), J_chi2_prior/(J[0].index.max()+1)])
        #reduced = np.sum([noe_chi2/(noe[0].index.max()+1), cs_chi2/(cs[0].index.max()+1), J_chi2/(J[0].index.max()+1)])

#        reduced_prior.append(np.sum([noe_chi2_prior/(noe[0].index.max()+1),
#                                cs_HA_chi2_prior/len(HA_indices),
#                                cs_H_chi2_prior/len(H_indices),
#                                J_chi2_prior/(J[0].index.max()+1)]))
#        reduced.append(np.sum([noe_chi2/(noe[0].index.max()+1),
#                          cs_HA_chi2/len(HA_indices),
#                          cs_H_chi2/len(H_indices),
#                          J_chi2/(J[0].index.max()+1)]))
#        reduced_biceps.append(np.sum([noe_chi2_biceps/(noe[0].index.max()+1),
#                          cs_HA_chi2_biceps/len(HA_indices),
#                          cs_H_chi2_biceps/len(H_indices),
#                          J_chi2_biceps/(J[0].index.max()+1)]))

        #reduced_prior.append(np.sum([noe_chi2_prior,
        #                        cs_HA_chi2_prior,
        #                        cs_H_chi2_prior,
        #                        J_chi2_prior])/Nd)
        #reduced.append(np.sum([noe_chi2,
        #                  cs_HA_chi2,
        #                  cs_H_chi2,
        #                  J_chi2])/Nd)
        #reduced_biceps.append(np.sum([noe_chi2_biceps,
        #                  cs_HA_chi2_biceps,
        #                  cs_H_chi2_biceps,
        #                  J_chi2_biceps])/Nd)


        #separated_chi2_prior = {"NOE": noe_chi2_prior,
        reduced_prior.append({"FF": FF, "nstates": nstates,
            "NOE": noe_chi2_prior,
            "J": J_chi2_prior,
            "CS_H": cs_H_chi2_prior,
            "CS_Ha": cs_HA_chi2_prior})
        reduced.append({"FF": FF, "nstates": nstates,
            "NOE": noe_chi2,
            "J": J_chi2,
            "CS_H": cs_H_chi2,
            "CS_Ha": cs_HA_chi2})
        reduced_biceps.append({"FF": FF, "nstates": nstates,
            "NOE": noe_chi2_biceps,
            "J": J_chi2_biceps,
            "CS_H": cs_H_chi2_biceps,
            "CS_Ha": cs_HA_chi2_biceps})

    _reduced_prior = pd.DataFrame(reduced_prior)
    _reduced = pd.DataFrame(reduced)
    _reduced_biceps = pd.DataFrame(reduced_biceps)

    reduced_prior = _reduced_prior.drop(columns=["FF", "nstates"], axis=1)
    reduced = _reduced.drop(columns=["FF", "nstates"], axis=1)
    reduced_biceps = _reduced_biceps.drop(columns=["FF", "nstates"], axis=1)


    results["J_forward_model_error"] = sigma_J_column
    results["cs_forward_model_error"] =  sigma_cs_column
    results["noe_forward_model_error"] = sigma_noe_column
    results["chi-squared"] = chi2
    results["reduced chi-squared"] = reduced.sum(axis=1).to_numpy()/Nd #chi2/Nd
    #plot_heatmap_of_chi2(results, figname=f"figures/chi-squared.pdf")
    results["chi-squared_prior"] = chi2_prior
    results["reduced chi-squared_prior"] = reduced_prior.sum(axis=1).to_numpy()/Nd #chi2_prior/Nd

    results["biceps-chi-squared"] = reduced_biceps.sum(axis=1).to_numpy()/Nd #biceps_chi2/Nd



    dir = f"figures/separate_chi2-prior"
    biceps.toolbox.mkdir(dir)

    plot_heatmaps_of_chi2_prior(results, reduced=True, cols=["reduced chi-squared_prior"], Nds=[Nd], figname=f"{dir}/chi_squared_all{ext}.pdf",
            save_tables=False, title_fontsize=20, title_position=(0.5,0.98), annot_fontsize=14, all_data=1,
            cbar_loc="right", grid=(1, 1), positions=[(0,0)], figsize=(8,6))

    biceps.toolbox.mkdir(dir)
    #_reduced_prior = _reduced_prior.groupby(["FF", "nstates"]).agg("mean")
    # make figure
    plt.close()
    plt.clf()
    cols = []
    Nds = []
    temp_results = results.copy(deep=True)
    for col in _Nd.keys():
        temp_results["reduced chi-squared_prior"] = _reduced_prior[col].to_numpy()/_Nd[col]
        temp_results[f"reduced chi-squared_prior_{col}"] = _reduced_prior[col].to_numpy()/_Nd[col]
        cols.append(f"reduced chi-squared_prior_{col}")
        Nds.append(_Nd[col])
        filename = f"{dir}/reduced_chi-squared_{col}_prior{ext}.pdf"

    plot_heatmaps_of_chi2_prior(temp_results, reduced=True, cols=cols, Nds=Nds, figname=f"{dir}/chi_squared_horizontial{ext}.pdf",
            save_tables=False, title_fontsize=20, title_position=(0.5,0.98), annot_fontsize=14,  all_data=0,
            cbar_loc="top", grid=(1, 4), positions=[(0,0), (0,1), (0,2), (0,3)], figsize=(20,7))

    plot_heatmaps_of_columns(results, stat_models, columns=["D_KL"]*len(stat_models), figname=f"figures/BICePs_D_KL_all{ext}.pdf",
                          title=r"$D_{KL}$ of BICePs reweighted populations ($\lambda=0 \rightarrow \lambda=1$)",
                          title_fontsize=20, title_position=(0.5, 0.96), annot_fontsize=14,
                          #title_fontsize=20, title_position=None, annot_fontsize=14,
                          grid=(1, 4), positions=[(0,0), (0,1), (0,2), (0,3)], figsize=(20,7), cbar_loc="top")





    stat_models = _stat_models.copy()

    plot_ml_sigmas = 1
    if plot_ml_sigmas:
        # Maximum likelihood sigmas:{{{
        dir = f"figures/max_likelihood_sigmas"
        nstates = 500
        #nstates = 10
        biceps.toolbox.mkdir(dir)
        for i,stat_model in enumerate(stat_models): #list(set(results["stat_model"].to_numpy())):
            if stat_model != "Gaussian": continue
            _results = results.iloc[np.where(results[["stat_model"]].to_numpy()==new_sm[i])[0]]

            #for nstates in list(set(_results["nstates"].to_numpy())):
            r = _results.iloc[np.where(_results["nstates"].to_numpy()==nstates)[0]]
            res = r.copy()
            print(stat_model)
            print(list(set(res["nsteps"].to_numpy())))
            nsteps = int(list(set(res["nsteps"].to_numpy()))[0])
            nreplica = int(list(set(res["nreplica"].to_numpy()))[0])

            x,y = [],[]
            for ff in FF_list:
                try:
                    results_dir = f"{sys_name}/{ff}/nclusters_{nstates}/{stat_model}_*_sigma/{nsteps}_steps_{nreplica}_replicas_2_lam__swap_every_0_change_Nr_every_0"
                    files = biceps.toolbox.get_files(f"{results_dir}*/mlp.pkl")
                    for file in files:
                        if len(file.split("trial_")) >= 2:
                            trial = int(file.split("trial_")[-1].split("/mlp.pkl")[0])
                        else:
                            trial = 0

                        mlp = pd.read_pickle(file)

                        #print(mlp)
                        #mlp = mlp.iloc[[1]]
                        mlp = mlp.iloc[[1]]
                        columns = [col for col in mlp.columns.to_list() if "sigma" in col]
                        mlp = mlp[columns]
                        mlp.columns = [col.split("sigma_")[-1] for col in columns]

                        # make figure
                        fig = plt.figure(figsize=(12, 4))
                        gs = gridspec.GridSpec(1, 1)
                        ax = plt.subplot(gs[(0,0)])
                        colors = ["c", "y", "r", "g"]
                        ticklabels = []
                        max_value = 0
                        for c,col in enumerate(mlp.columns.to_list()):
                            if "noe" in col:
                                ax.bar(col, mlp[col], color=colors[0])
                                #ticklabels.append(re.findall(r'\d+', col)[0].split("noe")[-1])
                            elif "J" in col:
                                ax.bar(col, mlp[col], color=colors[1])
                                #ticklabels.append(re.findall(r'\d+', col)[0].split("J")[-1])
                            elif "H" in col:
                                ax.bar(col, mlp[col], color=colors[2])
                                #ticklabels.append(re.findall(r'\d+', col)[0].split("H")[-1])
                            if max_value < mlp[col].max():
                                max_value = mlp[col].max()
                            #print(col)
                            ticklabels.append(re.findall(r'\d+', col)[0])

                        if len(mlp.columns.to_list()) > 10:
                            visible_tick_labels = [ticklabels[i].split("noe")[-1].split("J")[-1].split("H")[-1] if i % 4 == 0 else "" for i in range(len(ticklabels))]
                            # Set the tick locations and labels
                            ax.set_xticks(ax.get_xticks())
                            ax.set_xticklabels(visible_tick_labels, rotation=10, fontsize=10)
                        else:
                            visible_tick_labels = [ticklabels[i].split("noe")[-1].split("J")[-1].split("H")[-1] for i in range(len(ticklabels))]
                            ax.set_xticks(ax.get_xticks())
                            ax.set_xticklabels(visible_tick_labels, rotation=10, fontsize=10)


                        xlim = len(mlp.columns.to_list())+1
                        ylim = max_value + 0.5
                        ax.set_xlim(-1, xlim)
                        ax.set_ylim(0, ylim)
                        ax.text(xlim*0.65, ylim*0.925, 'NOE distances', fontsize=16, color=colors[0])
                        ax.text(xlim*0.35, ylim*0.925, 'J-coupling', fontsize=16, color=colors[1])
                        ax.text(xlim*0.05, ylim*0.925, 'Chemical shift', fontsize=16, color=colors[2])
                        ax.set_ylabel(r"$\sigma$", fontsize=16)
                        ax.set_xlabel(r"Data restraint index", fontsize=16)

                        fig = ax.get_figure()
                        fig.tight_layout()
                        fig.savefig(f"{dir}/max_likelihood_sigmas_{ff}_{nstates}_{stat_model}_trial_{trial}.pdf")
                except(Exception) as e:
                    print(e)

    # }}}


    plot_biceps_chi2_vs_beauchamp = 1
    if plot_biceps_chi2_vs_beauchamp:
    # BICePs chi2 Vs beauchamp{{{
        dir = f"figures/chi2_versus_D_KL"
        biceps.toolbox.mkdir(dir)

        comparable_FFs = ['AMBER99SBnmr1-ildn', 'AMBER99SB-ildn', 'OPLS-aa', 'AMBER99', 'AMBER99SB', 'CHARMM27']
        #comparable_FFs = [ff.replace('AMBER','A').replace('CHARM','C') for ff in comparable_FFs]
        y = np.array([701.,909.,1216.,1745.,830.,952.], dtype=float)/524.          # interpolated colorbar
        print(y)
        print(list(set(results["FF"].to_numpy())))

        correlation_results = []

        nstates = 500 # IMPORTANT:
        for i,stat_model in enumerate(stat_models):

            #results.iloc[np.where((results[["stat_model"]].to_numpy()==new_sm[i]) & (results[["nstates"]].to_numpy()==nstates) )[0]]
            _results = results.iloc[np.where(results[["stat_model"]].to_numpy()==new_sm[i])[0]]
            #for nstates in list(set(_results["nstates"].to_numpy())):
            r = _results
            res = r.copy()
            nsteps = int(list(set(res["nsteps"].to_numpy()))[0])
            nreplica = int(list(set(res["nreplica"].to_numpy()))[0])
            if stat_model =="Gaussian": continue

            x = []
            for ff in comparable_FFs:
                results_dir = f"{analysis_dir}/{ff}/nclusters_{nstates}/{stat_model}_single_sigma/{nsteps}_*"
                print(results_dir)
                files = biceps.toolbox.get_files(f"{results_dir}/mlp.pkl")
                print(files)
                # use the first trial

                ff = ff.replace('AMBER','A').replace('CHARM','C')
                chi2 = res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["biceps-chi-squared"].to_numpy().mean()
                mlp = pd.concat([pd.read_pickle(file).iloc[[1]] for file in files], axis=1).mean()

                sigma_H = mlp["sigma_H0"].to_numpy().mean()
                sigma_noe = mlp["sigma_noe0"].to_numpy().mean()
                sigma_J = mlp["sigma_J0"].to_numpy().mean()
                new_chi2 = 0.5*chi2/(sigma_H**2+sigma_noe**2+sigma_J**2)
                value = new_chi2
                print(f"new_chi2 = {value}")
                x.append(value)
                #y.append(res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["reduced chi-squared_prior"].to_numpy().mean())
                #y.append()
            x = np.array(x)
            y = np.array(y)
            #y -= y.min()
            R2 = np.corrcoef(x, y)[0,-1]
            R2 = R2**2
            print(R2)
            correlation_results.append({"nstates": nstates, "stat_model":new_sm[i],
                                        "R2":R2, "BICePs chi-squared":x, "Beauchamp chi-squared":y,
                })

            simple_plot(x,y,
                        xlabel=r'BICePs $\chi^{2}/N$',
                        ylabel=r'Beauchamp 2012 $\chi^{2}$',
                        name=f"{dir}/Beauchamp_correlation_{nstates}.pdf",
                        size=111,Type='scatter',
                        color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
                        xLine=None,yLine=None,
                        #annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.min(),text_y=y.max(),
                        annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.max()-1,text_y=y.min(),
                        annotate_x=x.min(),annotate_y=y.max(),
                        arrow='->')
        correlation_results = pd.DataFrame(correlation_results)
        correlation_results.to_pickle(f"{dir}/Beauchamp_correlation_with_BICePs_chi-squared.pkl")




    #stat_models = new_sm.copy()
    plot_biceps_vs_bioEn = 0
    if plot_biceps_vs_bioEn:
    # BICePs Vs 0.5 chi2/simga^2 + D_{KL}{{{
    # NOTE: Look at the plot of BICePs Vs 0.5 chi2/simga^2 + D_{KL}
        dir = f"figures/chi2_versus_D_KL"
        biceps.toolbox.mkdir(dir)

        #_models = ["Bayesian", "GaussianSP"]
        #for stat_model in list(set(results["stat_model"].to_numpy())):
        nstates = 500 # IMPORTANT:
        #for stat_model in stat_models: #_models:
    #    for stat_model in [students_model]:

        for i,stat_model in enumerate(stat_models):

            #results.iloc[np.where((results[["stat_model"]].to_numpy()==new_sm[i]) & (results[["nstates"]].to_numpy()==nstates) )[0]]
            _results = results.iloc[np.where(results[["stat_model"]].to_numpy()==new_sm[i])[0]]
            #for nstates in list(set(_results["nstates"].to_numpy())):
            r = _results
            res = r.copy()
            nsteps = int(list(set(res["nsteps"].to_numpy()))[0])
            nreplica = int(list(set(res["nreplica"].to_numpy()))[0])
            if stat_model =="Gaussian": continue

            x,y = [],[]
            for ff in FF_list:
                results_dir = f"{analysis_dir}/{ff}/nclusters_{nstates}/{stat_model}_single_sigma/{nsteps}_*"
                print(results_dir)
                files = biceps.toolbox.get_files(f"{results_dir}/mlp.pkl")
                print(files)
                #if stat_model == students_model:
                #    results_dir = f"{sys_name}/{ff}/nclusters_{nstates}/Students_new_single_sigma/{nsteps}_steps_{nreplica}_*"
                #    files = biceps.toolbox.get_files(f"{results_dir}/mlp.pkl")
                #if stat_model == students_model:
                #    results_dir = f"{sys_name}/{ff}/nclusters_{nstates}/GaussianGB_single_sigma/{nsteps}_steps_{nreplica}_*"
                #    files = biceps.toolbox.get_files(f"{results_dir}/mlp.pkl")


    #            # FIXME: this is user controlled & might be different in the future
    #            #print(res.columns.to_list())
    #            #exit()
    #            #x.append(res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["D_KL"].to_numpy().mean())
    #            D_KL = res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["D_KL"].to_numpy().mean()
    #            #chi2 = res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["reduced chi-squared_prior"].to_numpy().mean()
    #            chi2 = res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["biceps-chi-squared"].to_numpy().mean()

                # use the first trial
                D_KL = res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["D_KL"].to_numpy().mean()
                chi2 = res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["biceps-chi-squared"].to_numpy().mean()

                mlp = pd.concat([pd.read_pickle(file).iloc[[1]] for file in files], axis=1).mean()
                #print(mlp)

                sigma_H = mlp["sigma_H0"].to_numpy().mean()
                sigma_noe = mlp["sigma_noe0"].to_numpy().mean()
                sigma_J = mlp["sigma_J0"].to_numpy().mean()

                #res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["biceps-chi-squared"] = 0.5*chi2/(sigma_H**2+sigma_noe**2+sigma_J**2)
                new_chi2 = 0.5*chi2/(sigma_H**2+sigma_noe**2+sigma_J**2)
                value = new_chi2 + D_KL
                print(f"new_chi2 + D_KL = {value}")
                x.append(value)
                #y.append(res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["reduced chi-squared_prior"].to_numpy().mean())
                y.append(res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["BICePs Score lam=1"].to_numpy().mean())
            x = np.array(x)
            y = np.array(y)
            #y -= y.min()
            R2 = np.corrcoef(x, y)[0,-1]
            R2 = R2**2
            print(R2)

            try:
                simple_plot(x,y,
                            xlabel=r'$0.5\chi^{2} / \sigma^{2} + D_{KL}$',
                            ylabel=r'BICePs Score',
                            name=f"{dir}/BICePs_versus_chi-squared_and_D_KL_{nstates}_{stat_model}.pdf",
                            size=111,Type='scatter',
                            color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
                            xLine=None,yLine=None,
                            annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.max()-1,text_y=y.min(),
                            annotate_x=x.min(),annotate_y=y.max(),
                            arrow='->')
            except(Exception) as e:
                print(e)
        #        exit()
        #exit()

#    plot_heatmaps_of_columns(results, stat_models, columns=["biceps-chi-squared"]*len(stat_models), figname=f"figures/BICePs_chi_squared_all{ext}.pdf",
#                          title=r"BICePs \chi^{2} ($\lambda=0 \rightarrow \lambda=1$)",
#                          title_fontsize=20, title_position=(0.5, 0.96), annot_fontsize=14,
#                          #title_fontsize=20, title_position=None, annot_fontsize=14,
#                          grid=(1, 4), positions=[(0,0), (0,1), (0,2), (0,3)], figsize=(20,7), cbar_loc="top")




    # }}}


# Plot heatmap of chi2 and compare correlations: {{{


# NOTE: appending reduced chi2 to rankings table
    #table_headings = ["reduced chi-squared_prior", "reduced chi-squared",
    states = 500
    results["reduced chi-squared_prior"] = reduced_prior.sum(axis=1).to_numpy()/Nd #chi2_prior/Nd
    b = results.iloc[np.where(results["nstates"] == states)[0]].groupby("FF").agg("mean").sort_values("reduced chi-squared_prior")
    rankings = b.index.to_list()
    rankings = [ff.replace('AMBER','A').replace('CHARM','C') for ff in rankings]
    rankings_table.append(rankings)
    rankings_table_columns.append(r"$\chi^{2}$ (prior)")

    headers = ["reduced chi-squared", "BICePs Score lam=1"]
    ascending_order = [1, 1]
    for o,header in enumerate(headers):
        for i,stat_model in enumerate(stat_models):
            header = headers[o]
            results_df = results.where((results["stat_model"]==new_sm[i]) & (results["nstates"] == states))
            results_df = results_df.groupby(["FF", "uncertainties"]).agg("mean")
            #biceps_score_cols = [x for i,x in enumerate(results_df.columns.to_list()) if "BICePs Score" in str(x)]
            data = results_df[header].reset_index().drop("uncertainties", axis=1).groupby(["FF"]).agg("mean")
            data = data.sort_values(header, ascending=1) # ascending_order[o]
            rankings = data.index.to_list()
            rankings = [ff.replace('AMBER','A').replace('CHARMM','C') for ff in rankings]
            rankings_table.append(rankings)
            header = header.replace('lam=1','').replace('reduced chi-squared',r'$\chi^{2}$').replace("BICePs Score", "BS")
            rankings_table_columns.append(f"{header} ({new_sm[i]})")
    df = pd.DataFrame(np.array(rankings_table).T, columns=rankings_table_columns)
    print(df)
    df1 = df[[s for idx,s in enumerate(df.columns.to_list()) if "chi" in s]]
    df2 = df[[s for idx,s in enumerate(df.columns.to_list()) if "BS" in s]]
    df1.to_latex("figures/FF_chi2_rankings.tex", escape=False)
    df1.to_html("figures/FF_chi2_rankings.html")
    df2.to_latex("figures/FF_BS_rankings.tex", escape=False)
    df2.to_html("figures/FF_BS_rankings.html")


#    plot_heatmap_of_chi_squared(results, stat_models, figname=f"figures/chi2.pdf")
#    plot_heatmap_of_chi2(results, stat_models, reduced=False, Nd=Nd, figname=f"figures/chi2.pdf")
#    plot_heatmap_of_chi2(results, stat_models, reduced=True, Nd=Nd, figname=f"figures/reduced_chi2.pdf")
#
#    plot_heatmap_of_chi2_prior(results, reduced=False, Nd=Nd, figname=f"figures/chi2_prior.pdf")
#    plot_heatmap_of_chi2_prior(results, reduced=True, Nd=Nd, figname=f"figures/reduced_chi2_prior.pdf")
    #exit()

# NOTE: Plotting correlations between our chi2 predictions and Beauchamp 2012
    dir = f"figures/Beauchamp_2012_correlations"
    biceps.toolbox.mkdir(dir)

    comparable_FFs = ['AMBER99SBnmr1-ildn', 'AMBER99SB-ildn', 'OPLS-aa', 'AMBER99', 'AMBER99SB', 'CHARMM27']
    comparable_FFs = [ff.replace('AMBER','A').replace('CHARM','C') for ff in comparable_FFs]
#    y = np.array([440, 430, 675, 690, 475, 510])/395      # eyeballed (ubiquitin)
    #y = np.array([730, 1000, 1210, 1675, 920, 1030])/524  # eyeballed (all data)
    y = np.array([701.,909.,1216.,1745.,830.,952.], dtype=float)/524.          # interpolated colorbar
    print(y)

    print(list(set(results["FF"].to_numpy())))

    results["reduced chi-squared_prior"] = reduced_prior.sum(axis=1).to_numpy()/Nd #chi2_prior/Nd

    #for nstates in list(set(results["nstates"].to_numpy())):
    for nstates in [500]: #list(set(results["nstates"].to_numpy())):
        _results = results.iloc[np.where(results["nstates"].to_numpy()==nstates)[0]]
        res = _results.copy()
        x = []
        for ff in comparable_FFs:
            x.append(res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["reduced chi-squared_prior"].to_numpy().mean())
        x = np.array(x)
        print(x)
        #["chi-squared"]
        R2 = np.corrcoef(x, y)[0,-1]
        R2 = R2**2
        print(R2)

        simple_plot(x,y,
                    xlabel=r'$\chi^{2}$',
                    ylabel=r'approx Beauchamp 2012 $\chi^{2}$',
                    name=f"{dir}/Beauchamp_correlation_{nstates}.pdf",
                    size=111,Type='scatter',
                    color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
                    xLine=None,yLine=None,
                    #annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.min(),text_y=y.max(),
                    annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.max()-1,text_y=y.min(),
                    annotate_x=x.min(),annotate_y=y.max(),
                    arrow='->')

# NOTE: now compare our BICePs scores to Beauchamp 2012 chi2
    correlation_results = []

    for i,stat_model in enumerate(stat_models): #list(set(results["stat_model"].to_numpy())):
        _results = results.iloc[np.where(results[["stat_model"]].to_numpy()==new_sm[i])[0]]

        #for nstates in list(set(_results["nstates"].to_numpy())):
        for nstates in [500]:
            r = _results.iloc[np.where(_results["nstates"].to_numpy()==nstates)[0]]
            res = r.copy()
            x = []
            for ff in comparable_FFs:
                x.append(res.iloc[np.where(res["FF"].to_numpy()==ff)[0]]["BICePs Score lam=1"].to_numpy().mean())
            x = np.array(x)
            print(x)
            #["chi-squared"]
            R2 = np.corrcoef(x, y)[0,-1]
            R2 = R2**2
            print(R2)
            correlation_results.append({"nstates": nstates, "stat_model":new_sm[i],
                                        "R2":R2, "BICePs score":x, "Beauchamp chi-squared":y,
                })



            try:
                simple_plot(x,y,
                            xlabel=r'BICePs Score',
                            ylabel=r'approx. Beauchamp 2012 $\chi^{2}$',
                            name=f"{dir}/Beauchamp_correlation_with_BICePs_scores_{nstates}_{stat_model}.pdf",
                            size=111,Type='scatter',
                            color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
                            xLine=None,yLine=None,
                            #annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.min(),text_y=y.max(),
                            annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.max()-1,text_y=y.min(),
                            annotate_x=x.min(),annotate_y=y.max(),
                            arrow='->')
            except(Exception) as e:
                print(e)
    correlation_results = pd.DataFrame(correlation_results)
    correlation_results.to_pickle(f"{dir}/Beauchamp_correlation_with_BICePs_scores.pkl")




# NOTE: Plotting correlations between our BICePs scores and reduced chi2
    dir = f"figures/BICePs_chi2_correlations"
    biceps.toolbox.mkdir(dir)
    for i,stat_model in enumerate(stat_models):
        print(new_sm[i])
        _results = results.iloc[np.where(results["stat_model"] == new_sm[i])[0]]
        res = _results.copy()
        x = res["reduced chi-squared"].to_numpy()
        y = res["BICePs Score lam=1"].to_numpy()

        R2 = np.corrcoef(x, y)[0,-1]
        R2 = R2**2
        print(R2)

        simple_plot(x,y,
                    xlabel=r'$\chi^{2}$',
                    ylabel=r'BICePs Score',
                    name=f"{dir}/BICePs_chi2_correlation_{stat_model}.pdf",
                    size=111,Type='scatter',
                    color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
                    xLine=None,yLine=None,
                    #annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.min(),text_y=y.max(),
                    annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.max()-1,text_y=y.min(),
                    annotate_x=x.min(),annotate_y=y.max(),
                    arrow='->')

# NOTE: Plotting correlations between our BICePs scores and reduced chi2 from prior
    dir = f"figures/BICePs_chi2_correlations"
    biceps.toolbox.mkdir(dir)
    for i,stat_model in enumerate(stat_models):
        print(new_sm[i])
        _results = results.iloc[np.where(results["stat_model"] == new_sm[i])[0]]
        res = _results.copy()
        x = res["reduced chi-squared_prior"].to_numpy()
        y = res["BICePs Score lam=1"].to_numpy()

        R2 = np.corrcoef(x, y)[0,-1]
        R2 = R2**2
        print(R2)

        simple_plot(x,y,
                    xlabel=r'$\chi^{2}$ (of prior)',
                    ylabel=r'BICePs Score',
                    name=f"{dir}/BICePs_chi2_from_prior_correlation_{stat_model}.pdf",
                    size=111,Type='scatter',
                    color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
                    xLine=None,yLine=None,
                    #annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.min(),text_y=y.max(),
                    annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.max()-1,text_y=y.min(),
                    annotate_x=x.min(),annotate_y=y.max(),
                    arrow='->')



# NOTE: Plotting correlations between our statistical models (likelihood functions)
#    dir = f"figures/stat_model_correlations"
#    biceps.toolbox.mkdir(dir)
#
#    _stat_models = ["Bayesian","GaussianSP"] # ,"Outliers", "Gaussian""SinglePrior","Outliers","Gaussian"]
#    combos = itertools.combinations(_stat_models, 2)
#    for i,pair in enumerate(combos):
#        stat_model1, stat_model2 = pair
#        _results1 = results.iloc[np.where(results["stat_model"] == stat_model1)[0]]
#        _results2 = results.iloc[np.where(results["stat_model"] == stat_model2)[0]]
#        #res = _results.copy()
#        x = _results1["BICePs Score lam=1"].to_numpy()
#        y = _results2["BICePs Score lam=1"].to_numpy()
#
#        R2 = np.corrcoef(x, y)[0,-1]
#        R2 = R2**2
#        print(R2)
#
#        simple_plot(x,y,
#                    xlabel=f'{stat_model1}',
#                    ylabel=f'{stat_model2}',
#                    name=f"{dir}/BICePs_score_correlation_{stat_model1}_and_{stat_model2}.pdf",
#                    size=111,Type='scatter',
#                    color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
#                    xLine=None,yLine=None,
#                    #annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.min(),text_y=y.max(),
#                    annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.max()-1,text_y=y.min(),
#                    annotate_x=x.min(),annotate_y=y.max(),
#                    arrow='->')

# }}}


    exit()

# NOTE: HERE{{{
for FF in FF_list:
    for clusters in [5,10,50,75,100,500]:
        #files = biceps.toolbox.get_files(f"{sys_name}/{FF}/*/*/*/*__reweighted_populations.csv")
        files = biceps.toolbox.get_files(f"{analysis_dir}/{FF}/nclusters_{clusters}/*/*/*__reweighted_populations.csv")
        #print(files)
        #exit()
        files = biceps.toolbox.get_files(f"{analysis_dir}/{FF}/nclusters_{clusters}/*/*/*__reweighted_populations.csv")

        # Get the files for prior pops and assignments (microstate --> macrostate)
        # ..for each FF/microstate clustering
        assignment_files = biceps.toolbox.get_files(f"{data_dir}/{sys_name}/{FF}/nclusters_{clusters}/inverse_distances_k*_msm_assignments.csv")
        prior_pops_files = biceps.toolbox.get_files(f"{data_dir}/{sys_name}/{FF}/nclusters_{clusters}/inverse_distances_k*_msm_pops.csv")
        assignment_df = pd.read_csv(assignment_files[0], index_col=0)
        prior_df =  pd.read_csv(prior_pops_files[0], index_col=0)
        # Get a dataframe of prior macrostate populations (from MSM)
        microstates = assignment_df.index.to_numpy()
        macrostates = assignment_df[assignment_df.columns[0]].to_numpy()
        prior_pops = prior_df[prior_df.columns[0]].to_numpy()
        ntop = len(prior_pops)
        topN_pops = prior_pops[np.argsort(prior_pops)[-ntop:]]
        topN_labels = [np.where(topN_pops[i]==prior_pops)[0][0] for i in range(len(topN_pops))]
        macro_asignment = []
        for i,state in enumerate(topN_labels):
            index = np.where(microstates == state)[0]
            macrostate = int(macrostates[int(index)])
            macro_asignment.append({"macrostate": macrostate, "microstate":state, "population":topN_pops[i]})
        df = pd.DataFrame(macro_asignment)
        grouped = df.groupby(["macrostate"])
        df = df.set_index("macrostate")
        macrostate_populations = grouped.sum().drop("microstate", axis=1)
        print(macrostate_populations)


        #exit()

        #energies = np.loadtxt(f'{sys_name}/{sys_name}_energy_microstate.txt')/4184.*(6.022e23)  # Joules to kcal/mol
        #energies = energies/0.5959   # convert to reduced free energies F = f/kT
        #energies -= energies.min()  # set ground state to zero, just in case

        ## CLN001 MSM
        #State1 = 9.2*0.01   #random
        #State2 = 5.1*0.01   #helix
        #State3 = 9.3*0.01   #misfolded
        #State4 = 76.5*0.01  #folded

        results = []
        if sys_name == "CLN001":
            results.append({
                "stat_model": "Experiment", "nreplicas": 0, "nlambdas": 0, "nsteps": 0, "data_uncertainty": "none",
                #0: 0.83, 1:0, 2:0
                0: experimental_folded.nominal_value, 1:0, 2:0
                })
            #results.append({
            #    "stat_model": "MSM", "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
            #    4: 0.765, 3:0.093, 2:0.051, 1:0.092
            #    })
            results.append({
                "stat_model": "MSM", "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
                **macrostate_populations.to_dict()[macrostate_populations.columns[0]]
                })
            print(results)
            #exit()


        if sys_name == "CLN025":
            results.append({
                "stat_model": "Experiment", "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
                3: 0.975, 2:0, 1:0, 0:0
                })
            results.append({
                "stat_model": "MSM", "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
                4: 0.939, 3:0.032, 2:0.029, 1:0.001
                })

        for file in files:
            #if file.split("/")[0] == sys_name:
            print(file)
            stat_model = file.split("/")[-1].split("_")[0]

# NOTE: TODO: HELP: FIXME: IMPORTANT: ##########################################
            # NOTE: Removing SinglePrior model from the plots
            if stat_model == "SinglePrior": continue
# NOTE: TODO: HELP: FIXME: IMPORTANT: ##########################################


            #print(stat_model)
            data_uncertainty = file.split("/")[-1].split("_sigma")[0].split("_")[-1]
            nreplicas = int(file.split("/")[-2].split("_replicas")[0].split("_")[-1])
            #print(nreplicas)
            #if (stat_model == "Bayesian") and (nreplicas > 1):
            #    continue
            #if (data_uncertainty == "multiple"):
            #    continue
            nsteps = int(file.split("/")[-2].split("_steps")[0].split("_")[-1])
            #if nsteps > 100000:
            #    continue
            nlambdas = int(file.split("/")[-2].split("_lam")[0].split("_")[-1])
            df = pd.read_csv(file)
            grouped = df.groupby(["macrostate"])
            df = df.set_index("macrostate")
            pop_dict = grouped.sum()["population"].to_dict()
            results.append({
                "FF": FF,
                "stat_model": stat_model,
                "nreplicas": nreplicas,
                "nlambdas": nlambdas,
                "nsteps": nsteps,
                "data_uncertainty": data_uncertainty,
                **pop_dict
                })
            #print(results)
            #exit()

        df = pd.DataFrame(results)
        print(df)
        #exit()

        #grouped = df.groupby([4,3,2,1])
        grouped = df.groupby(["stat_model","nreplicas","data_uncertainty", "nsteps"]).agg("mean")
        print(grouped)

        try:
            #ax = grouped.bar([4,3,2,1], alpha=0.5, edgecolor='black', linewidth=1.2, color="b", figsize=(14, 6))

            if pub_images:
                ax = df.plot.bar(x="stat_model", y=[0,1,2],  figsize=(10, 4), rot=0)
            else:
                ax = df.plot.bar(x="stat_model", y=[0,1,2],  figsize=(10, 4), rot=65)
            ax.axhline(y=df.iloc[np.where(df["stat_model"] == "Experiment")[0]].to_dict()[0][0], color='k', linestyle="--", label="_nolegend_")
            ax.axhspan(experimental_lower, experimental_upper, facecolor='orange', alpha=0.4)
            xticklabels = []
            model_labels = df["stat_model"].to_numpy()
            replica_labels = df["nreplicas"].to_numpy()
            step_labels = df["nsteps"].to_numpy()
            for i in range(len(model_labels)):
                if replica_labels[i] == 0:
                    xticklabels.append(model_labels[i])
                if replica_labels[i] != 0:
                    if pub_images:
                        xticklabels.append(str(model_labels[i]))
                    else:
                        xticklabels.append(str(str(model_labels[i])+"\n"+str(step_labels[i])+" steps & "+str(replica_labels[i])+" replica"))
            ax.set_xticklabels([f"{x}" for x in xticklabels])
            ax.set_xlabel(r"", fontsize=16)
            ax.set_ylabel(r"", fontsize=16)
            ax.set_ylim(0,1)

            # Setting the ticks and tick marks
            ticks = [ax.xaxis.get_minor_ticks(),
                     ax.xaxis.get_major_ticks()]
            marks = [ax.get_xticklabels(),
                    ax.get_yticklabels()]
            for k in range(0,len(ticks)):
                for tick in ticks[k]:
                    tick.label.set_fontsize(16)
            for k in range(0,len(marks)):
                for mark in marks[k]:
                    mark.set_size(fontsize=16)
                    if k == 0:
                        mark.set_rotation(s=0)



            #if sys_name == "CLN001": ax.legend(["Exp", "4: folded","3: misfolded", "2: helix", "1: unfolded"], loc="best")
            #if sys_name == "CLN025": ax.legend(["Exp", "4: folded","3: misfolded", "2: unfolded", "1: helix"], loc="best")
            if sys_name == "CLN001": ax.legend(["Exp", "0: folded","1: misfolded", "2: unfolded"], loc="best")#, fontsize=16)
            if sys_name == "CLN025": ax.legend(["Exp", "4: folded","3: misfolded", "2: unfolded", "1: helix"], loc="best")#, fontsize=16)
            fig = ax.get_figure()
            fig.tight_layout()
            biceps.toolbox.mkdir(f"{clusters}_microstates")
            fig.savefig(f"{clusters}_microstates/{sys_name}_{FF}_populations.pdf", dpi=dpi)
            #print(f"RMSE = {df['error'].mean()} ± {df['error'].std()}")
        except(Exception) as e:
            continue
# }}}



## PLot scores:{{{
#
#for FF in FF_list:
#    for clusters in [5,10,50,75,100,500]:
#
#        # get files for all force fields and plot biceps score comparison
#        files = biceps.toolbox.get_files(f"{sys_name}/{FF}/nclusters_{clusters}/*/*/results.pkl")
#        results = []
#        for file in files:
#            nsteps = int(file.split("_steps")[0].split("/")[-1])
#            if nsteps >0:
#                df = pd.read_pickle(file)
#                results.append(df)
#        results = pd.concat(results)
#        #results = results.iloc[np.where(results["nlambda"] == 2)[0]]
#        db = results.copy()
#        #FF	System	nsteps	nstates	nlambda	nreplica	lambda_swap_every	Nd	uncertainties	stat_model
#        df = db.groupby(["FF","stat_model", "nsteps", "lambda_swap_every"]).agg("mean")
#        db = df.copy()
#
#
##        try:
#        # PLOT
#        fig = plt.figure(figsize=(14, 10))
#        gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
#        ax = plt.subplot(gs[0,0])
#        colors = ["b", "g", "r", "k"]
#
#        print(results)
#
#        cols = [i for i,x in enumerate(results.columns.to_list()) if "BICePs Score" in x]
#        #print(cols)
#        cols = results.columns[cols].to_list()
#        scores_cols,std_cols = cols[::2][::-1],cols[1::2][::-1]
#
#        for k,col in enumerate(scores_cols):
#
#            db[col] /= db["nreplica"].to_numpy()
#            db[col].plot.bar(label=col, color=colors[k], yerr=db[std_cols[k]].to_numpy(), ax=ax)
#            #ax.errorbar(x=db.index.to_numpy(), y=db[col].to_numpy(),
#            #            yerr=db[std_cols[k]].to_numpy(), color=colors[k],
#            #            ls="None", fmt='-o', capsize=3)
#
#        biceps_scores = db[scores_cols[0]].to_numpy()
#        #print(biceps_scores)
#        for j in range(len(biceps_scores)):
#            ax.text( j-0.25, biceps_scores[j]-.35, f"{biceps_scores[j]:.2f}", color='g' , fontsize=12, rotation=80)
#
#
#        #ax.set_xticks(xticklabels)
#        #ax.set_xticklabels([f"{x}" for x in xticklabels])
#
#        ax.tick_params(axis='both', which='major', labelsize=14)
#        ax.set_ylabel("BICePs Score", fontsize=16)
#        #ax.set_xlabel(r"$N_{d}$", fontsize=16)
#        ax.xaxis.label.set_size(16)
#        ax.legend()
#        ax.set_ylim(-3.5,0)
#
#
#
#        ticks = [ax.xaxis.get_minor_ticks(),
#                 ax.xaxis.get_major_ticks(),]
#        marks = [ax.get_xticklabels(),
#                ax.get_yticklabels(),]
#        for k in range(0,len(ticks)):
#            for tick in ticks[k]:
#                tick.label.set_fontsize(16)
#        for k in range(0,len(marks)):
#            for mark in marks[k]:
#                mark.set_size(fontsize=16)
#                mark.set_rotation(s=80)
#        fig = ax.get_figure()
#        fig.tight_layout()
#        biceps.toolbox.mkdir(f"{clusters}_microstates")
#        fig.savefig(f"{clusters}_microstates/biceps_scores_{sys_name}_{sys_name}_{FF}_prelim_results.png", dpi=dpi)
#
#
#        #except(Exception) as e:
#        #    continue
#
## }}}
#


#    exit()

#    # KDE for maximum likelihood sigmas:{{{
#    dir = f"figures/max_likelihood_sigmas"
#    nstates = 500
#    nstates = 10
#    biceps.toolbox.mkdir(dir)
#    for stat_model in list(set(results["stat_model"].to_numpy())):
#        _results = results.iloc[np.where(results[["stat_model"]].to_numpy()==stat_model)[0]]
#
#        #for nstates in list(set(_results["nstates"].to_numpy())):
#        r = _results.iloc[np.where(_results["nstates"].to_numpy()==nstates)[0]]
#        res = r.copy()
#        print(stat_model)
#        print(list(set(res["nsteps"].to_numpy())))
#        nsteps = int(list(set(res["nsteps"].to_numpy()))[0])
#        nreplica = int(list(set(res["nreplica"].to_numpy()))[0])
#
#        x,y = [],[]
#        for ff in FF_list:
#            try:
#                results_dir = f"{sys_name}/{ff}/nclusters_{nstates}/{stat_model}_*_sigma/{nsteps}_steps_{nreplica}_replicas_2_lam__swap_every_0_change_Nr_every_0"
#                files = biceps.toolbox.get_files(f"{results_dir}*/mlp.pkl")
#                for file in files:
#                    if len(file.split("trial_")) >= 2:
#                        trial = int(file.split("trial_")[-1].split("/mlp.pkl")[0])
#                    else:
#                        trial = 0
#
#                    mlp = pd.read_pickle(file)
#
#                    #print(mlp)
#                    #mlp = mlp.iloc[[1]]
#                    mlp = mlp.iloc[[1]]
#                    columns = [col for col in mlp.columns.to_list() if "sigma" in col]
#                    mlp = mlp[columns]
#                    mlp.columns = [col.split("sigma_")[-1] for col in columns]
#
#                    # make figure
#                    fig = plt.figure(figsize=(12, 4))
#                    gs = gridspec.GridSpec(1, 1)
#                    ax = plt.subplot(gs[(0,0)])
#                    colors = ["c", "y", "r", "g"]
#                    ticklabels = []
#                    for c,col in enumerate(mlp.columns.to_list()):
#                        if "noe" in col:
#                            ax.bar(col, mlp[col], color=colors[0])
#                        elif "J" in col:
#                            ax.bar(col, mlp[col], color=colors[1])
#                        elif "H" in col:
#                            ax.bar(col, mlp[col], color=colors[2])
#                        #print(col)
#                        ticklabels.append(re.findall(r'\d+', col)[0])
#                    if len(mlp.columns.to_list()) > 10:
#                        for r,row in enumerate(ax.get_xticklabels()):
#                            #row.set_text(row.get_text().split("sigma_")[-1])#.split("")[-1])
#                            if r %2 == 0:
#                                row.set_text(ticklabels[r])
#                            else:
#                                row.set_text("")
#                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
#                    ax.set_xlim(-1, len(mlp.columns.to_list())+1)
#
#                    fig = ax.get_figure()
#                    fig.tight_layout()
#                    fig.savefig(f"{dir}/max_likelihood_sigmas_{ff}_{nstates}_{stat_model}_trial_{trial}.pdf")
#            except(Exception) as e:
#                print(e)
#
## }}}
#
    #exit()


#    nstates = 500
#    _reduced_prior = _reduced_prior.iloc[np.where(_reduced_prior["nstates"].to_numpy() == nstates)[0]]
#    _reduced_prior = _reduced_prior.groupby(["FF", "nstates"]).agg("mean")
#    print(_reduced_prior)
#
#    # make figure
#    plt.close()
#    plt.clf()
#    for col in _reduced_prior.columns.to_list():
#        print(col)
#        print(_reduced_prior[col])
#        #ax = _reduced_prior[col].plot.bar(figsize=(10, 4), rot=35)
#        ax = _reduced_prior[col].plot.scatter(figsize=(10, 4), rot=35)
#        #ax.errobar(_reduced_prior[col].index, _reduced_prior[col].to_numpy()
#
#        fig = ax.get_figure()
#        fig.tight_layout()
#        fig.savefig(f"{dir}/chi-squared_{col}_prior_{nstates}.pdf")
#        plt.close()
#        plt.clf()













