# Libraries:{{{
import string,re
import biceps
from biceps.toolbox import three2one
import itertools
import pandas as pd
import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None

import seaborn as sb
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import uncertainties as u ################ Error Prop. Library
#from tqdm import tqdm
import matplotlib

mpl_colors = matplotlib.colors.get_named_colors_mapping()
mpl_colors = list(mpl_colors.values())[::5]
extra_colors = mpl_colors.copy()
#mpl_colors = ["k","lime","b","brown","c","green",
mpl_colors = ["b","brown","c","green",
              "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
              #'#acc2d9', "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
        '#e78ea5', '#983fb2', '#b7e1a1', '#430541', '#507b9c', '#c9d179',
            '#2cfa1f', '#fd8d49', '#b75203', '#b1fc99']+extra_colors[::2]+ ["k","grey"]

# }}}

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
    results_df = results.where(results["stat_model"]=="Bayesian")
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
    if reduced: prefix = r"Reduced $\chi^{2}/N$"
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
                mark.set_rotation(s=25)

            #mark.set_rotation(s=65)

    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)






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
        data = data.reset_index().pivot("FF", "nstates", "D_KL")

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
                    mark.set_rotation(s=25)
        x,y = -0.1, 1.02
        ax4.text(x,y, string.ascii_lowercase[i], transform=ax4.transAxes,
                size=20, weight='bold')



    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)


def plot_heatmap_of_biceps_scores(df, stat_models, figname="BICePs_Scores.pdf", save_tables=True):
    #files = biceps.toolbox.get_files("CLN001/*/nclusters_*/*/*/results.pkl")
    #results = []
    #for file in files:
    #    results.append(pd.read_pickle(file))
    #results = pd.concat(results)
    results = df.copy()

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    #stat_models = ["Bayesian","Outliers","GaussianSP","Gaussian"]
    #stat_models = ["Bayesian","OutliersSP","Outliers","Gaussian"]
#    stat_models = ["Bayesian","SinglePrior","Outliers","Gaussian"]
#    stat_models = ["SinglePrior"]#,"Outliers","Gaussian"]
#    stat_models = ["SinglePrior","Outliers"]#,"Gaussian"]
    positions = [(0,0), (0,1), (1,0), (1,1)]
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

        print(results_df)
        data = results_df[score_cols[-1]].reset_index().drop("uncertainties", axis=1).groupby(["FF","nstates"]).agg("mean")
        print(data)
        data["BICePs Score lam=1"] = data["BICePs Score lam=1"].to_numpy()#/nreplica
        data = data.reset_index().pivot("FF", "nstates", "BICePs Score lam=1")
        #if save_tables: data.to_latex(f"{figname.split('.')[0]}_{stat_model}.tex")
        if save_tables: data.to_html(f"{figname.split('.')[0]}_{stat_model}.html")

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



    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)


def plot_heatmap_of_biceps_scores_std(df, stat_models, figname="BICePs_Scores.pdf", save_tables=True):
    #files = biceps.toolbox.get_files("CLN001/*/nclusters_*/*/*/results.pkl")
    #results = []
    #for file in files:
    #    results.append(pd.read_pickle(file))
    #results = pd.concat(results)
    results = df.copy()

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    #stat_models = ["Bayesian","Outliers","GaussianSP","Gaussian"]
    #stat_models = ["Bayesian","OutliersSP","Outliers","Gaussian"]
#    stat_models = ["Bayesian","SinglePrior","Outliers","Gaussian"]
#    stat_models = ["SinglePrior"]#,"Outliers","Gaussian"]
#    stat_models = ["SinglePrior","Outliers"]#,"Gaussian"]
    positions = [(0,0), (0,1), (1,0), (1,1)]
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
        data = data.reset_index().pivot("FF", "nstates", "BICePs Score lam=1")
        #if save_tables: data.to_latex(f"{figname.split('.')[0]}_{stat_model}.tex")
        if save_tables: data.to_html(f"{figname.split('.')[0]}_{stat_model}.html")

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



    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)






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



    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)



#:}}}

# correlation_plot_3_by_3:{{{
def correlation_plot_3_by_3(df, x_col, y_col, FF_order=None, xlabel=None, ylabel=None,
                labels=None, figname="3-by-3-correlations.pdf", figsize=(12, 12)):

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
    if FF_order:
        FFs = FF_order
    else:
        FFs = df["FF"].to_numpy()

    axs = []
    legend_labels = []
    legend_handles = []
    for i,FF in enumerate(FFs):

        k = i+1
        if i == 0:
            ax = fig.add_subplot(gs[positions[i]])
        else:
            ax = fig.add_subplot(gs[positions[i]], sharex=ax, sharey=ax)
        print(FF)
        row = df.iloc[np.where(FF == df["FF"].to_numpy())[0]]
        x,y = row[x_col].to_numpy()[0], row[y_col].to_numpy()[0]

        if i == 0:
            if labels != None:
                legend_labels = row[labels].to_numpy()[0]
                print(legend_labels)

        #print(type(x),type(y))
        #print(x,y)
        #x, y = np.array(x), np.array(y)

        _min = np.min([x.min(), y.min()])
        _max = np.max([x.max(), y.max()])
        ax.plot([_min, _max], [_min, _max], 'k-', label="_nolegend_")

        R2 = np.corrcoef(x, y)[0][1]**2
        R2 = r'R$^{2}$ = %0.2f'%R2
        MAE = metrics.mean_absolute_error(x, y)
        MAE = r'MAE = %0.2f'%MAE
        RMSE = np.sqrt(metrics.mean_squared_error(x, y))
        RMSE = r'RMSE = %0.2f'%RMSE


        FF = FF.replace('AMBER','A').replace('CHARM','C')
        #annotate_text = f"{R2}\n{MAE}\n{RMSE}"
        #annotate_text = f"{FF}\n{R2}\n{MAE}\n{RMSE}"
        annotate_text = f"{FF}\n{R2}\n{MAE}"
        #text_x,text_y = np.min(x)+0.1, np.max(y)-0.4

        ########################################################################
        #NOTE: FIXME: This is for Tim...
        annotate_text = f"{R2}\n{MAE}"
        ax.set_title(f'{FF}', loc='left', fontsize=16)
        ########################################################################
        if len(y) > len(mpl_colors):
            colors = "k"
        else:
            colors = []
            for j in range(len(y)):
                colors.append(mpl_colors[j])

        ax.scatter(x, y, c=colors, edgecolor="k", label=annotate_text)

        leg = ax.legend(loc="best", fontsize=14,
                        handlelength=0, handletextpad=0, fancybox=True)
        ax.add_artist(leg)
        for item in leg.legendHandles:
            item.set_visible(False)

        if i % 3 != 0:
            ax.tick_params(labelbottom=0, labeltop=0, labelleft=0,
                             labelright=0, bottom=1, top=1, left=1, right=1)
        else:
            ax.tick_params(labelbottom=0, labeltop=0, labelleft=1,
                             labelright=0, bottom=1, top=1, left=1, right=1)

        if i in list(range(len(positions)))[-3:]:
            ax.tick_params(labelbottom=1, labeltop=False, labelleft=0,
                             labelright=False, bottom=1, top=1, left=1, right=1)

        if i == list(range(len(positions)))[-3]:
            ax.tick_params(labelbottom=1, labeltop=0, labelleft=1,
                             labelright=0, bottom=1, top=1, left=1, right=1)

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
        axs.append(ax)

    #handles, labels = ax.get_legend_handles_labels()
    #print(ax.legend_elements())

    if labels != None:
        print("adding legend labels!")
        custom_handles = []
        custom_labels = []
        # Create the legend handles by adding empty lines
        for i,label in enumerate(legend_labels):
            handle = plt.Line2D([], [], color=mpl_colors[i], marker='', linestyle='-')
            custom_handles.append(handle)
            custom_labels.append(label)

        if len(y) < len(mpl_colors):
            legend2 = axs[5].legend(custom_handles, custom_labels, loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=16)
            axs[5].add_artist(legend2)
            #axs[5].legend(custom_handles, custom_labels, loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=16)


    if xlabel:
        fig.text(0.5, 0.04, xlabel, ha='center', fontsize=16)
    else:
        fig.text(0.5, 0.04, 'Experimental Observable', ha='center', fontsize=16)

    if ylabel:
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=16)
    else:
        fig.text(0.04, 0.5, 'Predicted Observable', va='center', rotation='vertical', fontsize=16)

    #fig.tight_layout(pad=5.4, w_pad=0.5, h_pad=1.0)
    fig.subplots_adjust(left=0.12, bottom=0.13, top=0.92, right=0.83, wspace=0.20, hspace=0.2)
    fig.savefig(f"{figname}", dpi=600)






# }}}


dpi=600
#sys_name = "CLN025"
sys_name = "CLN001"
pub_images = 1

FF_list = ["AMBER14SB","AMBER99SB-ildn","CHARMM27","AMBER99",
    "AMBER99SBnmr1-ildn","CHARMM36","AMBER99SB","CHARMM22star","OPLS-aa"]



# get files for all force fields and plot biceps score comparison
files = biceps.toolbox.get_files(f"{sys_name}/*/nclusters_*/*/*/results.pkl")
results = []
for file in files:

    if len(file.split("Bayes")) < 2:
        continue

    # NOTE: there is a lot of hardcoding in this script... here is just one example of it..
    if len(file.split("trial_")) >= 2:
        continue # FIXME: skip this shit for now
        trial = int(file.split("trial_")[-1].split("/results.pkl")[0])
    else:
        trial = 0
    df = pd.read_pickle(file)
    df["trial"] = trial
    results.append(df)
results = pd.concat(results)
#print(results)
#print(len(results))
#exit()



###############################################################################

stat_models = ["Bayesian"]
df = results.copy()
df["FF"] = [ff.replace('AMBER','A').replace('CHARM','C') for ff in df["FF"].to_numpy()]


biceps.toolbox.mkdir("figures")
#pairplot_of_biceps_scores(results, stat_models, figname="figures/cross_correlations.png")

experimental_folded = u.ufloat(0.61029, 0.03375)
val = experimental_folded.nominal_value
dev = experimental_folded.std_dev
experimental_lower, experimental_upper = val-dev, val+dev

rankings_table = []
rankings_table_columns = []

raw = []
#########################################
plot_chi_squared = 1 #True #False
if plot_chi_squared:
    # TODO: put the prior forward model uncertainties inside the results dataframe
    df = pd.read_pickle("../Systems_v04/prior_std_error_in_forward_model.pkl")

    sigma_H = 0.2351 # comes from shiftx2 performance
    sigma_HA = 0.1081 # comes from shiftx2 performance
    sigma_J = 0.34   # comes from uncertainty in karplus parameters
    sigma_NOE = 1.0  # set to 1 due to experimental upper and lower limit
    sigma_J_column, sigma_noe_column, sigma_cs_column = [], [], []
    chi2, chi2_prior = [], []
    biceps_chi2 = []
    reduced_prior, reduced, reduced_biceps = [],[],[]
    #print(results.columns.to_list())
    #exit()
    for row in results.iterrows():
        #print(row[1])
        #exit()
        FF = row[1][0] # ff
        nstates = row[1][3] # nstates
        system = row[1][1] # system
        stat_model = row[1][9] # stat_model?
        # search for row
        sel_row = df.iloc[np.where( (df["sys_name"] == system) & (df["forcefield"] == FF) & (df["nstates"] == nstates))[0]]
        #print(sel_row)
        sigma_noe = float(sel_row["std"].to_numpy()) # FIXME: if no longer a float
        #sigma_noe = sel_row["std"].to_numpy()[0]
        #sigma_noe = sigma_NOE # FIXME

        # append to lists
        sigma_J_column.append(sigma_J)
        sigma_cs_column.append(sigma_H)
        sigma_noe_column.append(sigma_noe)
        data_dir = f"../Systems_v04/{sys_name}/{FF}/nclusters_{nstates}" # FIXME: this is user controlled & might be different in the future
        ############################################################################
        ############################################################################
        ############################################################################
        #pops = row[1][19] # reweighted pops
        pops = row[1][np.where('micro pops' == results.columns.to_numpy())[0]].values[0]
#        print(pops)
#        prior_pops = row[1][21] # prior pops
        prior_pops = row[1][np.where('prior micro pops' == results.columns.to_numpy())[0]].values[0]

        noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/CS_J_NOE/*.noe")]
        #  Get the ensemble average observable
        noe_Exp = noe[0]["exp"].to_numpy()
        noe_model = [i["model"].to_numpy() for i in noe]
#        print(prior_pops)
        try:
            noe_prior = np.array([w*noe_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
        except(Exception) as e:
            continue
        noe_reweighted = np.array([w*noe_model[i] for i,w in enumerate(pops)]).sum(axis=0)


        J = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE/*.J')]
        #  Get the ensemble average observable
        J_Exp = J[0]["exp"].to_numpy()
        J_model = [i["model"].to_numpy() for i in J]

        J_prior = np.array([w*J_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
        J_reweighted = np.array([w*J_model[i] for i,w in enumerate(pops)]).sum(axis=0)


        cs = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE/*.cs*')]
        #  Get the ensemble average observable
        cs_Exp = cs[0]["exp"].to_numpy()
        cs_model = [i["model"].to_numpy() for i in cs]
        cs_atom_name1 = cs[0]["atom_name1"].to_numpy()

        cs_prior = np.array([w*cs_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
        cs_reweighted = np.array([w*cs_model[i] for i,w in enumerate(pops)]).sum(axis=0)

        ######################################################################
        noe_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}-{three2one(row[1]['res2'])}.{row[1]['atom_name2']}" for row in noe[0].iterrows()]
        noe_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2']] for row in noe[0].iterrows()])

        J_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}\n{three2one(row[1]['res2'])}.{row[1]['atom_name2']}\n{three2one(row[1]['res3'])}.{row[1]['atom_name3']}\n{three2one(row[1]['res4'])}.{row[1]['atom_name4']}" for row in J[0].iterrows()]
        J_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2'], row[1]['atom_index3'], row[1]['atom_index4']] for row in J[0].iterrows()])

        cs_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}" for row in cs[0].iterrows()]
        cs_label_indices = np.array([[row[1]['atom_index1']] for row in cs[0].iterrows()])
        ######################################################################


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
                         "sigma": _sigma_noe[i], "label": noe_labels[i]
                })

        J_chi2_prior = compute_chi2(J_Exp, J_prior, _sigma_J)
        J_chi2 = compute_chi2(J_Exp, J_reweighted, _sigma_J)
        J_chi2_biceps = compute_chi2(J_Exp, J_reweighted, np.ones(len(_sigma_J)))
        for i in range(len(J_reweighted)):
            data.append({"index":i,
                "reweighted obs":J_reweighted[i], "prior obs":J_prior[i], "exp obs":J_Exp[i],
                         "sigma": _sigma_J[i], "label": J_labels[i]
                })

        HA_indices = []
        H_indices = []
        for i in range(len(cs_reweighted)):
            #print(cs_atom_name1[i])
            if cs_atom_name1[i] == "HA":
                data.append({"index":i,
                    "reweighted obs":cs_reweighted[i], "prior obs":cs_prior[i], "exp obs":cs_Exp[i],
                             "sigma": sigma_HA, "label": cs_labels[i]
                    })
                HA_indices.append(i)
            elif cs_atom_name1[i] == "H":
                data.append({"index":i,
                    "reweighted obs":cs_reweighted[i], "prior obs":cs_prior[i], "exp obs":cs_Exp[i],
                             "sigma": sigma_H, "label": cs_labels[i]
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
        #data1 = data1.iloc[139:] # NOTE: uncomment to test using only CS and J data
        f_exp, fX, fX_prior, sigma = data1["exp obs"].to_numpy(), data1["reweighted obs"].to_numpy(), data1["prior obs"].to_numpy(), data1["sigma"].to_numpy()
        chi2.append(compute_chi2(f_exp, fX, sigma))
        chi2_prior.append(compute_chi2(f_exp, fX_prior, sigma))
        #Nd = np.sum([noe[0].index.max()+1, cs[0].index.max()+1, J[0].index.max()+1])
        _Nd = {"NOE": noe[0].index.max()+1, "CS_Ha":len(HA_indices), "CS_H":len(H_indices), "J":J[0].index.max()+1}
        Nd = np.sum(list(_Nd.values()))

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


        raw.append({"FF": FF, "nstates": nstates,
            "stat_model": stat_model,
            "exp NOE": noe_Exp,
            "calc NOE": noe_prior,
            "exp J": J_Exp,
            "calc J": J_prior,
            "exp CS_H": cs_Exp[H_indices],
            "calc CS_H": cs_prior[H_indices],
            "exp CS_Ha": cs_Exp[HA_indices],
            "calc CS_Ha": cs_prior[HA_indices],
            "noe_labels": noe_labels,
            "J_labels": J_labels,
            "H_labels": np.array(cs_labels)[H_indices],
            "HA_labels": np.array(cs_labels)[HA_indices],
                     })

    raw = pd.DataFrame(raw)

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

    # make figure
    plt.close()
    plt.clf()
    for col in _Nd.keys():
#        print(col)
#        print(_reduced_prior[col])
        results["reduced chi-squared_prior"] = _reduced_prior[col].to_numpy()/_Nd[col]
        filename = f"{dir}/reduced_chi-squared_{col}_prior.pdf"
#        plot_heatmap_of_chi2_prior(results, reduced=True, Nd=_Nd[col], figname=filename, datatype=col)
#        plt.close()
#        plt.clf()

    results["reduced chi-squared_prior"] = reduced_prior.sum(axis=1).to_numpy()/Nd #chi2_prior/Nd

    # TODO: plot linear correlation plots 3x3 of experimental observable to theoretical observable
    order = ['AMBER14SB','AMBER99','AMBER99SB','AMBER99SB-ildn','AMBER99SBnmr1-ildn','CHARMM22star','CHARMM27','CHARMM36','OPLS-aa']


    data = raw.iloc[np.where( (raw["nstates"] == 100) & (raw["stat_model"] == stat_models[0]) )[0]]

    print(data)
    #exit()

    dir = f"figures/3-by-3-corr"
    biceps.toolbox.mkdir(dir)

    figname = f"{dir}/3-by-3-correlation_NOE.pdf"
    x,y = "exp NOE", "calc NOE"
    xlabel, ylabel = 'Experimental NOE (Å)','Calculated NOE (Å)'
    #labels = "noe_labels"
    correlation_plot_3_by_3(df=data, x_col=x, y_col=y, FF_order=order,
                            xlabel=xlabel, ylabel=ylabel, labels=None,
                            figname=figname, figsize=(12, 12))


    figname = f"{dir}/3-by-3-correlation_J.pdf"
    x,y = "exp J", "calc J"
    xlabel, ylabel = 'Experimental J-coupling (Hz)','Calculated J-coupling (Hz)'
    labels = "J_labels"
    correlation_plot_3_by_3(df=data, x_col=x, y_col=y, FF_order=order,
                            xlabel=xlabel, ylabel=ylabel, labels=labels,
                            figname=figname, figsize=(12, 12))

    figname = f"{dir}/3-by-3-correlation_CS_H.pdf"
    x,y = "exp CS_H", "calc CS_H"
    xlabel, ylabel = 'Experimental Chemical Shift H (ppm)','Calculated Chemical Shift H (ppm)'
    labels = "H_labels"
    correlation_plot_3_by_3(df=data, x_col=x, y_col=y, FF_order=order,
                            xlabel=xlabel, ylabel=ylabel, labels=labels,
                            figname=figname, figsize=(12, 12))

    figname = f"{dir}/3-by-3-correlation_CS_Ha.pdf"
    x,y = "exp CS_Ha", "calc CS_Ha"
    xlabel, ylabel = r'Experimental Chemical Shift H$\alpha$ (ppm)',r'Calculated Chemical Shift H$\alpha$ (ppm)'
    labels = "HA_labels"
    correlation_plot_3_by_3(df=data, x_col=x, y_col=y, FF_order=order,
                            xlabel=xlabel, ylabel=ylabel, labels=labels,
                            figname=figname, figsize=(12, 12))














