import biceps
import pandas as pd
import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None

import seaborn as sb
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


sys_name = "CLN001"

FF_list = ["AMBER14SB","AMBER99SB-ildn","CHARMM27","AMBER99",
    "AMBER99SBnmr1-ildn","CHARMM36","AMBER99SB","CHARMM22star","OPLS-aa"]



def transform_coord(a,b,c):
    x = 0.5*(2*c+a)/(a+b+c)
    y = 0.5*(a*np.sqrt(3))/(a+b+c)
    return x,y

def triangle_plot(labels, a, b, c, FF, figname="populations.png"):

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    #print(len(markers))
    #print(len(colors))
    #print(len(labels))
    #print(len(FF))
    #print(colors)
    #exit()
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])

    # draw the triangle grid
    x,y = transform_coord(0.,0.,1.)
    ax.plot([0.0, x],[0.0, y], color="k", linewidth=3.0, label="_nolegend_")
    x,y = transform_coord(1.,0.,0.)
    ax.plot([0.0, x],[0.0, y], color="k", linewidth=3.0, label="_nolegend_")
    x,y = transform_coord(1.,0.,0.)
    ax.plot([1.0, x],[0.0, y], color="k", linewidth=3.0, label="_nolegend_")

    ticks = np.linspace(0., 1.0, 11)
    #print(ticks)
    #exit()

    ax.scatter(0.0,0.0, color="r", marker="+", linewidth=2.5, s=450, label="_nolegend_")
    for k,tick in enumerate(ticks[1:]):
        #print(tick)
        x,y = transform_coord(0.0,tick,0.0)
        ax.scatter(tick, y, color="y", marker="+", linewidth=2.5, s=450, label="_nolegend_")

        x,y = transform_coord(tick,0.0,0.0)
        ax.scatter(x, y*tick, color="r", marker="+", linewidth=2.5, s=450, label="_nolegend_")

        ax.scatter(x*tick, y*tick, color="b", marker="+", linewidth=2.5, s=450, label="_nolegend_")

        ax.scatter(0.5+x*tick, y*ticks[-2:None:-1][k], color="c", marker="+", linewidth=2.5, s=450, label="_nolegend_")

        ################
#        x,y = transform_coord(tick,0.0,0.0)
#        ax.scatter(tick, y, color="k", marker="+", linewidth=2.5, s=450, label="_nolegend_")

        #x,y = transform_coord(tick,0.0,0.0)
        #ax.scatter(x, y, color="r", marker="+", linewidth=2.5, s=450)

        #ax.scatter(tick, y*np.sqrt(2)/2+tick, color="r", marker="+", linewidth=2.5, s=450)



    ############################################################################
    # NOTE: add a shaded region for experimental folded state population
    vals = [transform_coord(tick,0.0,0.0) for tick in ticks[5:8]]
    sub = []
    for k, v in enumerate(vals):
        vx,vy = v
        sub.append(np.array([0.35+vx*ticks[5:8][k], vy*ticks[-3:None:-1][k]]))
    sub = np.array(sub).T

    x,y = np.concatenate([
        np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in ticks[6:9]]).T,
        sub,], axis=1)

    ax.fill(x, y, facecolor='y', alpha=0.15)
    ############################################################################

    FFi = FF[0]
    stat_models = list(set(labels))
    stat_models = {model:j for j,model in enumerate(stat_models)}
    forcefields = list(set(FF))
    forcefields = {_ff:j for j,_ff in enumerate(forcefields)}

    for i in range(len(labels)):
        #print(labels[i])
        j = stat_models[labels[i]]
        k = forcefields[FF[i]]
        x,y = transform_coord(a[i],b[i],c[i])
        #ax.scatter(x,y, marker=markers[k], color=colors[k], label=labels[i])
        ax.scatter(x,y, color=colors[j],
                   alpha=0.35,
                   marker=markers[k],
                   edgecolor="k",
                   #label=labels[i],
                   label="_nolegend_",
                   s=200)


    #print(matplotlib.rcParams.keys())
    #x_range = [-0.2,1.1]
    x_range = [-0.02, 1.02]
    y_range = [-0.02,1.0]

    # NOTE: creating legend text for labeling the markers for each force field
    inc = 0.0
    for j in range(len(FF_list)):
        inc -= 0.05
        x,y = 0.03, 1.04+inc
        ax.text(x, y,
                '%s'%(list(forcefields.keys())[j]),
                transform=ax.transAxes, fontsize=14, color="k",
                verticalalignment='top',
                )
        ax.scatter(x+x_range[0]-0.02,
                   #y+(y_range[1]-y),
                   y-0.01,
                   color="k", marker=markers[j], s=120)

    # NOTE: creating legend text for color coding the statistical models
    #inc = -0.05
    for j in range(len(list(stat_models.keys()))):
        inc -= 0.05
        #props = dict(boxstyle='square', facecolor=None, alpha=0.5, pad=0.3)
        x,y = 0.03, 1.04+inc
        ax.text(x, y,
                '%s'%(list(stat_models.keys())[j]),
                transform=ax.transAxes, fontsize=14, color=colors[j],
                verticalalignment='top',
                )



    # NOTE: Creating letters for Folded, F; Unfolded, U; Misfolded M
    x,y = 0.49, 0.92
    ax.text(x, y, 'F', transform=ax.transAxes, fontsize=30, color="k", verticalalignment='top')
    x,y = -0.03, 0.04
    ax.text(x, y, 'U', transform=ax.transAxes, fontsize=30, color="k", verticalalignment='top')
    x,y = 1., 0.04
    ax.text(x, y, 'M', transform=ax.transAxes, fontsize=30, color="k", verticalalignment='top')



    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, loc='best', fontsize=16)
    plt.axis('off')
    fig.tight_layout(pad=1.5, w_pad=0.75, h_pad=2.25)
    fig.savefig(f"{figname}", dpi=600)


# get files for all force fields and plot biceps score comparison
files = biceps.toolbox.get_files(f"{sys_name}/*/nclusters_*/*/*/results.pkl")
results = []
for file in files:
    results.append(pd.read_pickle(file))
results = pd.concat(results)
biceps.toolbox.mkdir("figures")

_results = []
for FF in FF_list:
    for clusters in [500]:
        files = biceps.toolbox.get_files(f"{sys_name}/{FF}/nclusters_{clusters}/*/*/*__reweighted_populations.csv")
        # Get the files for prior pops and assignments (microstate --> macrostate)
        # ..for each FF/microstate clustering
        assignment_files = biceps.toolbox.get_files(f"../Systems_v04/{sys_name}/{FF}/nclusters_{clusters}/inverse_distances_k*_msm_assignments.csv")
        prior_pops_files = biceps.toolbox.get_files(f"../Systems_v04/{sys_name}/{FF}/nclusters_{clusters}/inverse_distances_k*_msm_pops.csv")
        assignment_df = pd.read_csv(assignment_files[0], index_col=0)
        prior_df =  pd.read_csv(prior_pops_files[0], index_col=0)
        # Get a dataframe of prior macrostate populations (from MSM)
        microstates = assignment_df.index.to_numpy()
        macrostates = assignment_df[assignment_df.columns[0]].to_numpy()
        prior_pops = prior_df[prior_df.columns[0]].to_numpy()
        nstates = len(prior_pops)
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

        results = []
        if sys_name == "CLN001":
            results.append({
                "stat_model": "Experiment", "nreplicas": 0, "nlambdas": 0, "nsteps": 0, "data_uncertainty": "none",
                2: 0.83, 1:0, 1:0
                })
            results.append({

                "FF": FF,
                "stat_model": "MSM", "nstates": nstates, "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
                **macrostate_populations.to_dict()[macrostate_populations.columns[0]]
                })
            _results.append(pd.DataFrame(results))

#        for file in files:
#            print(file)
#            stat_model = file.split("/")[-1].split("_")[0]
#            data_uncertainty = file.split("/")[-1].split("_sigma")[0].split("_")[-1]
#            nreplicas = int(file.split("/")[-2].split("_replicas")[0].split("_")[-1])
#            nsteps = int(file.split("/")[-2].split("_steps")[0].split("_")[-1])
#            nlambdas = int(file.split("/")[-2].split("_lam")[0].split("_")[-1])
#            df = pd.read_csv(file)
#            grouped = df.groupby(["macrostate"])
#            df = df.set_index("macrostate")
#            pop_dict = grouped.sum()["population"].to_dict()
#            results.append({
#                "FF": FF,
#                "nstates": nstates,
#                "stat_model": stat_model,
#                "nreplicas": nreplicas,
#                "nlambdas": nlambdas,
#                "nsteps": nsteps,
#                "data_uncertainty": data_uncertainty,
#                **pop_dict
#                })
#            _results.append(pd.DataFrame(results))
#            #print(results)
            #exit()



        #try:
        #    #ax = grouped.bar([4,3,2,1], alpha=0.5, edgecolor='black', linewidth=1.2, color="b", figsize=(14, 6))
        #    ax = df.plot.bar(x="stat_model", y=[2,1,0],  figsize=(10, 4), rot=65)
        #    ax.axhline(y=df.iloc[np.where(df["stat_model"] == "Experiment")[0]].to_dict()[2][0], color='k', linestyle="--")
        #    xticklabels = []
        #    model_labels = df["stat_model"].to_numpy()
        #    replica_labels = df["nreplicas"].to_numpy()
        #    step_labels = df["nsteps"].to_numpy()
        #    for i in range(len(model_labels)):
        #        if replica_labels[i] == 0:
        #            xticklabels.append(model_labels[i])
        #        if replica_labels[i] != 0:
        #            xticklabels.append(str(str(model_labels[i])+"\n"+str(step_labels[i])+" steps & "+str(replica_labels[i])+" replica"))
        #    ax.set_xticklabels([f"{x}" for x in xticklabels])
        #    ax.set_xlabel(r"", fontsize=16)
        #    ax.set_ylabel(r"", fontsize=16)
        #    ax.set_ylim(0,1)
        #    #if sys_name == "CLN001": ax.legend(["Exp", "4: folded","3: misfolded", "2: helix", "1: unfolded"], loc="best")
        #    #if sys_name == "CLN025": ax.legend(["Exp", "4: folded","3: misfolded", "2: unfolded", "1: helix"], loc="best")
        #    if sys_name == "CLN001": ax.legend(["Exp", "2: folded","1: misfolded", "0: unfolded"], loc="best")
        #    if sys_name == "CLN025": ax.legend(["Exp", "4: folded","3: misfolded", "2: unfolded", "1: helix"], loc="best")
        #    fig = ax.get_figure()
        #    fig.tight_layout()
        #    biceps.toolbox.mkdir(f"{clusters}_microstates")
        #    fig.savefig(f"{clusters}_microstates/{sys_name}_{sys_name}_{FF}_prelim_results.png")
        #    #print(f"RMSE = {df['error'].mean()} ± {df['error'].std()}")
        #except(Exception) as e:
        #    continue


df = pd.concat(_results)
#df = pd.DataFrame(results)
df = df.iloc[np.where(df["nstates"] == 500)[0]]
#print(df)
#exit()
results = df.copy()
results = results.iloc[np.where(results["stat_model"].to_numpy()=="MSM")[0]]
#print(results)
#exit()
results = results.fillna(0)
labels = results["stat_model"].to_numpy()
FF = results["FF"].to_numpy()
b = results[2].to_numpy() # unfolded = left
c = results[1].to_numpy() # misfolded = right
a = results[0].to_numpy() # folded = top
triangle_plot(labels, a, b, c, FF, figname=f"figures/triangle_populations_prior.png")
exit()







