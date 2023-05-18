# Libraries:{{{
import string
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
import uncertainties as u ################ Error Prop. Library
# }}}


sys_name = "CLN001"

FF_list = ["AMBER14SB","AMBER99SB-ildn","CHARMM27","AMBER99",
    "AMBER99SBnmr1-ildn","CHARMM36","AMBER99SB","CHARMM22star","OPLS-aa"]

experimental_folded = u.ufloat(0.61029, 0.03375)
val = experimental_folded.nominal_value
dev = experimental_folded.std_dev
experimental_lower, experimental_upper = val-dev, val+dev




# Methods:{{{
def transform_coord(a,b,c):
    x = 0.5*(2*c+a)/(a+b+c)
    y = 0.5*(a*np.sqrt(3))/(a+b+c)
    return x,y

def triangle_plot(labels, a, b, c, FF, figname="populations.png", figsize=(9,8)):

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    #colors = ["b", "g", "r", "m", "c", "orange", "yellow", "lime", "k", "fuchia"]
    colors = ["b", "g", "r", "m", "c", "orange", "lime","olive", "gray"]+list(colors)
    #print(colors)
    #exit()
    fig = plt.figure(figsize=figsize)
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

#    ax.scatter(0.0,0.0, color="r", marker="+", linewidth=2.5, s=250, label="_nolegend_")
    ax.scatter(0.0,0.0, color="k", marker="+", linewidth=2.5, s=250, label="_nolegend_")
    for k,tick in enumerate(ticks[1:]):
        #print(tick)
        x,y = transform_coord(0.0,tick,0.0)
#        ax.scatter(tick, y, color="y", marker="+", linewidth=2.5, s=250, label="_nolegend_")
        #ax.scatter(tick, y, color="k", marker="+", linewidth=2.5, s=250, label="_nolegend_")

        x,y = transform_coord(tick,0.0,0.0)
#        ax.scatter(x, y*tick, color="r", marker="+", linewidth=2.5, s=250, label="_nolegend_")

#        ax.scatter(x*tick, y*tick, color="b", marker="+", linewidth=2.5, s=250, label="_nolegend_")
        ax.scatter(x*tick, y*tick, color="k", marker="+", linewidth=2.5, s=250, label="_nolegend_")

#        ax.scatter(0.5+x*tick, y*ticks[-2:None:-1][k], color="c", marker="+", linewidth=2.5, s=250, label="_nolegend_")
        ax.scatter(0.5+x*tick, y*ticks[-2:None:-1][k], color="k", marker="+", linewidth=2.5, s=250, label="_nolegend_")

        ################
#        x,y = transform_coord(tick,0.0,0.0)
#        ax.scatter(tick, y, color="k", marker="+", linewidth=2.5, s=250, label="_nolegend_")

        #x,y = transform_coord(tick,0.0,0.0)
        #ax.scatter(x, y, color="r", marker="+", linewidth=2.5, s=250)

        #ax.scatter(tick, y*np.sqrt(2)/2+tick, color="r", marker="+", linewidth=2.5, s=250)

#    ############################################################################
#    # NOTE: add a shaded region for experimental folded state population
#    vals = [transform_coord(tick,0.0,0.0) for tick in ticks[5:8]]
#    sub = []
#    for k, v in enumerate(vals):
#        vx,vy = v
#        sub.append(np.array([0.35+vx*ticks[5:8][k], vy*ticks[-3:None:-1][k]]))
#    sub = np.array(sub).T
#
#    x,y = np.concatenate([
#        np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in ticks[6:9]]).T,
#        sub,], axis=1)
#
#    ax.fill(x, y, facecolor='y', alpha=0.15)
#    ############################################################################

    ############################################################################
    # NOTE: add a shaded region for experimental folded state population ( in red)
    tick_loc = [experimental_lower, experimental_upper]
    vals = [transform_coord(tick,0.0,0.0) for tick in tick_loc]
    sub = []
    for k, v in enumerate(vals):
        vx,vy = v
        sub.append(np.array([0.39+vx*tick_loc[k], vy*tick_loc[::-1][k]]))
    sub = np.array(sub).T

    x,y = np.concatenate([
        #np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in ticks[6:9]]).T,
        np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in tick_loc]).T,
        sub,], axis=1)

    #ax.fill(x, y, facecolor='y', alpha=0.25)
    ax.fill(x, y, facecolor='orange', alpha=0.3)
    ############################################################################

    ############################################################################
    # NOTE: add a shaded region for experimental folded state population
    tick_loc = [experimental_upper, 1.0]
    vals = [transform_coord(tick,0.0,0.0) for tick in tick_loc]
    sub = []
    for k, v in enumerate(vals):
        vx,vy = v
        sub.append(np.array([0.18+vx*tick_loc[k], vy*tick_loc[::-1][k]]))
    sub = np.array(sub).T

    x,y = np.concatenate([
        #np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in ticks[6:9]]).T,
        np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in tick_loc]).T,
        sub,], axis=1)

    ax.fill(x, y, facecolor='y', alpha=0.15)
    ############################################################################



    FFi = FF[0]
    stat_models = list(set(labels))
    stat_models = {model:j for j,model in enumerate(stat_models)}
    forcefields = list(set(FF))
    forcefields = {_ff:j for j,_ff in enumerate(forcefields)}

    for i in range(len(labels)):
        j = stat_models[labels[i]]
        k = forcefields[FF[i]]
        x,y = transform_coord(a[i],b[i],c[i])
        ax.scatter(x,y, c=colors[j],
                   #alpha=0.35,
                   alpha=0.65,
                   marker=markers[k],
                   edgecolor="k",
                   linewidths=2,
                   #label=labels[i],
                   label="_nolegend_",
                   s=300)


    #print(matplotlib.rcParams.keys())
    #x_range = [-0.2,1.1]
    x_range = [-0.02, 1.02]
    y_range = [-0.02,1.0]

    # NOTE: creating legend text for labeling the markers for each force field
    inc = 0.0
    for j in range(len(FF_list)):
        inc -= 0.05
        x,y = 0.035, 1.04+inc
        ax.text(x, y,
                '%s'%(list(forcefields.keys())[j]),
                transform=ax.transAxes, fontsize=14, color="k",
                verticalalignment='top',
                )
        ax.scatter(x+x_range[0]-0.02,
                   #y+(y_range[1]-y),
                   y-0.015,
                   color="k", marker=markers[j], s=200)

    # NOTE: creating legend text for color coding the statistical models
    #inc = -0.05
    for j in range(len(list(stat_models.keys()))):
        inc -= 0.05
        #props = dict(boxstyle='square', facecolor=None, alpha=0.5, pad=0.3)
        x,y = 0.035, 1.04+inc
        ax.text(x, y,
                '%s'%(list(stat_models.keys())[j]),
                transform=ax.transAxes, fontsize=14, color=colors[j],
                verticalalignment='top',
                )



    # NOTE: Creating letters for Folded, F; Unfolded, U; Misfolded M
    x,y = 0.49, 0.94
    ax.text(x, y, 'F', transform=ax.transAxes, fontsize=30, color="k", verticalalignment='top')
    x,y = -0.04, 0.04
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
# }}}

# triangle_subplots:{{{
def triangle_subplots(labels, a, b, c, FF, figname="populations.png",
                      grid=(2, 2), positions=[(0,0), (0,1), (1,0), (1,1)],
                      figsize=(10, 9.)):

    markers = matplotlib.markers.MarkerStyle.filled_markers
    colors = matplotlib.colors
    colors = np.array(list(colors.__dict__['CSS4_COLORS'].values()))[10::3]
    colors = ["b", "g", "r", "m", "c", "orange", "fuchia"]+list(colors)
    #print(colors)
    #exit()
    fig = plt.figure(figsize=figsize) #(9, 8)) # figsize=(12, 12)
    gs = gridspec.GridSpec(grid[0], grid[1])#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    print(labels)
    _labels = labels.copy()
    _a = a.copy()
    _b = b.copy()
    _c = c.copy()
    _FF = FF.copy()
    set_of_FF = list(set(FF[0]))
    print(set_of_FF)

    marker_dict = {set_of_FF[i]: markers[i] for i in range(len(set_of_FF))}
    print(marker_dict)
    stat_model_color_dict = {}

    #positions = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(len(labels)):
        ax = plt.subplot(gs[positions[i]])
        labels = _labels[i]
        a = _a[i]
        b = _b[i]
        c = _c[i]
        FF = _FF[i]

        #ax = plt.subplot(gs[0])

        # draw the triangle grid
        x,y = transform_coord(0.,0.,1.)
        ax.plot([0.0, x],[0.0, y], color="k", linewidth=2.0, label="_nolegend_")
        x,y = transform_coord(1.,0.,0.)
        ax.plot([0.0, x],[0.0, y], color="k", linewidth=2.0, label="_nolegend_")
        x,y = transform_coord(1.,0.,0.)
        ax.plot([1.0, x],[0.0, y], color="k", linewidth=2.0, label="_nolegend_")

        ticks = np.linspace(0., 1.0, 11)

        ax.scatter(0.0,0.0, color="k", marker="+", linewidth=1.5, s=150, label="_nolegend_")
        for k,tick in enumerate(ticks[1:]):
            x,y = transform_coord(0.0,tick,0.0)
            ax.scatter(tick, y, color="k", marker="+", linewidth=1.5, s=150, label="_nolegend_")

            x,y = transform_coord(tick,0.0,0.0)

            ax.scatter(x*tick, y*tick, color="k", marker="+", linewidth=1.5, s=150, label="_nolegend_")

            ax.scatter(0.5+x*tick, y*ticks[-2:None:-1][k], color="k", marker="+", linewidth=1.5, s=150, label="_nolegend_")

        ############################################################################
        # NOTE: add a shaded region for experimental folded state population ( in red)
        tick_loc = [experimental_lower, experimental_upper]
        vals = [transform_coord(tick,0.0,0.0) for tick in tick_loc]
        sub = []
        for k, v in enumerate(vals):
            vx,vy = v
            sub.append(np.array([0.39+vx*tick_loc[k], vy*tick_loc[::-1][k]]))
        sub = np.array(sub).T

        x,y = np.concatenate([
            #np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in ticks[6:9]]).T,
            np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in tick_loc]).T,
            sub,], axis=1)

        #ax.fill(x, y, facecolor='y', alpha=0.25)
        ax.fill(x, y, facecolor='orange', alpha=0.4)
        ############################################################################

        ############################################################################
        # NOTE: add a shaded region for experimental folded state population
        tick_loc = [experimental_upper, 1.0]
        vals = [transform_coord(tick,0.0,0.0) for tick in tick_loc]
        sub = []
        for k, v in enumerate(vals):
            vx,vy = v
            sub.append(np.array([0.18+vx*tick_loc[k], vy*tick_loc[::-1][k]]))
        sub = np.array(sub).T

        x,y = np.concatenate([
            #np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in ticks[6:9]]).T,
            np.array([np.array(transform_coord(tick,0.0,0.0))*tick for tick in tick_loc]).T,
            sub,], axis=1)

        ax.fill(x, y, facecolor='y', alpha=0.15)
        ############################################################################


        FFi = FF[0]
        stat_models = list(set(labels))
        stat_models = {model:j for j,model in enumerate(stat_models)}
        forcefields = list(set(FF))
        forcefields = {_ff:j for j,_ff in enumerate(forcefields)}

        for l in range(len(labels)):
            #print(stat_models)
            j = stat_models[labels[l]]
            #print(j)
            #exit()
            k = forcefields[FF[l]]
            x,y = transform_coord(a[l],b[l],c[l])
            if labels[l] == "MSM":
                stat_model_color_dict["MSM"] = "b"
            else:
                stat_model_color_dict[labels[l]] = "g"
            ax.scatter(x, y,
                       #color=colors[j],
                       color=stat_model_color_dict[labels[l]],
                       #alpha=0.35,
                       alpha=0.65,
                       #marker=markers[k],
                       marker=marker_dict[FF[l]],
                       edgecolor="k",
                       linewidths=2,
                       #label=labels[i],
                       label="_nolegend_",
                       s=300)
            ax.set_aspect('equal')


        #print(matplotlib.rcParams.keys())
        #x_range = [-0.2,1.1]
        x_range = [-0.02, 1.02]
        y_range = [-0.04,1.0]

        if i == 0:
            # NOTE: creating legend text for labeling the markers for each force field
            inc = 0.0
            print(FF_list)
            for j in range(len(FF_list)):
                inc -= 0.08
                x,y = 1.0, 1.04+inc
                ax.text(x, y,
                        '%s'%(list(forcefields.keys())[j]),
                        transform=ax.transAxes, fontsize=16, color="k",
                        verticalalignment='top',
                        )
                if j < len(FF_list)/2:
                    shifty = 0.035
                else:
                    shifty = 0.045
                ax.scatter(x+x_range[0]-0.02,
                           #y+(y_range[1]-y),
                           y-shifty,
                           color="k",
                           #marker=markers[j],
                           marker=marker_dict[list(forcefields.keys())[j]],
                           s=200)


            # NOTE: creating legend text for color coding the statistical models
            #inc = -0.05
            for j in range(len(list(stat_models.keys()))):
                if "MSM" != list(stat_models.keys())[j]: continue

                #inc -= 0.05
                inc -= 0.08
                #props = dict(boxstyle='square', facecolor=None, alpha=0.5, pad=0.3)
                #x,y = 0.035, 1.04+inc
                x,y = 1.0, 1.04+inc
                ax.text(x, y,
                        '%s'%(list(stat_models.keys())[j]),
                        transform=ax.transAxes, fontsize=16, color=stat_model_color_dict[labels[l]],
                        verticalalignment='top',
                        fontweight="bold"
                        )



        # NOTE: Creating letters for Folded, F; Unfolded, U; Misfolded M
        #x,y = 0.49, 0.94
        #x,y = 0.49, 1.
        x,y = 0.49, 0.975
        ax.text(x, y, 'F', transform=ax.transAxes, fontsize=22, color="k", verticalalignment='top')
        #x,y = -0.04, 0.04
        x,y = -0.06, 0.04
        ax.text(x, y, 'U', transform=ax.transAxes, fontsize=22, color="k", verticalalignment='top')
        x,y = 1., 0.04
        ax.text(x, y, 'M', transform=ax.transAxes, fontsize=22, color="k", verticalalignment='top')


        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles, loc='best', fontsize=16)
        ax.axis('off')
        title = list(set(labels))
        title = [s for s in title if s != "MSM"]
        title_color = stat_model_color_dict[[model for model in list(stat_model_color_dict.keys()) if model != "MSM"][0]]
        #print(title_color)
        try:
            print(title)
            ax.set_title(f"{title[0]}", color=title_color, size=16, fontweight="bold")
        except(Exception) as e:
            print(e)


        x,y = -0.1, 1.02
        ax.text(x,y, string.ascii_lowercase[i], transform=ax.transAxes,
                size=20, weight='bold')

    #gs.tight_layout(fig, pad=1.5, w_pad=0.5, h_pad=2.0)
    #gs.tight_layout(fig, pad=1., w_pad=-1.5, h_pad=2.0)
    #gs.tight_layout(fig, pad=1., w_pad=-5.0, h_pad=2.0)
    gs.tight_layout(fig, pad=1., w_pad=-8.0, h_pad=2.0)
    fig.savefig(f"{figname}", dpi=600)
# }}}



if __name__ == "__main__":

    # get files for all force fields and plot biceps score comparison
    files = biceps.toolbox.get_files(f"{sys_name}/*/nclusters_*/*/*/results.pkl")
    results = []

    stat_models = ["BayesianModel", "Good-Bad", "StudentsModel", "GaussianModel"]
    for file in files:
        for stat_model in stat_models:
            if stat_model in file:
                add_file = True
                break
            else:
                add_file = False
        if add_file:
            results.append(pd.read_pickle(file))
    results = pd.concat(results)

    print(list(set(results["stat_model"].to_numpy())))
    #exit()



    biceps.toolbox.mkdir("figures")
    _states_ = [5,10,50,75,100,500]
    _results = []
    for FF in FF_list:
        for clusters in _states_:
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
#            print(macrostate_populations)

            results = []
            if sys_name == "CLN001":
                results.append({
                    "stat_model": "Experiment", "nreplicas": 0, "nlambdas": 0, "nsteps": 0, "data_uncertainty": "none",
                    0: experimental_folded.nominal_value, 1:0, 2:0
                    })
                results.append({
                    "FF": FF,
                    "stat_model": "MSM", "nstates": nstates, "nreplicas": 0, "nlambdas": 0, "nsteps": 0,  "data_uncertainty": "none",
                    **macrostate_populations.to_dict()[macrostate_populations.columns[0]]
                    })
                _results.append(pd.DataFrame(results))


            for file in files:
                print(file)
                stat_model = file.split("/")[-1].split("_single_sigma")[0].split("_multiple_sigma")[0]
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
                    "nstates": nstates,
                    "stat_model": stat_model,
                    "nreplicas": nreplicas,
                    "nlambdas": nlambdas,
                    "nsteps": nsteps,
                    "data_uncertainty": data_uncertainty,
                    **pop_dict
                    })
                _results.append(pd.DataFrame(results))


    df = pd.concat(_results)
    # NOTE: Removing SinglePrior model from the plots
#    df = df.iloc[np.where(df["stat_model"] != "SinglePrior")[0]]
#    df = df.iloc[np.where(df["stat_model"] != "StudentsOut")[0]]
#    df = df.iloc[np.where(df["stat_model"] != "Students")[0]]
    #df = df.iloc[np.where(df["stat_model"] != "GaussainSP")[0]]
    #df = df.iloc[np.where(df["stat_model"] != "SingleSigma")[0]]

    # NOTE: getting standard deviation between trials
    pops_std = df.groupby(["stat_model", "FF", "nstates", "nreplicas", "nlambdas", "nsteps", "data_uncertainty"]).agg("std")
    print(pops_std)


    # NOTE: getting average populations over trials
    pops_mean = df.groupby(["stat_model", "FF", "nstates", "nreplicas", "nlambdas", "nsteps", "data_uncertainty"]).agg("mean")
    df = pops_mean.reset_index()

    #results = df.copy()
    #df = pd.DataFrame(results)
    dir = f"figures/triangle_populations"
    biceps.toolbox.mkdir(dir)

    for nstates in _states_:
        results = df.iloc[np.where(df["nstates"] == nstates)[0]].copy()
        #results = df.copy()
        results = results.fillna(0)
        labels = results["stat_model"].to_numpy()
        FF = results["FF"].to_numpy()
        b = results[2].to_numpy() # unfolded = left
        c = results[1].to_numpy() # misfolded = right
        a = results[0].to_numpy() # folded = top
        triangle_plot(labels, a, b, c, FF, figname=f"{dir}/triangle_populations_{nstates}.pdf")

    # create plots for each stat_model with 500 states
    _nstates_ = [500]
    _nstates_ = [5,10,50,75,100,500]
    for nstates in _nstates_:
        results = df.iloc[np.where(df["nstates"] == nstates)[0]].copy()
        results = results.fillna(0)

        new_sm = ["Bayesian", "Good-Bad", "Student's", "Gaussian"]
        for i in range(len(new_sm)):
            results = biceps.toolbox.change_stat_model_name(results, sm=stat_models[i], new_sm=new_sm[i])

        model_order = new_sm.copy()


        _labels,_FF,_a,_b,_c = [],[],[],[],[]
        #for stat_model in list(set(results["stat_model"].to_numpy())):
        for stat_model in model_order:
            _results = results.iloc[np.where((results["stat_model"] == stat_model) | (results["stat_model"] == "MSM"))[0]]
            _results = _results.sort_values("stat_model")
            labels = _results["stat_model"].to_numpy()
            FF = _results["FF"].to_numpy()
            b = _results[2].to_numpy() # unfolded = left
            c = _results[1].to_numpy() # misfolded = right
            a = _results[0].to_numpy() # folded = top
            if stat_model == "MSM": continue
            _labels.append(labels)
            FF = [ff.replace('AMBER','A').replace('CHARM','C') for ff in FF]
            _FF.append(FF)
            _a.append(a)
            _b.append(b)
            _c.append(c)
            plt.close()
            triangle_plot(labels, a, b, c, FF, figname=f"{dir}/triangle_populations_{nstates}_{stat_model}.pdf")

        triangle_subplots(_labels, _a, _b, _c, _FF,
                          grid=(2, 2), positions=[(0,0), (0,1), (1,0), (1,1)],figsize=(10, 9.),
                          figname=f"{dir}/triangle_subplots_populations_{nstates}.pdf")

        triangle_subplots(_labels, _a, _b, _c, _FF,
#                          grid=(2, 2), positions=[(0,0), (0,1), (1,0), (1,1)],figsize=(10, 9.),
                          #grid=(3, 2), positions=[(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],figsize=(12, 12),
                          #grid=(4, 2), positions=[(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)],figsize=(12, 14),
#                          figname=f"{dir}/triangle_subplots_populations_{nstates}.pdf")
                          grid=(1, 4), positions=[(0,0), (0,1), (0,2), (0,3)], figsize=(20,7),
                          figname=f"{dir}/triangle_subplots_populations_{nstates}_horizontal.pdf")

#        ###############################################################################
#        #model_order = ["Bayesian", "OutliersSP"]
#        model_order = ["Bayesian"]
#        _labels,_FF,_a,_b,_c = [],[],[],[],[]
#        #for stat_model in list(set(results["stat_model"].to_numpy())):
#        for stat_model in model_order:
#            _results = results.iloc[np.where((results["stat_model"] == stat_model) | (results["stat_model"] == "MSM"))[0]]
#            _results = _results.sort_values("stat_model")
#            labels = _results["stat_model"].to_numpy()
#            FF = _results["FF"].to_numpy()
#            b = _results[2].to_numpy() # unfolded = left
#            c = _results[1].to_numpy() # misfolded = right
#            a = _results[0].to_numpy() # folded = top
#            if stat_model == "MSM": continue
#            _labels.append(labels)
#            FF = [ff.replace('AMBER','A').replace('CHARM','C') for ff in FF]
#            _FF.append(FF)
#            _a.append(a)
#            _b.append(b)
#            _c.append(c)
#            plt.close()
#        triangle_subplots(_labels, _a, _b, _c, _FF,
#                          grid=(1, 1), positions=[(0,0)], figsize=(7,7),
#                          figname=f"{dir}/triangle_subplots_populations_{nstates}_Bayes.pdf")
#                          #figname=f"{dir}/triangle_subplots_populations_{nstates}_Bayes_OutliersSP.pdf")








    exit()







