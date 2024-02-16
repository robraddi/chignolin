
# Python libraries:{{{
import string
import numpy as np
import sys, time, os, gc
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import scipy
from scipy.stats import maxwell
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
#from scipy import stats
import biceps
#from biceps.decorators import multiprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#:}}}

# Methods:{{{
def normalize(v, order=1, axis=0):
    norm = np.atleast_1d(np.linalg.norm(v, ord=order, axis=axis))
    norm[norm==0] = 1
    #if norm == 0: return v
    return v / np.expand_dims(norm, axis)

# get_boltzmann_weighted_states:{{{
def get_boltzmann_weighted_states(nstates, σ=0.16, domain=(4, 6), loc=3.15,
                                  scale=2.0, verbose=False, plot=True):
    """Get Boltzmann weighted states and perturbed states with error equal to σ.

    Args:
        nstates(int): number of states
        σ(float): sigma is the error in the prior

    Returns:
        botlzmann weighted states and perturbed states inside tuple \
                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
    """
    import scipy.stats as stats
    iter = True

    kT = 1#0.5959 # kcal/mol
    frozen = maxwell(loc=loc, scale=scale)
    mean, var, skew, kurt = frozen.stats(moments='mvsk')
    pops = [0.65, 0.15, 0.20]
    energies = -np.log(pops)
    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
    while iter == True:
        samples = df
        #if plot:
        #    ax = df.plot(x="E", y="Pops", kind="line", color="k", figsize=(14, 6))
        #    fig = ax.get_figure()
        #    # sample energies from Boltzmann distribution
        #    for i in samples.to_numpy():
        #        ax.axvline(x=i[0], ymin=0, ymax=1, color="red")#print(energies)
        #    fig.savefig("boltzmann_energies.png")

        #samples = samples.sort_values(["Pops"], ascending=False).reset_index(drop=True)
        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()

        perturbed_E = E+σ*np.random.randn(len(w))
        perturbed_w = normalize(frozen.pdf(perturbed_E))[0]
        if not np.array_equal(np.round(perturbed_w, 2),np.array([.14, .37, .49])):
           print(perturbed_w)
           continue
        else:
           print(perturbed_w)

        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
        if all(i >= 0.0 for i in perturbed_w): iter = False
        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
        except(Exception) as e: iter = True

    print(f"Populations:           {w}")
    print(f"Perturbed Populations: {perturbed_w}")
    if verbose:
        print(f"Std Error per state (Pops): {np.std([w.transpose(), perturbed_w.transpose()], axis=0)}")
        print(f"Avg Std Error per state (Pops):     {np.std([w.transpose(), perturbed_w.transpose()], axis=0).mean()}")
        print(f"Avg Std Error per state (Energies): {np.std([E.transpose(), perturbed_E.transpose()], axis=0).mean()}")
        print(f"RMSE in populations: {np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))}")

    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]
# }}}

def gen_synthetic_data(dir, nStates, Nd, σ_prior=0, μ_prior=0, σ_data=0, μ_data=0,
        boltzmann_domain=(4, 6), boltzmann_loc=3.15, boltzmann_scale=2.0,
        draw_from=["uniform","uniform"], verbose=False):
    """Generate synthetic input files for biceps including the
    experimental NOE data weighted by all the states. To mimic random errors,
    Gaussian noise can be added by tuning the parameter σ. To mimic systematic
    errors, a shift in the data can be achieved by changing μ.
    """

    biceps.toolbox.mkdir(dir)
    states,perturbed_states,RMSE = get_boltzmann_weighted_states(nStates, σ=σ_prior, domain=boltzmann_domain, loc=boltzmann_loc, scale=boltzmann_scale, verbose=verbose)
    np.savetxt(f"{dir}/avg_prior_error.txt", np.array([RMSE]))
    energies,pops = states
    np.savetxt(f"{dir}/energies.txt",energies)
    np.savetxt(f"{dir}/pops.txt",pops)

    perturbed_energies,perturbed_pops = perturbed_states
    np.savetxt(f"{dir}/prior.txt", perturbed_energies)
    np.savetxt(f"{dir}/prior_pops.txt",perturbed_pops)

    biceps.toolbox.mkdir(dir)
    dir = dir+"/NOE_J"
    weights = pops
    biceps.toolbox.mkdir(dir)
    ############################################################################
    if draw_from[0] == "uniform": x = np.random.uniform(low=1.0, high=10.0, size=(len(weights),Nd))
    if draw_from[0] == "normal": x = np.random.normal(loc=4.0, scale=1.0, size=(len(weights),Nd))
    Nd1 = int(Nd/2)
    if draw_from[0] == "multi": x = np.concatenate([np.random.normal(loc=3.0, scale=1.0, size=(len(weights),Nd1)),np.random.normal(loc=6.0, scale=1.0, size=(len(weights),Nd1))], axis=1)
    exp = np.array([w*x[i] for i,w in enumerate(weights)]).sum(axis=0)
    true = exp.copy()

    if (μ_data or σ_data) != 0.0:
        print(f"\n(μ_data, σ_data) = ({μ_data}, {σ_data})\n")
        if σ_data > 0.0:
            #exp += σ_data*np.random.randn(len(exp))
            exp = exp+σ_data*np.random.random(len(exp))
        if μ_data > 0.0:
            offset = np.random.uniform(3.0, 5.0, len(exp))
            for i in range(len(exp)):
                if np.random.random() <= 0.30:
                    exp[i] = exp[i] + offset[i]

    for i in range(len(weights)):
        model = pd.read_pickle("template.noe")
        model["true"] = np.nan
        _model = pd.DataFrame()
        for j in range(Nd):
            model["restraint_index"], model["model"], model["exp"], model["true"] = j+1, x[i,j], exp[j], true[j]
            _model = _model.append(model, ignore_index=True)
        _model.to_pickle(dir+"/%s.noe"%i)

    if len(draw_from) == 2:
        # Same thing for the other observable...
        if draw_from[1] == "uniform": y = np.random.uniform(low=1.0, high=10.0, size=(len(weights),Nd))
        if draw_from[1] == "normal": y = np.random.normal(loc=2.0, scale=1.0, size=(len(weights),Nd))
        exp = np.array([w*y[i] for i,w in enumerate(weights)]).sum(axis=0)
        true = exp.copy()

        if (μ_data or σ_data) != 0.0:
            print(f"\n(μ_data, σ_data) = ({μ_data}, {σ_data})\n")
            if σ_data > 0.0:
                #exp += σ_data*np.random.randn(len(exp))
                exp = exp+σ_data*np.random.random(len(exp))
            if μ_data > 0.0:
                offset = np.random.uniform(3.0, 5.0, len(exp))
                for i in range(len(exp)):
                    if np.random.random() <= 0.30:
                        exp[i] = exp[i] + offset[i]

        for i in range(len(weights)):
            model = pd.read_pickle("template.J")
            model["true"] = np.nan
            _model = pd.DataFrame()
            for j in range(Nd):
                model["restraint_index"], model["model"], model["exp"], model["true"] = j+1, y[i,j], exp[j], true[j]
                _model = _model.append(model, ignore_index=True)
            _model.to_pickle(dir+"/%s.J"%i)


def synthetic_data(dir, nStates, Nd, σ_prior=0, μ_prior=0, σ_data=0, μ_data=0,
        boltzmann_domain=(4, 6), boltzmann_loc=3.15, boltzmann_scale=2.0,
        draw_from=["uniform","uniform"], verbose=False):
    """Synthetic input files for biceps including the
    experimental NOE data weighted by all the states. To mimic random errors,
    Gaussian noise can be added by tuning the parameter σ. To mimic systematic
    errors, a shift in the data can be achieved by changing μ.
    """

    biceps.toolbox.mkdir(dir)
    states,perturbed_states,RMSE = get_boltzmann_weighted_states(nStates, σ=σ_prior, domain=boltzmann_domain, loc=boltzmann_loc, scale=boltzmann_scale, verbose=verbose)
    np.savetxt(f"{dir}/avg_prior_error.txt", np.array([RMSE]))
    energies,pops = states
    np.savetxt(f"{dir}/energies.txt",energies)
    np.savetxt(f"{dir}/pops.txt",pops)

    perturbed_energies,perturbed_pops = perturbed_states
    np.savetxt(f"{dir}/prior.txt", perturbed_energies)
    np.savetxt(f"{dir}/prior_pops.txt",perturbed_pops)

    biceps.toolbox.mkdir(dir)
    dir = dir+"/NOE_J"
    weights = pops
    biceps.toolbox.mkdir(dir)
    ############################################################################
    x = np.array([np.random.normal(loc=3.0, scale=1.0, size=Nd),
                  np.random.normal(loc=4.5, scale=1.0, size=Nd),
                  np.random.normal(loc=6.0, scale=1.0, size=Nd)])

#    x = np.array([np.random.normal(loc=3.0, scale=0.35, size=Nd),
#                  np.random.normal(loc=4.5, scale=0.35, size=Nd),
#                  np.random.normal(loc=6.0, scale=0.35, size=Nd)])

    exp = np.array([w*x[i] for i,w in enumerate(weights)]).sum(axis=0)

#    _pops = pops.copy()
#    _pops = np.sort(_pops)
#    indices = [np.where(pops == _pops[i])[0][0] for i in range(len(pops))]
#    #exit()
#
#    exp = []
#    for i in range(Nd):
#        rand = np.random.rand()
#        if rand <= _pops[0]:
#            exp.append(x[indices[0],i])
#            continue
#        elif rand <= _pops[1]:
#            exp.append(x[indices[1],i])
#            continue
#        else:
#            exp.append(x[indices[2],i])
#            continue
#    exp = np.array(exp)
    true = exp.copy()

    if (μ_data or σ_data) != 0.0:
        print(f"\n(μ_data, σ_data) = ({μ_data}, {σ_data})\n")
        if σ_data > 0.0:
            #exp += σ_data*np.random.randn(len(exp))
            exp = exp+σ_data*np.random.random(len(exp))
        if μ_data > 0.0:
            offset = np.random.uniform(3.0, 5.0, len(exp))
            for i in range(len(exp)):
                if np.random.random() <= 0.30:
                #if np.random.random() <= 0.20:
                #if np.random.random() <= 0.05:
                    #exp[i] = exp[i] + offset[i]
                    exp[i] = np.abs(exp[i] + offset[i])


    exp_sigma = np.array([np.std(true)])
    print(exp_sigma)


    for i in range(len(weights)):
        model = pd.read_pickle("template.noe")
        model["true"] = np.nan
        _model = pd.DataFrame()
        for j in range(Nd):
            model["restraint_index"], model["model"], model["exp"], model["true"] = j+1, x[i,j], exp[j], true[j]
            _model = _model.append(model, ignore_index=True)
        _model.to_pickle(dir+"/%s.noe"%i)



#:}}}

# Append to Database:{{{
def append_to_database(A, dbName="database_Nd.pkl", verbose=False, **kwargs):
    n_lambdas = A.K
    pops = A.P_dP[:,n_lambdas-1]
    BS = A.f_df
    populations = kwargs.get("populations")
    RMSE = np.sqrt(metrics.mean_squared_error(pops, populations))
    if verbose: print(f"\n\nRMSE = {RMSE}")

    data = pd.DataFrame()
    data["nsteps"] = [kwargs.get("nsteps")]
    data["nstates"] = [kwargs.get("nStates")]
    data["nlambda"] = [kwargs.get("n_lambdas")]
    data["nreplica"] = [kwargs.get("nreplicas")]
    data["lambda_swap_every"] = [kwargs.get("lambda_swap_every")]
    data["Nd"] = [kwargs.get("Nd")]
    data["prior error"] = [kwargs.get("σ_prior")]
    if (kwargs.get("σ_data")==0.0) and (kwargs.get("μ_data")==0.0): data["data error type"] = ["None"]
    if (kwargs.get("σ_data")>0.0) and (kwargs.get("μ_data")==0.0): data["data error type"] = ["Random"]
    if (kwargs.get("σ_data")==0.0) and (kwargs.get("μ_data")>0.0): data["data error type"] = ["Systematic"]
    if (kwargs.get("σ_data")>0.0) and (kwargs.get("μ_data")>0.0): data["data error type"] = ["Random & Systematic"]
    data["RMSE"] = [RMSE]
    data["uncertainties"] = [kwargs.get("data_uncertainty")]
    data["stat_model"] = [kwargs.get("stat_model")]

    for i,lam in enumerate(kwargs.get("lambda_values")):
        model = A.get_model_scores(model=i)
        data["BIC%0.2g"%lam] = [model["BIC"]]
    #data["BIC score"] = [-0.5*(BIC1-BIC0)]

    for i,lam in enumerate(kwargs.get("lambda_values")):
        lam = "%0.2g"%lam
        data["BICePs Score lam=%s"%lam] = [BS[i,0]]
        data["BICePs Score Std lam=%s "%lam] = [2*BS[i,1]] # at 95% C

    data["prior pops"] = [np.loadtxt(f"{kwargs.get('dir')}/pops.txt")]
    data["pops"] = [pops]
    data["D_KL"] = [np.nansum([pops[i]*np.log(pops[i]/populations[i]) for i in range(len(pops))])]
    data["RMSE"] = [np.sqrt(metrics.mean_squared_error(pops, populations))]
    data["k"] = [kwargs.get("k")]

    data["priors"] = [np.loadtxt(f"{kwargs.get('dir')}/prior.txt")]
    data["avg prior error"] = [np.loadtxt(f"{kwargs.get('dir')}/avg_prior_error.txt")]
    # NOTE: Saving results to database
    if os.path.isfile(dbName):
       db = pd.read_pickle(dbName)
    else:
        if verbose: print("Database not found...\nCreating database...")
        db = pd.DataFrame()
        db.to_pickle(dbName)
    # append to the database
    db = db.append(data, ignore_index=True)
    db.to_pickle(dbName)
    gc.collect()

# }}}

# Plot 1D posterior with table:{{{

def plot_1D_posterior_with_table(results, figname="fig.png"):

    from matplotlib import gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels
    import seaborn as sns
    plt.close()


    xmin,xmax = (np.min(x_0)-0.1,np.max(x_0)+0.1)
    nbins = 100
    binsx = np.linspace(xmin,xmax,nbins)
    #nbins = 50
#    hist,bin_edges = np.histogram(x_true, bins=nbins)
    hh2r, ee2r = np.histogram(x_true, bins=binsx, density=True)

    ymin,ymax= (0,np.max(hh2r)+0.5)
    binsy = np.linspace(xmin,xmax,nbins)
    bins2d = [np.linspace(xmin,xmax,60),np.linspace(ymin,ymax,60)]
    lw=1.6

    # here define boundaries for states
    xb=5
    yb=4
    idx_yl = np.where(bins2d[1]<yb)[0]
    idx_xl = np.where(bins2d[0]<xb)[0]
    idx_yu = np.where(bins2d[1]>=yb)[0][:-1]
    idx_xu = np.where(bins2d[0]>=xb)[0][:-1]

    label_fontsize = 16
    # start with a rectangular Figure
    figsize = (10,14)
    fig = plt.figure(1, figsize=figsize)
    grid = plt.GridSpec(12, 14, hspace=0.25, wspace=0.01)
    axTable = fig.add_subplot(grid[:4, :])
    axScatter = fig.add_subplot(grid[8:10, 7:])
    axRef = fig.add_subplot(grid[4:6, 7:], sharex=axScatter)
    axHistx = fig.add_subplot(grid[6:8, 7:], sharex=axScatter)
    axHisty = fig.add_subplot(grid[10:12, 7:], sharex=axScatter)

#    divider = make_axes_locatable(axScatter)
#    axHistx = divider.append_axes("top", size="100%", pad=0.05, sharex=axScatter)
#    axHisty = divider.append_axes("bottom", size="100%", pad=0.05, sharex=axScatter)

    # NOTE: adding posterior distributions
    #axMethod1 = fig.add_subplot(grid[1:4, 10:])
    #axMethod2 = fig.add_subplot(grid[4:7, 10:], sharex=axMethod1)
    #axMethod3 = fig.add_subplot(grid[7:, 10:], sharex=axMethod1)

    axMethod1 = fig.add_subplot(grid[6:8, 2:6])
    axMethod2 = fig.add_subplot(grid[8:10, 2:6], sharex=axMethod1)
    axMethod3 = fig.add_subplot(grid[10:12, 2:6], sharex=axMethod1)

    axRef.tick_params(labelbottom=0, labeltop=False, labelleft=False, labelright=0,
                         bottom=0, top=False, left=False, right=0)

    axScatter.tick_params(labelbottom=0, labeltop=False, labelleft=False, labelright=0,
                         bottom=0, top=False, left=False, right=0)
    axHistx.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                         bottom=False, top=False, left=False, right=False)
    axHisty.tick_params(labelbottom=1, labeltop=False, labelleft=False, labelright=False,
                         bottom=1, top=False, left=False, right=False)

    #axScatter.yaxis.set_label_position("right")

    axMethod1.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                          bottom=0, top=False, left=True, right=False)
    axMethod2.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                          bottom=0, top=True, left=True, right=False)
    axMethod3.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                          bottom=True, top=True, left=True, right=False)

    axHistx.yaxis.set_label_position("right")

    # no labels
    axTable.xaxis.set_major_formatter(nullfmt)
    axTable.yaxis.set_major_formatter(nullfmt)

    # set limits and levels
    axHistx.set_ylim(ymin, ymax)
    axHistx.set_xlim(xmin, xmax)
    axHisty.set_ylim(ymin, ymax)
    axHisty.set_xlim(xmin, xmax)
    axScatter.set_ylim(ymin, ymax)
    axScatter.set_xlim(xmin, xmax)
    axRef.set_ylim(ymin, ymax)
    axRef.set_xlim(xmin, xmax)

    cols1 = sns.color_palette("Greys")
    cols2 = sns.color_palette("YlOrRd")
    cols3 = sns.color_palette("Blues")
    cols4 = sns.color_palette("Greens")
    _colors_ = ["g", "brown", "r", "m", 'y', 'c']
    true_color = "k"
    #prior_color = cols1[1] #_colors_[-1]
    prior_color = "orange" #cols1[1] #_colors_[-1]
    exp_color = "m"

    # plot reference distribution
    hh2r, ee2r = np.histogram(x_true, bins=binsx, density=True)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    hh2r, ee2r = np.histogram(x_true, bins=binsx, density=True)
    axHistx.step(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, lw=lw, c='k', label="Sample")

    hh2r, ee2r = np.histogram(x_true, bins=binsx, density=True)
    axHisty.step(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, lw=lw, c='k', label="Sample")

    hh2r, ee2r = np.histogram(x_true, bins=binsx, density=True)
    axScatter.step(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, lw=lw, c='k', label="Sample")

    ref_colors = ['blue', 'darkorange', 'darkcyan']
    hh2r, ee2r = np.histogram(x1["model"].to_numpy(), bins=binsx, density=True)
    axRef.step(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, lw=lw, c=ref_colors[0], label="Sample")
    axRef.fill_between(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, step='mid',
                           color=prior_color, label="Sample", alpha=0.5)

    hh2r, ee2r = np.histogram(x2["model"].to_numpy(), bins=binsx, density=True)
    axRef.step(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, lw=lw, c=ref_colors[1], label="Sample")
    axRef.fill_between(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, step='mid',
                           color=prior_color, label="Sample", alpha=0.5)

    hh2r, ee2r = np.histogram(x3["model"].to_numpy(), bins=binsx, density=True)
    axRef.step(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, lw=lw, c=ref_colors[2], label="Sample")
    axRef.fill_between(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, step='mid',
                           color=prior_color, label="Sample", alpha=0.5)

    axRef.axvline(x1["model"].to_numpy().mean(), color=ref_colors[0], linewidth=3, linestyle='--')
    t = axRef.annotate(r'$S_{0}$', (x1["model"].to_numpy().mean()+0.1, 0.95), size=16, color=ref_colors[0])
    axRef.axvline(x2["model"].to_numpy().mean(), color=ref_colors[1], linewidth=3, linestyle='--')
    t = axRef.annotate(r'$S_{1}$', (x2["model"].to_numpy().mean()+0.1, 0.95), size=16, color=ref_colors[1])
    axRef.axvline(x3["model"].to_numpy().mean(), color=ref_colors[2], linewidth=3, linestyle='--')
    t = axRef.annotate(r'$S_{2}$', (x3["model"].to_numpy().mean()+0.1, 0.95), size=16, color=ref_colors[2])

    #hh2r, ee2r = np.histogram(y_true, bins=binsy, density=True)
    # NOTE: avg of True
    axHistx.axvline(avgx_true, color=true_color, linewidth=4)
    axHisty.axvline(avgx_true, color=true_color, linewidth=4)
    axScatter.axvline(avgx_true, color=true_color, linewidth=4)

    # Now plot P0 NOTE: The Prior
    #hh, xe, ye = np.histogram2d(x_0, y_0, bins=bins2d, density=True)
    hh2, ee2 = np.histogram(x_0, bins=binsx, density=True)

    axHistx.axvline(avgx_exp, color=exp_color, linewidth=4)
    axHisty.axvline(avgx_exp, color=exp_color, linewidth=4)
    axScatter.axvline(avgx_exp, color=exp_color, linewidth=4)

    axHisty.set_xlabel("x", fontsize=16)

    RMSE = np.sqrt(metrics.mean_squared_error(prior_pops, true_pops))

   # Table information
    rows = ["","True","","Exp", "", 'Prior']
    columns= ["<x>",r"$S_{0}$ (%)",r"$S_{1}$ (%)",r"$S_{2}$ (%)", "RMSE"]
    empty_row = ["" for c in range(len(columns))]

    cell_text = [empty_row,
            ["%4.1f" % avgx_true, "%4.0f" % (true_pops[0]*100.),"%4.0f" % (true_pops[1]*100.),"%4.0f" % (true_pops[2]*100.), ""],\
            empty_row,
            ["%4.1f" % avgx_exp, "" ,"","", ""],\
            empty_row,
            ["%4.1f" % avgx_0, "%4.0f" % (prior_pops[0]*100.),"%4.0f" % (prior_pops[1]*100.),"%4.0f" % (prior_pops[2]*100.), "%0.3f" % RMSE],\
            ]
    colors = [None,true_color,None,exp_color,None,prior_color,] #cols3[-4],"#ffffff",cols4[-4]]

    minima = np.min([np.min(results.iloc[[i]]["pops"].to_numpy()[0][results.iloc[[i]]["pops"].to_numpy()[0].nonzero()]) for i,row in enumerate(results.index.to_list())])
    minima0 = np.min([np.min(results.iloc[[i]]["pops"].to_numpy()[0][results.iloc[[i]]["pops0"].to_numpy()[0].nonzero()]) for i,row in enumerate(results.index.to_list())])
    _min_ = [minima0, minima]
    _min_ = np.min(_min_)
    print(_min_)


    methodax = [axMethod1, axMethod2, axMethod3]

    histax = [axHistx, axScatter, axHisty]


    # Plot BICePs hist and append data information
    for i,row in enumerate(results.index.to_list()):
        rows.append("")
        cell_text.append(empty_row)
        colors.append(None)
        color = _colors_[i]
        row = results.iloc[[i]]
        reweighted_x, reweighted_y = row["reweighted_x"].to_numpy()[0], row["reweighted_y"].to_numpy()[0]
        #print(reweighted_x, reweighted_y)
        method = row["method"].to_numpy()[0]
        print(method)
        biceps_avg_post, pops = row["biceps_avg_post"].to_numpy()[0], row["pops"].to_numpy()[0]
        print(biceps_avg_post, pops)

        RMSE = np.sqrt(metrics.mean_squared_error(pops, true_pops))
        try:
            hh2r, ee2r = np.histogram(reweighted_x, bins=binsx, density=True)
            histax[i].step(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, lw=lw, c='k', label="Sample")
            histax[i].fill_between(0.5*(ee2r[1:]+ee2r[:-1]), hh2r, step='mid',
                                   color=color, label="Sample", alpha=0.5)
        except(Exception) as e:
            break

        histax[i].axvline(biceps_avg_post[0], color=color, linewidth=3.5)

        rows.append(method)

        cell_text.append(["%4.1f" % biceps_avg_post[0], "%4.0f" % (pops[0]*100.),"%4.0f" % (pops[1]*100.), "%4.0f" % (pops[2]*100.), "%0.3f" % RMSE])
        colors.append(color)


        # NOTE: plot populations
        methodax[i].errorbar(x=row["pops0"].to_numpy()[0], y=row["pops"].to_numpy()[0],
                             xerr=row["dpops0"].to_numpy()[0], yerr=row["dpops"].to_numpy()[0],
                             fmt="k.", ms=15
                             )

        try:
            val = int(str("%0.5e"%_min_).split("e-")[-1])
            limit = np.round(_min_, val) #- 1*10**-val
        except(Exception) as e:
            limit = _min_
        methodax[i].plot([limit, 1], [limit, 1], color='k', linestyle='-', linewidth=2.5)
        methodax[i].set_xlim(limit, 1.)
        methodax[i].set_ylim(limit, 1.)
        #if i < len(results.index.to_list()):
        methodax[-1].set_xlabel('$p_i$ (exp)', fontsize=label_fontsize)
        methodax[i].set_ylabel('$p_i$ (sim+exp)', fontsize=label_fontsize)
        methodax[i].set_xscale('log')
        methodax[i].set_yscale('log')

        #t = methodax[i].annotate(method, (limit, 0.1), size=16)#, color=color)
        t = methodax[i].annotate(method, (limit*2, 2e-1), size=16)#, color=color)
        t.set_bbox(dict(facecolor=color, alpha=0.5))

        #label key states
        for s in range(len(row["pops"].to_numpy()[0])):
            if row["pops"].to_numpy()[0][s] == 0.0:
                continue
            methodax[i].text( row["pops0"].to_numpy()[0][s],
                             row["pops"].to_numpy()[0][s],
                             # TODO: color of state labels
                             str(s), color=ref_colors[s],
                             fontsize=18) #label_fontsize)

    rowColours=colors
    #cellColours = [[rowColours[row]]+["w" for c in range(len(columns)-1)] for row in range(len(rowColours))]
    the_table = axTable.table(cellText=cell_text,\
                             rowLabels=rows,\
                             colLabels=columns,\
                             loc='best', cellLoc='center',
                             rowLoc='center', rowColours=rowColours,
                             #cellColours=cellColours,
                             colWidths=[0.14 for c in columns],
                             in_layout=True, zorder=1)#, edges="open")
    the_table.zorder=1
    #the_table.AXESPAD = 0.06
    #the_table.PAD = 0.15
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)
    axTable.axis('off')
    # iterate through cells of a table
    table_props = the_table.properties()
    table_cells = table_props['celld']
    # NOTE: API for table
    # https://matplotlib.org/stable/api/table_api.html

#    print(table_cells)
 #   table_cells = the_table.get_celld()
 #   print(table_cells)

#    for row in range(len(rows)):
#        if row%2 == 0:
#            the_table[(row,0)].set_facecolor(rowColours[row])




    _colors_ = [c for c in colors if c is not None]
    alphas = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9]
    c = 0
    for cell in table_cells.items():
        cell[1].set_edgecolor('none')
        cell[1].set_facecolor('white')
        cell[1].set_height(.071)
        if cell[0][-1] == -1:
            if cell[0][0]%2 == 0:
                cell[1].set_edgecolor('black')
                #help(cell[1])
                #exit()
                #print(cell[1].get_text())
                #cell[1].set_text_props(color=_colors_[c])
                the_color = (*matplotlib.colors.to_rgb(_colors_[c]), alphas[c])
                if (_colors_[c] == "k") or (_colors_[c] == "black"):
                    #cell[1].set_text_props(backgroundcolor='black', color="white")
                    cell[1].set_text_props(color="white")
                    cell[1].set_facecolor('black')
                    c += 1
                    continue
                #cell[1].set_text_props(backgroundcolor=the_color)
                cell[1].set_facecolor(the_color)
                c += 1

    the_table.get_celld()[(0,1)].set_text_props(color=ref_colors[0])
    the_table.get_celld()[(0,2)].set_text_props(color=ref_colors[1])
    the_table.get_celld()[(0,3)].set_text_props(color=ref_colors[2])


    #axs = [axTable, axHistx, axMethod1, axMethod2, axMethod3, axHistx, axHisty]
    axs = [axTable, axMethod1, axMethod2, axMethod3, axRef, axHistx, axScatter,  axHisty,  ]
    for n, ax in enumerate(axs):
    #for ax in axs:
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks(),
                 ]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels(),
                ]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                mark.set_rotation(s=10)
        #if n > 4: continue
        #else:
        x,y = -0.35, 1.02
        #if n == 0:  x,y = -0.10, 1.02
        #if n == 1:  x,y = -0.11, 1.02
        if n >= 4:  x,y = -0.1, 1.02
        if n == 0:  x,y = 0.055, 1.01

        ax.text(x,y, string.ascii_lowercase[n], transform=ax.transAxes,
                size=20, weight='bold', zorder=999)


    plt.tight_layout()
    fig.savefig(figname, dpi=600)
    plt.close()









#:}}}

# BICePs_Specs:{{{
################################################################################
####### Parameters #######
nStates,Nd = 3,500 # use 1000 or 5000 for production
n_lambdas,nreplicas,nsteps,change_Nr_every,swap_every=2,6,100000,0,0

#σ_prior, σ_data, μ_data = 0.16, 0.0, 0.0
σ_prior, σ_data, μ_data = 0.08, 0.5, 4.0

stat_model, data_uncertainty = "Students", "single"
#stat_model, data_uncertainty = "GB", "single"
_stat_model = stat_model
_data_uncertainty = data_uncertainty

burn = 0
scale_energies = 0
find_optimal_nreplicas = 0
multiprocess=1
swap_sigmas=1

write_every=10
#write_every=1

verbose = False
if verbose:
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

attempt_move_state_every = 1
attempt_move_sigma_every = 1
scale_and_offset = 0
move_ftilde_every = 0
dftilde = 1.10 #1.0 #0.1
ftilde_sigma = 1.0 #2.0 #1.0

walk_in_all_dim=False
plottype="step"
##########################
state_dir = f"toc-{nStates}-state-toy"
biceps.toolbox.mkdir(state_dir)
dir = f"{state_dir}/{nStates}_state_{Nd}_datapoints/Prior_error_{σ_prior}"

boltzmann_scale=1.
boltzmann_domain=(0.12,0.6)
boltzmann_loc=0.0
draw_from = ["uniform"]

draw_new_samples = 1
if draw_new_samples:
    synthetic_data(dir, nStates, Nd, σ_prior=σ_prior, σ_data=σ_data, μ_data=μ_data,
            boltzmann_scale=boltzmann_scale, boltzmann_domain=boltzmann_domain,
            boltzmann_loc=boltzmann_loc, draw_from=draw_from, verbose=True)



#exit()
populations = np.loadtxt(f"{dir}/pops.txt")
prior_pops = np.loadtxt(f"{dir}/prior_pops.txt")
energies = np.loadtxt(f"{dir}/prior.txt")
energies -= energies.min()

data_dir = dir+"/NOE_J"
input_data = biceps.toolbox.sort_data(data_dir)
print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
model = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.noe")])
ax2 = model["model"].plot.hist(alpha=0.5, bins=30, edgecolor='black', linewidth=1.2, density=1, color="b", figsize=(14, 6), label="data1", ax=None)
model["exp"].plot.hist(bins=20, alpha=0.5, ax=ax2, color="orange", density=True)
fig2 = ax2.get_figure()
fig2.savefig(f"{dir}/synthetic_experimental_NOEs.png")
fig2.clear()
plt.close()

outdir = f'{dir}/{nsteps}_steps_{nreplicas}_replica_{n_lambdas}_lam'
biceps.toolbox.mkdir(outdir)
print(f"nSteps of sampling: {nsteps}\nnReplicas: {nreplicas}")
lambda_values = np.linspace(0.0, 1.0, n_lambdas)

sigMin, sigMax, dsig = 0.001, 200, 1.01
arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
l = len(arr)
sigma_index = round(l*0.73)



if (stat_model == "Students"):
    beta,beta_index=(1., 100.0, 10000),0
    _arr = np.linspace(*beta)
    _l = len(_arr)
    print("Alpha starts here: ",_arr[beta_index])
    phi,phi_index=(1., 2.0, 1),0
    gamma,gamma_index=(1.0, 2.0, np.e),0
elif (stat_model == "GB"):
    # NOTE: acting as gamma for Good-and-Bad
    #alpha=(1., 2, 1000)
    beta,beta_index=(1., 2.0, 1),0
    phi,phi_index=(1., 100.0, 10000),500
    #phi,phi_index=(1., 2.0, 1),0
    gamma,gamma_index=(1.0, 2.0, np.e),0
    _arr = np.linspace(*phi)
    _l = len(_arr)
    print("Alpha starts here: ",_arr[phi_index])
else:
    beta,beta_index=(1., 2.0, 1),0
    phi,phi_index=(1., 2.0, 1),0
    gamma,gamma_index=(1.0, 2.0, np.e),0


parameters = [
    dict(ref="uniform", sigma=(sigMin, sigMax, dsig),
    beta_index=beta_index, beta=beta,
    phi_index=phi_index, phi=phi,
    data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=sigma_index),
    ]
if len(draw_from) == 2:
    parameters = [
        dict(ref="uniform", sigma=(sigMin, sigMax, dsig),
        data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=sigma_index),
        #dict(ref="uniform", sigma=(sigMin, sigMax, dsig), gamma=(1.0, 2.0, np.e),
        #data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=sigma_index),

        dict(ref="uniform", sigma=(sigMin, sigMax, dsig), gamma=(1.0, 2.0, np.e),
            data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=sigma_index)
        ]

print(pd.DataFrame(parameters))
print("len = ",l)
print("simga = ",arr[sigma_index])


#print(np.linspace(*phi)[phi_index])
#exit()



# }}}

# Run BICePs:{{{
def run_biceps(data, prior, parameters, lambda_values, method):

    ensemble = biceps.ExpandedEnsemble(energies=prior, lambda_values=lambda_values)
    ensemble.initialize_restraints(data, parameters)
    sampler = biceps.PosteriorSampler(ensemble, nreplicas, write_every=write_every)
    sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
            attempt_move_state_every=attempt_move_state_every,
            attempt_move_sigma_every=attempt_move_sigma_every,
            burn=0, print_freq=1, walk_in_all_dim=walk_in_all_dim,
            verbose=0, progress=1, multiprocess=multiprocess, capture_stdout=0)
    print(sampler.acceptance_info)
    A = biceps.Analysis(sampler, outdir=outdir, nstates=len(prior), MBAR=True,
            scale_energies=scale_energies)

    plt.close()

    BS, pops = A.f_df, A.P_dP[:,n_lambdas-1]
    pops0 = A.P_dP[:,0]
    dpops0, dpops = A.P_dP[:,n_lambdas], A.P_dP[:,2*n_lambdas-1]
    RMSE = np.sqrt(metrics.mean_squared_error(pops, populations))
    print(f"Unperturbed (actual) populations: {populations}")
    print(f"Predicted populations:            {pops}")
    print(f"RMSE = {RMSE}")
    print(f"BICePs Score = {BS[1,0]}")
    plt.close()
    A.plot(plottype, figname=f"{method}.pdf")
    plt.close()
    #exit()

    ############################################################################
    #print(data)
    data_x = [pd.read_pickle(file[0]) for file in data]
    data_x = np.array([df["model"].to_numpy() for df in data_x])
    #print(data_x)
    reweighted_x = np.array([w*data_x[i] for i,w in enumerate(pops)]).sum(axis=0)

    biceps_avg_post = [np.average(reweighted_x)]
    reweighted_y = np.nan
    if len(draw_from) == 2:
        reweighted_y = np.array([w*y[i] for i,w in enumerate(pops)]).sum(axis=0)
        biceps_avg_post = [np.average(reweighted_x), np.average(reweighted_y)]
    print(biceps_avg_post)

    results = {"pops":pops, "dpops":dpops, "pops0":pops0, "dpops0":dpops0,
               "reweighted_x":reweighted_x,
               "reweighted_y":reweighted_y,
               "biceps_avg_post":biceps_avg_post}

    return results
# }}}



if __name__ == "__main__":

    ###########################################################################
    x1 = pd.read_pickle(f"{data_dir}/0.noe")
    x2 = pd.read_pickle(f"{data_dir}/1.noe")
    x3 = pd.read_pickle(f"{data_dir}/2.noe")
    x = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.noe")])
    x_true = x["true"].to_numpy()
    x_exp = x["exp"].to_numpy()
    x_0 = x["model"].to_numpy()

    # Calculate the average position in Ptrue and P0
    avgx_true = np.average(x_true) # the model data is untouched and is the true answer (mixture of states)
    avgx_exp = np.average(x_exp)
    avgx_0 = np.average(x_0)
    #print(avgx_exp)

    x = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.noe")]
    x = [i["model"].to_numpy() for i in x]

    exp_sigma = np.array([np.sqrt(metrics.mean_squared_error(x_true, x_exp))])
    #print(exp_sigma)
    #exp_sigma = np.array([np.std(x_true)])
    #print(exp_sigma)
    #exit()

    if len(draw_from) == 2:
        y1 = pd.read_pickle(f"{data_dir}/0.J")
        y2 = pd.read_pickle(f"{data_dir}/1.J")
        y3 = pd.read_pickle(f"{data_dir}/2.J")
        y = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.J")])
        y_true = y["true"].to_numpy()
        y_exp = y["exp"].to_numpy()
        y_0 = y["model"].to_numpy()
        avgy_true = np.average(y_0)
        # get experimental uncetainty
        avgy_exp = np.average(y_exp)
        avgy_0 = np.average(y_0)
        y = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.J")]
        y = [i["model"].to_numpy() for i in y]
        exp_sigma = np.array([np.std(x_exp), np.std(y_exp)])


    true_pops = populations.copy()
    prior = energies
    prior_pops = np.loadtxt(f"{dir}/prior_pops.txt")
    ###########################################################################

    df = []

    ############################################################################
    # NOTE: multiple replicas - Special likelihood with 1 uncertainty parameter (BICePs v3.0)
    ############################################################################
    method = "BICePs"
    results = run_biceps(input_data, prior, parameters, lambda_values, method)
    results["method"] = method
    df.append(results)

    ############################################################################
    # NOTE: Single replica - Gaussian likelihood with 1 uncertainty parameter (BICePs v2.0)
    ############################################################################
    nreplicas = 1
    find_optimal_nreplicas = 0
    stat_model, data_uncertainty = "GB", "single"
    parameters = [
        dict(ref="uniform", sigma=(sigMin, sigMax, dsig),
        data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=sigma_index),
    ]
    if len(draw_from) == 2:
        parameters = [
            dict(ref="uniform", sigma=(sigMin, sigMax, dsig),
            data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=sigma_index),

            dict(ref="uniform", sigma=(sigMin, sigMax, dsig), gamma=(1.0, 2.0, np.e),
                data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=sigma_index)
        ]

    method = "Single replica" #stat_model
    results = run_biceps(input_data, prior, parameters, lambda_values, method)
    results["method"] = method
    df.append(results)
    ############################################################################
    # NOTE: multiple replicas - Gaussian likelihood with fixed sigma (no uncertainty parameter)
    ############################################################################
    nreplicas = 6
    stat_model, data_uncertainty = "GB", _data_uncertainty
    print(stat_model)
    print(data_uncertainty)

    parameters = [
        dict(ref="uniform", sigma=(exp_sigma[0], exp_sigma[0]+1, np.e),
        data_uncertainty=data_uncertainty, stat_model=stat_model, sigma_index=0),
        ]

    method = "Fixed sigma"
    results = run_biceps(input_data, prior, parameters, lambda_values, method)
    results["method"] = method
    df.append(results)
#    df = pd.DataFrame(df)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    df = pd.DataFrame(df)

    print(df)
    df.to_csv(f"{outdir}/results.csv")

    plot_1D_posterior_with_table(df, figname=f"{outdir}/1D_posterior.pdf")











