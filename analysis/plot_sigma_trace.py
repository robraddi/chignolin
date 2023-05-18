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


def get_convergence(traj, method="auto", maxtau=5000, outdir="./"):
    C = biceps.Convergence(traj, outdir=outdir)
    #traces = C.get_traces()
    C.plot_traces(figname="traces.pdf", figsize=(16,10))
    C.get_autocorrelation_curves(method=method, maxtau=maxtau, nblocks=5)
    #print(C.labels)
    #print(C.tau_c)

    C.process(nblock=5, nfold=10, nround=100, savefile=True, block_avg=True, normalize=True)
    #try:
    #    C.process(nblock=5, nfold=10, nround=100, savefile=True, block_avg=True, normalize=True)
    #except(Exception) as e:
    #    pass
    #exit()



# get_confidence_ellipse:{{{
def get_confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none',
        verbose=False, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    if verbose==True:
        print(f"mean loc: ({mean_x}, {mean_y})")
        print(f"Cov: ({cov})")

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
#:}}}

# corner_plot:{{{
def corner_plot_test(data, scatter_colors="k", bins=25,
        figname="corner_plot.png",figsize=(16,16), verbose=False):
    from pandas.plotting import scatter_matrix
    import matplotlib.pylab as pylab

    fig1, (ax) = plt.subplots(1)
    ax.xaxis.label.set_rotation(0)
    ax.yaxis.label.set_rotation(0)

    axs = scatter_matrix(data, alpha=0.2, figsize=figsize, diagonal='hist', c=scatter_colors,
            hist_kwds=dict(facecolor="cornflowerblue", edgecolor="k", linewidth=1.1, bins=bins), ax=ax)
    n = len(data.columns)
    ds = data.to_numpy().transpose()
    for ax_x in range(n):
        for ax_y in range(n):
            # to get the axis of subplots
            ax = axs[ax_x, ax_y]
            xlabel,ylabel = list(data.keys())[ax_x],list(data.keys())[ax_y]
            label = '''%s
%s'''%(xlabel, ylabel)
            #ax.legend([label], fontsize=4, markerscale=0)
            # NOTE: fix ∆H axis limits

            if ax_y == 6: ax.set_xlim(right=2400)
            if ax_y == 8: ax.set_xlim(0.4, 0.8)
            if ax_y == 9: ax.set_xlim(-4, 20.0)

#            # NOTE: Remove all plots in top right corner
#            #if ax_x < ax_y: ax.set_visible(False)
#            if (ax_x > ax_y) or (ax_x < ax_y):
#                #NOTE: the axis need to be switched...
#                y = ds[ax_x] # data[list(data.keys())[ax_x]].to_numpy()
#                x = ds[ax_y] # data[list(data.keys())[ax_y]].to_numpy()
#                if verbose: print(label)
#                get_confidence_ellipse(x, y, ax,n_std=1, edgecolor="red", linestyle='--', verbose=verbose)
#                if verbose: print("\n\n")
#                get_confidence_ellipse(x, y, ax,n_std=2, edgecolor="black", linestyle='--')
#                get_confidence_ellipse(x, y, ax,n_std=3, edgecolor="blue", linestyle='--')

            # to make x axis name vertical
            ax.xaxis.label.set_rotation(45)
            ax.xaxis.label.set_size(8)
            # to make y axis name horizontal
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_size(8)
            # change spacing bewteen axis label and ticklabels
            # to make sure y axis names are outside the plot area
            ax.xaxis.labelpad = 8
            ax.yaxis.labelpad = 25
            # rotation of the xtick labels
            ax.tick_params(axis='x', rotation=45)

            if ax_x == ax_y:
                if ax_x != 0: ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                #ax.spines["top"].set_linewidth(3)
                #ax.spines["right"].set_linewidth(3)
                mean = data[list(data.keys())[ax_y]].mean()
                #print(f"{list(data.keys())[ax_y]} mean = {mean}")
                ax.axvline(x=mean, linewidth=1.2, color='r')

    #fig1.tight_layout(pad=0.1, w_pad=-2, h_pad=-0.5)
    fig1.tight_layout(pad=0.1, w_pad=-0.9, h_pad=-0.5)
    fig1.savefig(figname, dpi=700)
    return fig1
#:}}}

# corner_plot:{{{
def corner_plot(data, scatter_colors="k", scatter_cmap="coolwarm", bins=25, figname="corner_plot.png", figsize=(16, 16), verbose=False):
    from pandas.plotting import scatter_matrix
    import matplotlib.pylab as pylab
    from matplotlib import cm

    fig1, (ax) = plt.subplots(1)
    ax.xaxis.label.set_rotation(0)
    ax.yaxis.label.set_rotation(0)
    if scatter_cmap:
        z = data.index.to_numpy()
        cmap=cm.get_cmap(scatter_cmap)
        norm = plt.Normalize(vmin=np.min(z), vmax=np.max(z))
        c = cmap(norm(z))
    else:
        cmap = None
        c = scatter_colors
    axs = scatter_matrix(data, alpha=0.2, figsize=figsize, diagonal='hist', c=c,
                          hist_kwds=dict(facecolor="cornflowerblue", edgecolor="k",
                                         linewidth=1.1, bins=bins), ax=ax)

    n = len(data.columns)
    ds = data.to_numpy().transpose()
    for ax_x in range(n):
        for ax_y in range(n):
            # to get the axis of subplots
            ax = axs[ax_x, ax_y]
            xlabel,ylabel = list(data.keys())[ax_x],list(data.keys())[ax_y]
            label = '''%s
%s'''%(xlabel, ylabel)

            # to make x axis name vertical
            ax.xaxis.label.set_rotation(45)
            ax.xaxis.label.set_size(8)
            # to make y axis name horizontal
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_size(8)
            # change spacing bewteen axis label and ticklabels
            # to make sure y axis names are outside the plot area
            ax.xaxis.labelpad = 8
            ax.yaxis.labelpad = 25
            # rotation of the xtick labels
            ax.tick_params(axis='x', rotation=45)

            if ax_x == ax_y:
                if ax_x != 0: ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                #ax.spines["top"].set_linewidth(3)
                #ax.spines["right"].set_linewidth(3)
                mean = data[list(data.keys())[ax_y]].mean()
#                ax.axvline(x=mean, linewidth=1.0, color='r')

            # NOTE: condition to hide the top right subplots since they are just repeats

    #fig1.tight_layout(pad=0.1, w_pad=-0.9, h_pad=-0.5)
    fig1.tight_layout()
    fig1.savefig(figname, dpi=700)
    return fig1
# }}}




shift_y = 1
sharey = 1

figsize = (10, 5)
#figsize=(8,4)
label_fontsize=12
legend_fontsize=10
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 2, width_ratios=(4, 1), wspace=0.02)
ax1 = fig.add_subplot(gs[0,0])
if sharey: ax2 = fig.add_subplot(gs[0,1], sharey=ax1)
else: ax2 = fig.add_subplot(gs[0,1])

#ax3 = fig.add_subplot(gs[1,:])

burn = 0  # total samples: 10000
burn = 2000
#burn = 8000

test_burn = 8000

energies = 500
#figures = []
corrections = []

#stat_model,nreplicas = "GaussianGB",8
stat_model,nreplicas = "Bayesian", 1
stat_model,nreplicas = "Students_new",8
stat_model,nreplicas = "Students",8
#stat_model,nreplicas = "Gaussian", 8
data_uncertainty = "single"
#data_uncertainty = "multiple"

outdir = f"*/nclusters_{energies}/{stat_model}_{data_uncertainty}_sigma/10000000_steps_{nreplicas}_replicas_2_lam__swap_every_0_change_Nr_every_0_trial_*/sampler_obj.pkl"
#files = biceps.toolbox.get_files(outdir)[::4]
#files = biceps.toolbox.get_files(outdir)[0:2]
files = biceps.toolbox.get_files(outdir)
_files = []
_FF = ""
for j,file in enumerate(files):
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

#files = ["/Users/rr/github/biceps_chignolin_test/Analysis_v05/CLN001/CHARMM27/nclusters_500/Students_new_single_sigma/10000000_steps_8_replicas_2_lam__swap_every_0_change_Nr_every_0_trial_8/sampler_obj.pkl"]

for j,file in enumerate(files):
    print("\n\n\n",file,"\n\n\n")
    FF = file.split("/")[0]
    FF = FF.replace('AMBER','A').replace('CHARM','C')

    # NOTE: Look at only a single within each force field
    if (j != 0) and (FF == corrections[-1]["FF"]): continue

    label = FF
    outdir = file.split("sampler_obj.pkl")[0]
    sampler = biceps.toolbox.load_object(file)
    A = biceps.Analysis(sampler, outdir=outdir, nstates=energies, MBAR=0, verbose=0)
    A.plot_population_evolution()

    #figs,steps,dists = A.plot_energy_trace()
    df0 = A.get_traces(traj_index=0)
    cols = df0.columns.to_numpy()
    # NOTE: find the parameters that actually underwent sampling
    indices = []
    for k in range(len(cols)):
        df0_array = df0["%s"%(df0.columns.to_list()[k])].to_numpy()
        if all(df0_array == np.ones(df0_array.shape)): continue
        if all(df0_array == np.zeros(df0_array.shape)): continue
        indices.append(k)
    df0 = df0[cols[indices]]
    # NOTE: only looking at the sigmas for right now...
    df0 = df0[[col for col in df0.columns if "sigma" in col]]
    #print(df0)
    #exit()
    #hists0 = self.get_counts_and_bins_for_continuous_space(model=0)
    get_convergence(sampler.traj[0], method="block-avg-auto", maxtau=500, outdir=outdir)
#    exit()
    corner_plot(data=df0, bins="auto", figsize=(16,16), scatter_cmap="viridis", figname=f"{outdir}/corner_plot_test.png", verbose=True)
#    exit()

    corrections.append({"FF":FF})





















