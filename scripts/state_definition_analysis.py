import sys, time, os, gc, string, re
import biceps
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from biceps.toolbox import three2one
from matplotlib import cm



mpl_colors = matplotlib.colors.get_named_colors_mapping()
mpl_colors = list(mpl_colors.values())[::5]
extra_colors = mpl_colors.copy()
#mpl_colors = ["k","lime","b","brown","c","green",
mpl_colors = ["b","brown","c","green",
              "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
              #'#acc2d9', "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
        '#e78ea5', '#983fb2', '#b7e1a1', '#430541', '#507b9c', '#c9d179',
            '#2cfa1f', '#fd8d49', '#b75203', '#b1fc99']+extra_colors[::2]+ ["k","grey"]


# Plots:{{{
#def create_violinplot(df):
#    #fig = plt.figure(1, figsize=(10, 10))
#    #grid = plt.GridSpec(1, 1, hspace=0.01, wspace=0.01)
#    #ax = fig.add_subplot(grid[0, 0])
#    #sns.violinplot(data=grouped, x="model", y="restraint_index", hue="type", kind="violin", inner="stick", ax=ax)
#    sns.set_style("whitegrid")
#    g = sns.catplot(x="restraint_index", y="model", hue="state", col="exp", kind="violin", split=False, data=df, height=4, aspect=1)
#    g.map(sns.scatterplot, "restraint_index", "exp", color="red", s=50, marker="o")
#    g.set_titles(col_template="{col_name}")
#    g.set_axis_labels("", "Model")
#    g.savefig(f"{dir}/box_plot_J.pdf", dpi=600)
#    return g


def create_violinplot(df):
    fig, axs = plt.subplots(ncols=df["restraint_index"].max(), figsize=(15,5), sharey=True)
    for i, exp in enumerate(df["exp"].unique()):
        df_exp = df[df["exp"]==exp]
        ax = axs[i]
        for state in df_exp["state"].unique():
            df_state = df_exp[df_exp["state"]==state]
            for restraint_index in df_state["restraint_index"].unique():
                df_plot = df_state[df_state["restraint_index"]==restraint_index]
                #ax.violinplot(df_plot["model"], positions=[restraint_index], widths=0.8, showmeans=False, showextrema=False, showmedians=False)
                ax.scatter(df_state["restraint_index"], df_plot["model"], marker=".")
            ax.scatter(df_state["restraint_index"], df_state["exp"], color="red")
        ax.set_xlabel("Restraint Index")
        ax.set_title("Exp: {}".format(exp))
    axs[0].set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(f"{dir}/box_plot_J.pdf", dpi=600)

def plot_data(data_dir, nstates, figname):

    df = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_assignments.csv", index_col=0, skiprows=0, comment='#')
    msm_pops = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_pops.csv", index_col=0, skiprows=0, comment='#')
    noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/CS_J_NOE/*.noe")]
    J = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE/*.J')]
    cs = [pd.read_pickle(file) for file in biceps.toolbox.get_files(f'{data_dir}/CS_J_NOE/*.cs*')]

    var_results = {}

    #  Get the ensemble average observable
    noe_Exp = noe[0]["exp"].to_numpy()
    noe_model = [i["model"].to_numpy() for i in noe]
    noe_prior = np.array([w*noe_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    noe_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}-{three2one(row[1]['res2'])}.{row[1]['atom_name2']}" for row in noe[0].iterrows()]
    noe_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2']] for row in noe[0].iterrows()])

    #  Get the ensemble average observable
    J_Exp = J[0]["exp"].to_numpy()
    J_model = [i["model"].to_numpy() for i in J]
    J_prior = np.array([w*J_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    J_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}\n{three2one(row[1]['res2'])}.{row[1]['atom_name2']}\n{three2one(row[1]['res3'])}.{row[1]['atom_name3']}\n{three2one(row[1]['res4'])}.{row[1]['atom_name4']}" for row in J[0].iterrows()]
    J_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2'], row[1]['atom_index3'], row[1]['atom_index4']] for row in J[0].iterrows()])

    #  Get the ensemble average observable
    cs_Exp = cs[0]["exp"].to_numpy()
    cs_model = [i["model"].to_numpy() for i in cs]
    cs_prior = np.array([w*cs_model[i] for i,w in enumerate(msm_pops.to_numpy())]).sum(axis=0)
    cs_labels = [f"{three2one(row[1]['res1'])}.{row[1]['atom_name1']}" for row in cs[0].iterrows()]
    cs_label_indices = np.array([[row[1]['atom_index1']] for row in cs[0].iterrows()])

    # NOTE: Create the figure
    fig = plt.figure(figsize=(12,9))
    gs = gridspec.GridSpec(3, 1)
    ax1,ax2,ax3 = plt.subplot(gs[0,0]),plt.subplot(gs[1,0]),plt.subplot(gs[2,0])

    data = []
    for i in range(len(noe_prior)):
        try:
            data.append({"index":i,
                "prior noe":noe_prior[i], "exp noe":noe_Exp[i], "label":noe_labels[i]
                })
        except(Exception) as e:
            data.append({"index":i,
                "prior noe":noe_prior[i], "exp noe":noe_Exp[i], "label":noe_labels[i]
                })
    data1 = pd.DataFrame(data)
    _data1 = data1.sort_values(["exp noe"])
    noe_index = _data1.index.to_numpy()
    _data1 = _data1.reset_index()

    variance = np.zeros(len(_data1["exp noe"].to_numpy()))
    for state in range(nstates):
        ax1.scatter(x=_data1.index.to_numpy(), y=noe[state]["model"].to_numpy()[noe_index], color='k', label='_nolegend_', marker='.')
        dev = noe[state]["model"].to_numpy()[noe_index] - _data1["exp noe"].to_numpy()
        variance += dev*dev
    variance /= (nstates-1)
    var_results["noe"] = variance
    #print(variance)
    z = variance.copy()
    scatter_cmap="Greens"
    cmap=cm.get_cmap(scatter_cmap)
    norm = plt.Normalize(vmin=np.min(z), vmax=np.max(z))
    c = cmap(norm(z))
    ymax = 30
    ax1.scatter(x=_data1.index.to_numpy(), y=np.ones(len(variance))*ymax, color=c, edgecolor="k", label='_nolegend_', marker='s')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, fraction=0.05, location="top", orientation="horizontal", label=r"$\sigma^{2}$")
    #exit()

    ax1.scatter(x=_data1.index.to_numpy(), y=_data1["prior noe"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
    ax1.scatter(x=_data1.index.to_numpy(), y=_data1["exp noe"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
    ax1.set_xlim(-1, 140)
    ax1.set_ylim(0, ymax)
    ax1.legend(fontsize=14)
    ax1.set_xlabel(r"Index", size=16)
    ax1.set_ylabel(r"NOE distance ($\AA$)", size=16)

    data = []
    for i in range(len(J_prior)):
        try:
            data.append({"index":i,
                "prior J":J_prior[i], "exp J":J_Exp[i], "label":J_labels[i]
                })
        except(Exception) as e:
            data.append({"index":i,
                "prior J":J_prior[i], "exp J":J_Exp[i], "label":J_labels[i]
                })

    data1 = pd.DataFrame(data)

    variance = np.zeros(len(data1["exp J"].to_numpy()))
    for state in range(nstates):
        ax3.scatter(x=data1['label'].to_numpy(), y=J[state]["model"].to_numpy(), color='k', label='_nolegend_', marker='.')
        dev = J[state]["model"].to_numpy() - data1["exp J"].to_numpy()
        variance += dev*dev
    variance /= (nstates-1)
    var_results["J"] = variance
    #print(variance)
    z = variance.copy()
    scatter_cmap="Greens"
    cmap=cm.get_cmap(scatter_cmap)
    norm = plt.Normalize(vmin=np.min(z), vmax=np.max(z))
    c = cmap(norm(z))
    ymax = 12
    ax3.scatter(x=data1.index.to_numpy(), y=np.ones(len(variance))*ymax, color=c, edgecolor="k", label='_nolegend_', marker='s')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3, fraction=0.05, location="top", orientation="horizontal")#, label=r"$\sigma^{2}$")


    ax3.scatter(x=data1['label'].to_numpy(), y=data1["prior J"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
    ax3.scatter(x=data1['label'].to_numpy(), y=data1["exp J"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
    ax3.set_ylabel(r"J-coupling (Hz)", size=16)
    ax3.set_ylim(2, ymax)
    data = []
    for i in range(len(cs_prior)):
        try:
            data.append({"index":i,
                "prior cs":cs_prior[i], "exp cs":cs_Exp[i], "label":cs_labels[i]
                })
        except(Exception) as e:
            data.append({"index":i,
                "prior cs":cs_prior[i], "exp cs":cs_Exp[i], "label":cs_labels[i]
                })
    data1 = pd.DataFrame(data)

    variance = np.zeros(len(data1["exp cs"].to_numpy()))
    for state in range(nstates):
        ax2.scatter(x=data1['label'].to_numpy(), y=cs[state]["model"].to_numpy(), color='k', label='_nolegend_', marker='.')
        dev = cs[state]["model"].to_numpy() - data1["exp cs"].to_numpy()
        variance += dev*dev
    variance /= (nstates-1)
    var_results["cs"] = variance
    #print(variance)
    z = variance.copy()
    scatter_cmap="Greens"
    cmap=cm.get_cmap(scatter_cmap)
    norm = plt.Normalize(vmin=np.min(z), vmax=np.max(z))
    c = cmap(norm(z))
    ymax = 12
    ax2.scatter(x=data1.index.to_numpy(), y=np.ones(len(variance))*ymax, color=c, edgecolor="k", label='_nolegend_', marker='s')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2, fraction=0.05, location="top", orientation="horizontal")#, label=r"$\sigma^{2}$")



    ax2.scatter(x=data1['label'].to_numpy(), y=data1["prior cs"].to_numpy(), s=35, color="orange", label="Prior", edgecolor='black',)
    ax2.scatter(x=data1['label'].to_numpy(), y=data1["exp cs"].to_numpy(), s=25, color="r", label="Exp", edgecolor='black',)
    ax2.set_ylabel(r"Chemical Shift (ppm)", size=16)
    ax2.set_ylim(2, ymax)

    axs = [ax1,ax2,ax3]
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
    fig.savefig(f"{figname}", dpi=600)

    return var_results
# }}}


dir = os.getcwd()
out = "state_overlap"
biceps.toolbox.mkdir(out)

###############################################################################



df = pd.read_pickle("variance_of_state_overlap.pkl")
#print(df)

total_var = []
for i in range(len(df["FF"])):
    row = df.iloc[i]
    FF = row["FF"]
    nstates = row["nstates"]
    noe = np.array(row["noe"])
    J = np.array(row["J"])
    cs = np.array(row["cs"])
    total_var.append({"FF": FF, "nstates": nstates, "noe": noe.sum(),
        "J": J.sum(), "cs": cs.sum()})

total_var = pd.DataFrame(total_var)

for i,n in enumerate(list(set(total_var["nstates"].to_numpy()))):
    fig = plt.figure(figsize=(12,9))
    gs = gridspec.GridSpec(3, 1)
    ax1,ax2,ax3 = plt.subplot(gs[0,0]),plt.subplot(gs[1,0]),plt.subplot(gs[2,0])
    indices = np.where(total_var["nstates"].to_numpy() == n)[0]
    total_var.iloc[indices].plot.bar(x="FF", y="noe", ax=ax1)
    total_var.iloc[indices].plot.bar(x="FF", y="cs", ax=ax2)
    total_var.iloc[indices].plot.bar(x="FF", y="J", ax=ax3)
    fig.tight_layout()
    fig.savefig(f"{out}/variance_across_FF_{n}_states.png", dpi=600)


#print(df.columns)


exit()




sys_name = "CLN001"
skip_dirs = ['indices']
# recursively loop through all the directories FF and cluster variations
forcefield_dirs = next(os.walk(sys_name))[1]
forcefield_dirs = [ff for ff in forcefield_dirs if ff not in skip_dirs]
results = []
for forcefield in forcefield_dirs:#[2:-2]:
    FF_dir = f"{sys_name}/{forcefield}"
    print(FF_dir)
    data_dirs = next(os.walk(FF_dir))[1]
    for data_dir in data_dirs:
        data_dir = f"{FF_dir}/{data_dir}"
        print(data_dir)
        outdir_CS_J_NOE = f"{data_dir}/CS_J_NOE/"
        J_files = biceps.toolbox.get_files(f"{outdir_CS_J_NOE}/*.J")
        cs_files = biceps.toolbox.get_files(f"{outdir_CS_J_NOE}/*.cs*")
        noe_files = biceps.toolbox.get_files(f"{outdir_CS_J_NOE}/*.noe")
        nstates = len(J_files)
        var = plot_data(data_dir, nstates=nstates, figname=f"{out}/{forcefield}_{nstates}.pdf")
        results.append({"FF": forcefield, "nstates": nstates, "noe": var["noe"],
        "J": var["J"], "cs": var["cs"]})
        #exit()
results = pd.DataFrame(results)
results.to_pickle("variance_of_state_overlap.pkl")



#
#        dataframes = []
#
#        for i,file in enumerate(J_files):
#            df = pd.read_pickle(file)
#            df["state"] = np.ones(len(df))*int(i)
#            df["type"] = ["J" for j in range(len(df))]
#            dataframes.append(df)
#
#        df = pd.concat(dataframes)

        #create_violinplot(df)
#        exit()
#        grouped = df.groupby(["state", "type", "restraint_index"])
#        print(df)
#
#        fig = plt.figure(1, figsize=(10, 10))
#        grid = plt.GridSpec(1, 1, hspace=0.01, wspace=0.01)
#        ax = fig.add_subplot(grid[0, 0])
#        sns.violinplot(data=grouped, x="model", y="restraint_index", hue="type", kind="violin", inner="stick", ax=ax)
#        fig.savefig(f"{dir}/box_plot_J.pdf", dpi=600)
#        exit()
#
#
#        for i,file in enumerate(cs_files):
#            df = pd.read_pickle(file)
#            df["state"] = np.ones(len(df))*int(i)
#            df["type"] = ["cs" for j in range(len(df))]
#            dataframes.append(df)
#
#        for i,file in enumerate(noe_files):
#            df = pd.read_pickle(file)
#            df["state"] = np.ones(len(df))*int(i)
#            df["type"] = ["noe" for j in range(len(df))]
#            dataframes.append(df)
#
## ['exp', 'model', 'restraint_index', 'atom_index1', 'res1', 'atom_name1',
##'atom_index2', 'res2', 'atom_name2', 'atom_index3', 'res3', 'atom_name3',
##'atom_index4', 'res4', 'atom_name4', 'state', 'type']
#
#        df = pd.concat(dataframes)
#        #print(df)
#        #print(df.columns.to_list())
#        #df.groupby(["exp", "model", "restraint_index", "type", "state"
#        grouped = df.groupby(["state", "type", "restraint_index"]).agg("mean")
#        print(grouped)
#
#        ax = sns.catplot(data=grouped, x="state", y="restraint_index", hue="sex", kind="violin",)
#        fig = ax.get_figure()
#        fig.savefig(f"{dir}/box_plot.pdf", dpi=600)
#
#        #ax.axvline(exp, color="red", linewidth=2)
#
#        #J, cs_H, noe























