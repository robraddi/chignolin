#import sys, os
import biceps
import mdtraj as md
import numpy as np
import pandas as pd
import os
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from tqdm import tqdm # progress bar
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Methods:{{{

#def compute_ensemble_avg_NOE(grouped_files, indices, verbose=False):
#
#    data = []
#    for i,state in enumerate(grouped_files):
#        distances = []
#        for j,frame in enumerate(state):
#            d = md.compute_distances(md.load(frame), indices)*10. # convert nm to Å
#            distances.append(d)
#        data.append(np.mean(np.array(distances), axis=0))
#    return np.array(data)



#def get_NOE_indices(structure_file, verbose=False):
#
#    t = md.load(structure_file)
#    topology = t.topology
#    table = topology.to_dataframe()[0]
#    #print(table)
#    return np.array([(30,102), (30,91)])

def check_assignment(distances):
    _distances = distances#/10
    x,y = _distances[0],_distances[1]
    result = []

    if y <= 0.375:           result.append("folded")
    elif 0.375 < y <= 0.8:   result.append("misfolded")
    else:                    result.append("unfolded")

    if 0.5 < x <= 0.8:       result.append("folded")
    elif 0.25 < x <= 0.5:    result.append("misfolded")
    else:                    result.append("unfolded")

    return result



#:}}}



if __name__ == "__main__":

#    files = biceps.toolbox.get_files("*.pkl")
#    for file in files:
#        FF = file.split("_")[0]
#        #FF = "AMBER14SB"
#        #df = pd.read_pickle(f"{FF}_microstate_distance_check.pkl")
#        df = pd.read_pickle(file)
#        nclusters = list(set(df["ncluster"].to_numpy()))
#        #print(data.columns)
#        #exit()
#        xlabel = 'Asp3N-Gly7O'
#        ylabel = 'Asp3N-Thr8O'
#        for ncluster in nclusters:
#        #data = df.groupby(['FF', 'ncluster', 'microstate']).agg("min")
#            data = df.iloc[np.where(ncluster == df["ncluster"])[0]]
#            fig = plt.figure(figsize=(12,8))
#            gs = gridspec.GridSpec(1, 1)
#            ax1 = plt.subplot(gs[0,0])
#            ax1.scatter(x=data[xlabel].to_numpy(), y=data[ylabel].to_numpy(), s=25, color="b")
#            ax1.set_xlim(0, 20)
#            ax1.set_ylim(0, 20)
#            ax1.set_xlabel(xlabel, fontsize=16)
#            ax1.set_ylabel(ylabel, fontsize=16)
#            fig.tight_layout()
#            fig.savefig(f"{FF}_{ncluster}_.png")


    # NOTE: TODO:
    # 2) print out the prior population of this state with the distance


    legend_fontsize = 16

    files = biceps.toolbox.get_files("*.pkl")
    for file in files:
        FF = file.split("_")[0]
        #FF = "AMBER14SB"
        #df = pd.read_pickle(f"{FF}_microstate_distance_check.pkl")
        df = pd.read_pickle(file)
        nclusters = list(set(df["ncluster"].to_numpy()))
        #print(data.columns)
        #exit()
        xlabel = 'Asp3N-Gly7O'
        ylabel = 'Asp3N-Thr8O'
        for ncluster in nclusters:
        #data = df.groupby(['FF', 'ncluster', 'microstate']).agg("min")
            data = df.iloc[np.where(ncluster == df["ncluster"])[0]]
            fig = plt.figure(figsize=(12,8))
            gs = gridspec.GridSpec(1, 1)
            ax1 = plt.subplot(gs[0,0])
            x_data = data[xlabel].to_numpy()
            y_data = data[ylabel].to_numpy()
            ax1.scatter(x=x_data, y=y_data, s=25, color="b")

            indices = np.where((0 == data[f"match assignment ({xlabel})"]) & (0 == data[f"match assignment ({ylabel})"]))[0]
            data1 = data.iloc[indices]
            x_data = data1[xlabel].to_numpy()
            y_data = data1[ylabel].to_numpy()
            ax1.scatter(x=x_data, y=y_data, s=25, color="r")
            micro_data = data1[f"microstate"].to_numpy()
            for i in range(len(x_data)):
                text = f"{micro_data[i]}"
                ax1.text( x_data[i], y_data[i], str(text), color='g' , fontsize=legend_fontsize)

            ax1.set_xlim(0, 2.0)
            ax1.set_ylim(0, 2.0)
            ax1.set_xlabel(xlabel, fontsize=16)
            ax1.set_ylabel(ylabel, fontsize=16)
            fig.tight_layout()
            fig.savefig(f"{FF}_{ncluster}_.png")


        for ncluster in nclusters:
        #data = df.groupby(['FF', 'ncluster', 'microstate']).agg("min")
            data = df.iloc[np.where(ncluster == df["ncluster"])[0]]
            fig = plt.figure(figsize=(12,8))
            gs = gridspec.GridSpec(1, 1)
            ax1 = plt.subplot(gs[0,0])
            x_data = data[f"transformed ({xlabel})"].to_numpy()
            y_data = data[f"transformed ({ylabel})"].to_numpy()
            ax1.scatter(x=x_data, y=y_data, s=25, color="b")
            ax1.set_xlim(0, np.max(x_data))
            ax1.set_ylim(0, np.max(y_data))

            indices = np.where((0 == data[f"match assignment ({xlabel})"]) & (0 == data[f"match assignment ({ylabel})"]))[0]
            data1 = data.iloc[indices]
            #print(data1)
            data1.to_csv(f"{FF}_{ncluster}_misassign.csv")
            try:
                info = {"FF": data1["FF"].to_numpy()[0], "Total Population": np.sum(data1["population"].to_numpy()[::2]*100.),}
                print(pd.DataFrame(info))
            except(Exception) as e:
                #print(pd.DataFrame(info))
                pass
            #print(data1.columns)
            #exit()
            x_data = data1[f"transformed ({xlabel})"].to_numpy()
            y_data = data1[f"transformed ({ylabel})"].to_numpy()
            ax1.scatter(x=x_data, y=y_data, s=25, color="r")
            micro_data = data1[f"microstate"].to_numpy()
            #y_data = data1[f"transformed ({ylabel})"].to_numpy()
            for i in range(len(x_data)):
                text = f"{micro_data[i]}"
                ax1.text( x_data[i], y_data[i], str(text), color='g' , fontsize=legend_fontsize)

            ax1.set_xlabel(xlabel, fontsize=16)
            ax1.set_ylabel(ylabel, fontsize=16)
            fig.tight_layout()
            fig.savefig(f"{FF}_{ncluster}_transformed.png")





