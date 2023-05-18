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

# Methods:{{{

def compute_ensemble_avg_NOE(grouped_files, indices, verbose=False):

    data = []
    for i,state in enumerate(grouped_files):
        distances = []
        for j,frame in enumerate(state):
            d = md.compute_distances(md.load(frame), indices) #in nm
            distances.append(d)
        data.append(np.mean(np.array(distances), axis=0))
    return np.array(data)



def get_NOE_indices(structure_file, verbose=False):

    t = md.load(structure_file)
    topology = t.topology
    table = topology.to_dataframe()[0]
    #print(table)
    #print(table.iloc[[30,91]])
    #exit()
    # Asp-Gly, Asp-Thr
    return np.array([(30,91), (30,102)])


def check_assignment(distances):
    _distances = distances
    x,y = _distances[0],_distances[1]
    result = []

    if y <= 0.5:           result.append("folded")
    elif 0.5 < y <= 0.8:   result.append("misfolded")
    else:                    result.append("unfolded")

    if 0.4 < x <= 0.8:       result.append("folded")
    elif 0.0 < x < 0.4:    result.append("misfolded")
    else:                    result.append("unfolded")

    return result

def transform(distances):
    B = 5
    return np.exp(-B*(distances))/(1+np.exp(-B*(distances)))

#:}}}



if __name__ == "__main__":

    sys_name = "CLN001"

    get_indices = False #True
    # NOTE: Get indices for NOE distances
    forcefield_dirs = next(os.walk(sys_name))[1]
    forcefield = forcefield_dirs[0]
    FF_dir = f"{sys_name}/{forcefield}"
    data_dir = next(os.walk(FF_dir))[1][0]
    data_dir = f"{FF_dir}/{data_dir}"
    states = biceps.toolbox.get_files(f"{data_dir}/*_ncluster*_structure0.pdb")
    structure_file = states[0]
    indices = get_NOE_indices(structure_file, verbose=False)
    labels = ["Asp3N-Gly7O", "Asp3N-Thr8O"]

    mapping = {
            "0": "folded",
            "1": "misfolded",
            "2": "unfolded",
            }

    #print(indices)
    #exit()

    # recursively loop through all the directories FF and cluster variations
    forcefield_dirs = next(os.walk(sys_name))[1]
    pbar = tqdm(total=len(forcefield_dirs))

    for forcefield in forcefield_dirs:
        FF_dir = f"{sys_name}/{forcefield}"
        #print(FF_dir)
        #continue
        data_dirs = next(os.walk(FF_dir))[1]
        pbar1 = tqdm(total=len(data_dirs))

        results = []
        for data_dir in data_dirs:
            data_dir = f"{FF_dir}/{data_dir}"
            print(data_dir)
            nstates = int(data_dir.split("_")[-1])
            # NOTE: get Tim's assignments file
            df = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_assignments.csv", index_col=0, skiprows=0, comment='#')
            df_pops = pd.read_csv(f"{data_dir}/inverse_distances_k{nstates}_msm_pops.csv", index_col=0, skiprows=0, comment='#')
            assignment = df.to_numpy()
            pops = df_pops.to_numpy()
            ###################################################################
            states = biceps.toolbox.get_files(f"{data_dir}/*_ncluster*_structure0.pdb")
            nstates = len(states)
            snapshots = biceps.toolbox.get_files(f"{data_dir}/*_ncluster*_structure*.pdb")
            print(f"nstates = {nstates}; nsnapshots = {len(snapshots)}")
            files_by_state = [biceps.toolbox.get_files(
                f"{data_dir}/*_ncluster{state}_structure*.pdb"
                ) for state in range(nstates)]
            distances = compute_ensemble_avg_NOE(files_by_state, indices=indices)

            for cluster,dist in enumerate(distances):
                dist = np.concatenate(dist)
                _assign = check_assignment(dist)
                _result = {
                    "FF": forcefield,
                    "ncluster": nstates,
                    "microstate": cluster,
                    "population": float(pops[cluster][0]),
                    "macrostate assignment": int(assignment[cluster]),
                    "mapped macrostate assignment": mapping[f"{assignment[cluster][0]}"],
                    }
                for i,d in enumerate(dist):
                    _result.update({
                        f"{labels[i]}": d,
                        f"checked assignment ({labels[i]})": _assign[i],
                        f"transformed ({labels[i]})": transform(d),
                        })
                    # determine from the distances wheter or not the assignment holds true...
                    if _assign[i] == mapping[f"{assignment[cluster][0]}"]:
                        _result.update({f"match assignment ({labels[i]})": 1})
                    else:
                        _result.update({f"match assignment ({labels[i]})": 0})
                    results.append(_result)

            pbar1.update(1)
            print(pbar1)
        results = pd.DataFrame(results)
        results.to_pickle(f"{forcefield}_microstate_distance_check.pkl")
        results.to_csv(f"{forcefield}_microstate_distance_check.csv")
        pbar1.close()
        pbar.update(1)
        print(pbar)
    pbar.close()






