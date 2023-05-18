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

def compute_ensemble_avg_J(grouped_files, outdir, return_index=False, verbose=False):
    for i,state in enumerate(grouped_files):
        _state = []
        for j,frame in enumerate(state):
            J = biceps.toolbox.get_J3_HN_HA(frame, model="Habeck",
                    outname=f'{outdir}/J_state_{i}_snapshot_{j}.npy', verbose=False)
            _state.append(J[-1][0])
        data = np.mean(np.array(_state), axis=0)
        if verbose: print(data)
        if return_index: indices = np.array(J[0], dtype=int)
        np.savetxt(f'{outdir}/J_{i}.txt', data[:-1]) # NOTE: Remove the last GLY since it's not in experimental data

def compute_ensemble_avg_CS(grouped_files, outdir, temp=298.0, pH=5.0,
        atom_sel="H", return_index=False, verbose=False):
    for i,state in enumerate(grouped_files):
        _state = []
        for j,frame in enumerate(state):
            if verbose: print(f"Loading {frame} ...")
            frame = md.load(frame, top=state[0])
            shifts = md.nmr.chemical_shifts_shiftx2(frame, pH, temp)
            shifts = shifts[0].unstack(-1)
            shifts.to_pickle(f'{outdir}/CS_state_{i}_snapshot_{j}.pkl')
            shifts = shifts[atom_sel].to_numpy()[1:] # NOTE: Removing the first GLY since it's not in experimental data
            _state.append(shifts)
        data = np.mean(np.array(_state), axis=0)
        data = data[~np.isnan(data)]
        if verbose: print(data)
        if return_index: indices = np.array(J[0], dtype=int)
        np.savetxt(f'{outdir}/CS_{i}.txt', data)

def compute_ensemble_avg_NOE(grouped_files, indices, outdir, verbose=False):

    indices = np.loadtxt(indices, dtype=int)
    data = []
    for i,state in enumerate(grouped_files):
        distances = []
        for j,frame in enumerate(state):
            d = md.compute_distances(md.load(frame), indices)*10. # convert nm to Å
            distances.append(d)
        data.append(np.std(np.array(distances), axis=0))
        #data.append(np.std(np.array(distances)))
    # NOTE: an entire dataset uncertainty over all frames, all states, all datapoints
    data = np.array(data).mean()
    #print(data)
    #exit()

    # NOTE: an uncertainty for each data point over all frames, all states
    #data = np.array(data).mean(axis=1)
    #data = np.concatenate(data)
    #print(data)
    #exit()

    #data = np.mean(data)
    return data


## HOLD:{{{
##outdir = "NOE/"
##states = biceps.toolbox.get_files(data_dir+"cineromycinB_pdbs/*")
##nstates = len(states)
##ind=data_dir+'atom_indice_noe.txt'
##ind_noe = ind
##biceps.toolbox.mkdir(outdir)
##
##model_data_NOE = str(outdir+"*.txt")
#
#def get_NOE_indices(structure_file, outdir, return_index=False, verbose=False):
#
#    df = pd.read_csv("CLN001/CLN001_NOE_data.csv", index_col=0)
#    df = df.replace('.', np.NAN)
#    df = df.dropna(axis=1, how='all')
#    # PDB_residue_no_1 PDB_residue_name_1 PDB_atom_name_1
#    # PDB_residue_no_2 PDB_residue_name_2 PDB_atom_name_2
#    for i in range(len(df)):
#        row = df.iloc[[i]]
#        print(row["PDB_residue_no_1"].to_numpy())
#        print(row["PDB_residue_name_1"].to_numpy())
#        print(row["PDB_atom_name_1"].to_numpy())
#        print(row["PDB_residue_no_2"].to_numpy())
#        print(row["PDB_residue_name_2"].to_numpy())
#        print(row["PDB_atom_name_2"].to_numpy())
#        exit()
#
#    _state = []
#    selection_expression = 'name=="H"'
#    #selection_expression = '"H" in name'
#    t = md.load(structure_file)
#    topology = t.topology
#    sel = topology.select(selection_expression)
#    #print(sel)
#    pairs = np.array(list(combinations(sel,2)))
##    print(pairs)
##    print(pairs.shape)
##    exit()
#    table = topology.to_dataframe()[0]
#    #print(table)
#    print(table["resSeq"].to_numpy())
#    print(table["resName"].to_numpy())
#    #table["resName"]
#    #table["element"]
#
##serial  name element  resSeq resName  chainID segmentID
#
#
##:}}}
#


def get_NOE_indices(structure_file, verbose=False):

    #selection_expression = 'name=="H"'
    ##selection_expression = '"H" in name'
    t = md.load(structure_file)
    topology = t.topology
    table = topology.to_dataframe()[0]
    #serial  name element  resSeq resName  chainID segmentID

    df = pd.read_csv("CLN001/CLN001_NOE_data.csv", index_col=0)
    df = df.replace('.', np.NAN)
    df = df.dropna(axis=1, how='all')
    # PDB_residue_no_1 PDB_residue_name_1 PDB_atom_name_1
    # PDB_residue_no_2 PDB_residue_name_2 PDB_atom_name_2
    indices = []
    exp_data = []
    for i in range(len(df)):
        row = df.iloc[[i]]
        resIndex1 = int(row["PDB_residue_no_1"].to_numpy()[0])
        resName1 = str(row["PDB_residue_name_1"].to_numpy()[0])
        symbol1 = str(row["PDB_atom_name_1"].to_numpy()[0])
        ###
        resIndex2 = int(row["PDB_residue_no_2"].to_numpy()[0])
        resName2 = str(row["PDB_residue_name_2"].to_numpy()[0])
        symbol2 =  str(row["PDB_atom_name_2"].to_numpy()[0])

        index1 = list(set(np.where((resName1 == table["resName"].to_numpy()))[0]).intersection(
            np.where((symbol1 == table["name"].to_numpy()))[0]))
        index2 = list(set(np.where((resName2 == table["resName"].to_numpy()))[0]).intersection(
            np.where((symbol2 == table["name"].to_numpy()))[0]))

        if (len(index1) == 0) or (len(index2) == 0):
            if verbose: print("Warning: could not find index...")
        else:
            idx1,idx2 = table.iloc[index1[0]]["serial"], table.iloc[index2[0]]["serial"]
            indices.append(np.array([idx1,idx2]))
            exp_data.append(row["Distance_val"].to_numpy()[0])

    return np.array(indices),np.array(exp_data)






#:}}}



if __name__ == "__main__":

    sys_name = "CLN001"

    get_indices = False #True
    indices_NOE = f"{sys_name}/NOE_indices.txt"
    if get_indices:
        # NOTE: Get indices for NOE distances
        forcefield_dirs = next(os.walk(sys_name))[1]
        forcefield = forcefield_dirs[0]
        FF_dir = f"{sys_name}/{forcefield}"
        data_dir = next(os.walk(FF_dir))[1][0]
        data_dir = f"{FF_dir}/{data_dir}"
        states = biceps.toolbox.get_files(f"{data_dir}/*_ncluster*_nstructure0.pdb")
        structure_file = states[0]

        indices,exp_data = get_NOE_indices(structure_file, verbose=False)
        np.savetxt(indices_NOE, indices, fmt="%i")
        exp_data = np.array([list(range(len(exp_data))), exp_data]).T
        np.savetxt(f"{sys_name}/exp_NOE.txt", exp_data)
        exit()


    # recursively loop through all the directories FF and cluster variations
    results = []
    forcefield_dirs = next(os.walk(sys_name))[1]
    pbar = tqdm(total=len(forcefield_dirs))
    for forcefield in forcefield_dirs:#[2:-2]:
        FF_dir = f"{sys_name}/{forcefield}"
        print(FF_dir)
        #continue
        data_dirs = next(os.walk(FF_dir))[1]
        pbar1 = tqdm(total=len(data_dirs))
        for data_dir in data_dirs:
            data_dir = f"{FF_dir}/{data_dir}"
            print(data_dir)
            # Create output directories for biceps input files
            outdir_CS_J_NOE = f"{data_dir}/CS_J_NOE/"
            #if os.path.exists(outdir_CS_J_NOE+"0.noe"):
            #    pbar1.update(1)
            #    print(pbar1)
            #    continue
            biceps.toolbox.mkdir(outdir_CS_J_NOE)
            outdir_J = f"{data_dir}/J/"
            biceps.toolbox.mkdir(outdir_J)
            outdir_CS = f"{data_dir}/CS/"
            biceps.toolbox.mkdir(outdir_CS)
            outdir_NOE = f"{data_dir}/NOE/"
            biceps.toolbox.mkdir(outdir_NOE)

            states = biceps.toolbox.get_files(f"{data_dir}/*e0.pdb")
            nstates = len(states)


            snapshots = biceps.toolbox.get_files(f"{data_dir}/*structure*.pdb")
            print(f"nstates = {nstates}; nsnapshots = {len(snapshots)}")
            files_by_state = [biceps.toolbox.get_files(
                f"{data_dir}/*_ncluster{state}_*e*.pdb"
                ) for state in range(nstates)]
            print(files_by_state)
            #exit()

            std = compute_ensemble_avg_NOE(files_by_state, indices=indices_NOE, outdir=outdir_NOE)
            #compute_ensemble_avg_J(files_by_state, outdir_J)
            #compute_ensemble_avg_CS(files_by_state, outdir_CS, temp=300.0, pH=5.5,
            #        atom_sel="H", return_index=False, verbose=False)

            results.append({"sys_name": sys_name, "forcefield": forcefield, "nstates": nstates, "std": std})

            # NOTE: code to read in the experimental chemical shift data
            cs_indices = f"{sys_name}/cs_indices.txt"
            exp_cs_data = f"{sys_name}/exp_cs_H.txt"
            model_cs_data = f"{outdir_CS}/*.txt"

            J_indices = f"{sys_name}/J_indices.txt"
            exp_J_data = f"{sys_name}/exp_J.txt"
            model_J_data = f"{outdir_J}/*.txt"

            NOE_indices = f"{sys_name}/NOE_indices.txt"
            exp_NOE_data = f"{sys_name}/exp_NOE.txt"
            model_NOE_data = f"{outdir_NOE}/*.txt"

            verbose=True
            # NOTE: Prepare the input files for biceps
            #preparation = biceps.Restraint.Preparation(nstates=nstates, top_file=states[0], outdir=outdir_CS_J_NOE)
            #preparation.prepare_cs(exp_cs_data, model_cs_data, cs_indices, extension="H", verbose=verbose)
            #preparation.prepare_J(exp_J_data, model_J_data, J_indices, extension="J", verbose=verbose)
            #preparation.prepare_noe(exp_NOE_data, model_NOE_data, NOE_indices, verbose=verbose)
            pbar1.update(1)
            print(pbar1)
        pbar1.close()
        pbar.update(1)
        print(pbar)
    pbar.close()
    results = pd.DataFrame(results)
    results.to_pickle("prior_std_error_in_forward_model.pkl")





