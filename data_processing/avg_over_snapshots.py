# Libraries:{{{
#import sys, os
import biceps
import mdtraj as md
import numpy as np
import pandas as pd
import os, re
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from tqdm import tqdm # progress bar
from itertools import combinations

#:}}}

# Methods:{{{

def split(string):
    match = re.match(r"([a-z]+)([0-9]+)", string[0].upper()+string[1:].lower(), re.I)
    items = match.groups()
    return items[0]




def compute_ensemble_avg_J(grouped_files, outdir, model="Bax2007", return_index=False, verbose=False):
    """Additional models are possible, such as 'Habeck'
    """

    indices = None
    skip_idx = [2,5,8] # NOTE: Remove the last GLY since it's not in experimental data
    for i,state in enumerate(grouped_files):
        _state = []
        for j,frame in enumerate(state):
            J = biceps.toolbox.get_J3_HN_HA(frame, model=model,
                    outname=f'{outdir}/J_state_{i}_snapshot_{j}.npy', verbose=False)
            _state.append(J[-1][0])

        frame = md.load(frame, top=state[0])
        topology = frame.topology
        table = topology.to_dataframe()[0]
        #print(table)
        #for k in range(len(J[0])):
        #    print([topology.atom(idx) for idx in J[0][k]])
        #exit()

        data = np.mean(np.array(_state), axis=0)
        if verbose: print(data)
        np.savetxt(f'{outdir}/J_{i}.txt', np.delete(data, skip_idx, axis=0))
    if return_index: indices = np.delete(np.array(J[0], dtype=int), skip_idx, axis=0)
    return indices

#def compute_ensemble_avg_CS(grouped_files, outdir, temp=298.0, pH=5.0,
#        atom_sel="H", return_index=False, verbose=False):
#    for i,state in enumerate(grouped_files):
#        _state = []
#        for j,frame in enumerate(state):
#            if verbose: print(f"Loading {frame} ...")
#            frame = md.load(frame, top=state[0])
#            shifts = md.nmr.chemical_shifts_shiftx2(frame, pH, temp)
#            shifts = shifts[0].unstack(-1)
#            shifts.to_pickle(f'{outdir}/CS_state_{i}_snapshot_{j}.pkl')
#            shifts = shifts[atom_sel].to_numpy()[1:] # NOTE: Removing the first GLY since it's not in experimental data
#            _state.append(shifts)
#        data = np.mean(np.array(_state), axis=0)
#        data = data[~np.isnan(data)]
#        if verbose: print(data)
#        #if return_index: indices = np.array(J[0], dtype=int)
#        np.savetxt(f'{outdir}/CS_{i}.txt', data)



def compute_ensemble_avg_CS(grouped_files, outdir, temp=298.0, pH=5.0,
        atom_sel="H", return_index=False, verbose=False, compute=True):
    for i,state in enumerate(grouped_files):
        _state = []
        for j,frame in enumerate(state):
            if verbose: print(f"Loading {frame} ...")
            frame = md.load(frame, top=state[0])
            topology = frame.topology
            table = topology.to_dataframe()[0]
            seq = np.array([split(str(res)) for res in topology.residues])
            skip_idx = np.where(seq=="Gly")[0]

            if compute:
                shifts = md.nmr.chemical_shifts_shiftx2(frame, pH, temp)
                shifts = shifts[0].unstack(-1)
                shifts.to_pickle(f'{outdir}/CS_state_{i}_snapshot_{j}.pkl')
            else:
                shifts = pd.read_pickle(f'{outdir}/CS_state_{i}_snapshot_{j}.pkl')
            columns = shifts.columns.to_list()
            #atom_sel = [col for col in columns if col.startswith("H")]
            atom_sel = [col for col in columns if (col == "H") or (col == "HA")]
            shifts = shifts[atom_sel]#.to_numpy()
            _state.append(np.delete(shifts.to_numpy(), skip_idx, axis=0)) # NOTE: Removing the first GLY since it's not in experimental data
        data = np.mean(np.array(_state), axis=0)
        data = data[~np.isnan(data)]
        print(data)
        if verbose: print(data)
        np.savetxt(f'{outdir}/CS_{i}.txt', data)



def compute_ensemble_avg_NOE(grouped_files, indices, outdir, verbose=False):

    indices = np.loadtxt(indices, dtype=int)
    for i,state in enumerate(grouped_files):
        distances = []
        for j,frame in enumerate(state):
            d = md.compute_distances(md.load(frame), indices)*10. # convert nm to Å
            distances.append(d)
        data = np.mean(np.array(distances), axis=0)
        #print(np.mean([d[0][1] for d in distances]))
        #print(data)
        #exit()
        np.savetxt(f'{outdir}/NOE_{i}.txt', data)


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





#:}}}

# get_noe_indices:{{{

def get_NOE_indices(structure_file, verbose=False):

    t = md.load(structure_file)
    topology = t.topology
    table, bonds = topology.to_dataframe()
    #serial  name element  resSeq resName  chainID segmentID

    df = pd.read_csv("CLN001/CLN001_NOE_data.csv", index_col=0)
    df = df.replace('.', np.NAN)
    df = df.dropna(axis=1, how='all')

    #print(df)
    # PDB_residue_no_1 PDB_residue_name_1 PDB_atom_name_1
    # PDB_residue_no_2 PDB_residue_name_2 PDB_atom_name_2
    indices = []
    exp_data,exp_err,restraint_index = [],[],[]
    _id, id = int(df.iloc[[0]]["ID"].to_numpy()[0]), 0
    for i in range(len(df)):
        row = df.iloc[[i]]

        resIndex1 = int(row["PDB_residue_no_1"].to_numpy()[0])
        resName1 = str(row["PDB_residue_name_1"].to_numpy()[0])
        symbol1 = str(row["PDB_atom_name_1"].to_numpy()[0])
        resSeq1 = int(row["Seq_ID_1"].to_numpy()[0])
        ###
        resIndex2 = int(row["PDB_residue_no_2"].to_numpy()[0])
        resName2 = str(row["PDB_residue_name_2"].to_numpy()[0])
        symbol2 =  str(row["PDB_atom_name_2"].to_numpy()[0])
        resSeq2 = int(row["Seq_ID_2"].to_numpy()[0])

        index1 = list(set(np.where((resName1 == table["resName"].to_numpy()))[0]).intersection(
            np.where((symbol1 == table["name"].to_numpy()))[0]).intersection(
                                np.where((resSeq1 == table["resSeq"].to_numpy()))[0]))
        index2 = list(set(np.where((resName2 == table["resName"].to_numpy()))[0]).intersection(
            np.where((symbol2 == table["name"].to_numpy()))[0]).intersection(
                                np.where((resSeq2 == table["resSeq"].to_numpy()))[0]))

        # NOTE: this is the same as above:
        #index1 = list(set(np.where((resSeq1 == table["resSeq"].to_numpy()) & (resName1 == table["resName"].to_numpy()))[0]).intersection(
        #    np.where((symbol1 == table["name"].to_numpy()))[0]).intersection())

        #index2 = list(set(np.where((resSeq2 == table["resSeq"].to_numpy()) & (resName2 == table["resName"].to_numpy()))[0]).intersection(
        #    np.where((symbol2 == table["name"].to_numpy()))[0]).intersection())


        if (len(index1) == 0) or (len(index2) == 0):
            if verbose: print("Warning: could not find index...")
        else:
            new_id = int(row["ID"].to_numpy()[0])
            if (_id != new_id):
                _id = new_id
                id += 1
            idx1,idx2 = table.iloc[index1[0]]["serial"], table.iloc[index2[0]]["serial"]
            indices.append(np.array([idx1,idx2]))
            exp_data.append(row["Distance_val"].to_numpy()[0])
            restraint_index.append(id)

    return np.array(indices),np.array(restraint_index),np.array(exp_data)

# }}}

# get_cs_indices:{{{
def get_cs_indices(structure_file, verbose=False):

    t = md.load(structure_file)
    topology = t.topology
    table = topology.to_dataframe()[0]


    #serial  name element  resSeq resName  chainID segmentID
    # NOTE: use CLN001_cs_data_.csv --- I made changes to the ID column
    df = pd.read_csv("CLN001/CLN001_cs_data_.csv", index_col=0)
    df = df.replace('.', np.NAN)
    df = df.dropna(axis=1, how='all')
    indices = []
    exp_data,exp_err,restraint_index = [],[],[]
    _id, id = int(df.iloc[[0]]["ID"].to_numpy()[0]), 0
    for i in range(len(df)):
        row = df.iloc[[i]]

        #resIndex1 = int(row["Seq_ID"].to_numpy()[0])
        resName1 = str(row["Comp_ID"].to_numpy()[0])
        symbol1 = str(row["Atom_ID"].to_numpy()[0])
        #symbol1 = str(row["Atom_type"].to_numpy()[0])
        resSeq1 = int(row["Seq_ID"].to_numpy()[0])
        if (symbol1 == "H") or (symbol1 == "HA"): pass
        else: continue

        if resName1.lower()=="gly":
            continue

        index1 = list(set(np.where((resName1 == table["resName"].to_numpy()))[0]).intersection(
            np.where((symbol1 == table["name"].to_numpy()))[0]).intersection(
                                np.where((resSeq1 == table["resSeq"].to_numpy()))[0]))

        if (len(index1) == 0):
            if verbose: print("Warning: could not find index...")
        else:
            new_id = int(row["ID"].to_numpy()[0])
            if (_id != new_id):
                _id = new_id
                id += 1
            idx1 = table.iloc[index1[0]]["serial"]
            indices.append(np.array([idx1]))
            exp_data.append(row["Val"].to_numpy()[0])
            exp_err.append(row["Val_err"].to_numpy()[0])
            restraint_index.append(id)

    return np.array(indices),np.array(restraint_index),np.array(exp_data),np.array(exp_err)
# }}}


if __name__ == "__main__":

    sys_name = "CLN001"

    get_indices = 0
    indices_dir = f"{sys_name}/indices"
    biceps.toolbox.mkdir(indices_dir)
    skip_dirs = ['indices']
    if get_indices:
        # NOTE: Get indices for NOE distances
        forcefield_dirs = next(os.walk(sys_name))[1]
        forcefield_dirs = [ff for ff in forcefield_dirs if ff not in skip_dirs]
        for forcefield in forcefield_dirs:

            FF_dir = f"{sys_name}/{forcefield}"
            data_dir = next(os.walk(FF_dir))[1][0]
            data_dir = f"{FF_dir}/{data_dir}"
            #print(data_dir)
            states = biceps.toolbox.get_files(f"{data_dir}/*_ncluster*_*structure0.pdb")
            structure_file = states[0]
            indices_NOE = f"{indices_dir}/{forcefield}_NOE_indices.txt"
            indices_cs = f"{indices_dir}/{forcefield}_cs_indices.txt"
            indices_J = f"{indices_dir}/{forcefield}_J_indices.txt"

            indices,restraint_index,exp_data,exp_err = get_cs_indices(structure_file, verbose=False)
            index_correction = 1
            indices = indices - index_correction
            np.savetxt(indices_cs, indices, fmt="%i")
            exp_data = np.array([restraint_index, exp_data, exp_err]).T
            np.savetxt(f"{sys_name}/exp_cs.txt", exp_data)
            #exit()

            indices,restraint_index,exp_data = get_NOE_indices(structure_file, verbose=False)
            index_correction = 1
            indices = indices - index_correction
            np.savetxt(indices_NOE, indices, fmt="%i")
            exp_data = np.array([restraint_index, exp_data]).T
            np.savetxt(f"{sys_name}/exp_NOE.txt", exp_data)
        exit()
    #exit()


    # recursively loop through all the directories FF and cluster variations
    forcefield_dirs = next(os.walk(sys_name))[1]
    forcefield_dirs = [ff for ff in forcefield_dirs if ff not in skip_dirs]
    pbar = tqdm(total=len(forcefield_dirs))
    for forcefield in forcefield_dirs:#[2:-2]:
        FF_dir = f"{sys_name}/{forcefield}"
        print(FF_dir)
        indices_NOE = f"{indices_dir}/{forcefield}_NOE_indices.txt"
        indices_cs = f"{indices_dir}/{forcefield}_cs_indices.txt"
        indices_J = f"{indices_dir}/{forcefield}_J_indices.txt"
        #continue
        data_dirs = next(os.walk(FF_dir))[1]
        pbar1 = tqdm(total=len(data_dirs))
        for data_dir in data_dirs:
            data_dir = f"{FF_dir}/{data_dir}"
            print(data_dir)
            # Create output directories for biceps input files
            outdir_CS_J_NOE = f"{data_dir}/CS_J_NOE/"
            if os.path.exists(outdir_CS_J_NOE+"0.noe"):
                pbar1.update(1)
                print(pbar1)
                continue
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

            compute_ensemble_avg_NOE(files_by_state, indices=indices_NOE, outdir=outdir_NOE)
            _indices_J = compute_ensemble_avg_J(files_by_state, outdir_J, return_index=1)
            np.savetxt(indices_J, _indices_J, fmt="%i")
            compute_ensemble_avg_CS(files_by_state, outdir_CS, temp=300.0, pH=5.5,
                    atom_sel="H", return_index=False, verbose=False, compute=False)

            # NOTE: code to read in the experimental chemical shift data
            #cs_indices = f"{sys_name}/cs_indices.txt"
            exp_cs_data = f"{sys_name}/exp_cs.txt"
            model_cs_data = f"{outdir_CS}/*.txt"

            #J_indices = f"{sys_name}/J_indices.txt"
            exp_J_data = f"{sys_name}/exp_J.txt"
            model_J_data = f"{outdir_J}/*.txt"

            #NOE_indices = f"{sys_name}/NOE_indices.txt"
            exp_NOE_data = f"{sys_name}/exp_NOE.txt"
            model_NOE_data = f"{outdir_NOE}/*.txt"


            verbose=True
            # NOTE: Prepare the input files for biceps
            preparation = biceps.Restraint.Preparation(nstates=nstates, top_file=states[0], outdir=outdir_CS_J_NOE)
            preparation.prepare_cs(exp_cs_data, model_cs_data, indices_cs, extension="H", verbose=verbose)
            preparation.prepare_J(exp_J_data, model_J_data, indices_J, extension="J", verbose=verbose)
            preparation.prepare_noe(exp_NOE_data, model_NOE_data, indices_NOE, verbose=verbose)
            pbar1.update(1)
            print(pbar1)
        pbar1.close()
        pbar.update(1)
        print(pbar)
    pbar.close()





