import pandas as pd
import biceps
import numpy as np
import mdtraj as md

pd.options.display.max_rows = None

# columns:{{{

"""
'Unnamed: 0', 'ID', 'Member_ID', 'Member_logic_code',
'Assembly_atom_ID_1', 'Entity_assembly_ID_1', 'Entity_ID_1',
'Comp_index_ID_1', 'Seq_ID_1', 'Comp_ID_1', 'Atom_ID_1', 'Atom_type_1',
'Atom_isotope_number_1', 'Resonance_ID_1', 'Assembly_atom_ID_2',
'Entity_assembly_ID_2', 'Entity_ID_2', 'Comp_index_ID_2', 'Seq_ID_2',
'Comp_ID_2', 'Atom_ID_2', 'Atom_type_2', 'Atom_isotope_number_2',
'Resonance_ID_2', 'Intensity_val', 'Intensity_lower_val_err',
'Intensity_upper_val_err', 'Distance_val', 'Distance_lower_bound_val',
'Distance_upper_bound_val', 'Contribution_fractional_val',
'Spectral_peak_ID', 'Spectral_peak_list_ID', 'PDB_record_ID_1',
'PDB_model_num_1', 'PDB_strand_ID_1', 'PDB_ins_code_1',
'PDB_residue_no_1', 'PDB_residue_name_1', 'PDB_atom_name_1',
'PDB_record_ID_2', 'PDB_model_num_2', 'PDB_strand_ID_2',
'PDB_ins_code_2', 'PDB_residue_no_2', 'PDB_residue_name_2',
'PDB_atom_name_2', 'Auth_entity_assembly_ID_1', 'Auth_asym_ID_1',
'Auth_chain_ID_1', 'Auth_seq_ID_1', 'Auth_comp_ID_1', 'Auth_atom_ID_1',
'Auth_alt_ID_1', 'Auth_atom_name_1', 'Auth_entity_assembly_ID_2',
'Auth_asym_ID_2', 'Auth_chain_ID_2', 'Auth_seq_ID_2', 'Auth_comp_ID_2',
'Auth_atom_ID_2', 'Auth_alt_ID_2', 'Auth_atom_name_2', 'Entry_ID',
'Gen_dist_constraint_list_ID'

"""

# }}}


df = pd.read_csv("CLN001/CLN001_NOE_data.csv")
new = df.copy()
new.drop(columns=['Unnamed: 0', 'ID', 'Member_ID', 'Member_logic_code',
          'Assembly_atom_ID_1', 'Entity_assembly_ID_1', 'Entity_ID_1',
          'Comp_index_ID_1', 'Atom_type_1',
          'Atom_isotope_number_1', 'Resonance_ID_1', 'Assembly_atom_ID_2',
          'Entity_assembly_ID_2', 'Entity_ID_2', 'Comp_index_ID_2',
          'Atom_type_2', 'Atom_isotope_number_2',
          'Resonance_ID_2', 'Intensity_val', 'Intensity_lower_val_err',
          'Intensity_upper_val_err', 'Contribution_fractional_val',
          'Spectral_peak_ID', 'Spectral_peak_list_ID', 'PDB_record_ID_1',
          'PDB_model_num_1', 'PDB_strand_ID_1', 'PDB_ins_code_1',
          'PDB_residue_no_1', 'PDB_residue_name_1', 'PDB_atom_name_1',
          'PDB_record_ID_2', 'PDB_model_num_2', 'PDB_strand_ID_2',
          'PDB_ins_code_2', 'PDB_residue_no_2', 'PDB_residue_name_2',
          'PDB_atom_name_2', 'Auth_entity_assembly_ID_1', 'Auth_asym_ID_1',
          'Auth_chain_ID_1', 'Auth_seq_ID_1', 'Auth_comp_ID_1', 'Auth_atom_ID_1',
          'Auth_alt_ID_1', 'Auth_atom_name_1', 'Auth_entity_assembly_ID_2',
          'Auth_asym_ID_2', 'Auth_chain_ID_2', 'Auth_seq_ID_2', 'Auth_comp_ID_2',
          'Auth_atom_ID_2', 'Auth_alt_ID_2', 'Auth_atom_name_2', 'Entry_ID',
          'Gen_dist_constraint_list_ID'], axis=1, inplace=True)



new["res1"] = [str(item[0])+str(item[1]) for item in zip(new['Comp_ID_1'].to_numpy(),new['Seq_ID_1'].to_numpy())]
new["res2"] = [str(item[0])+str(item[1]) for item in zip(new['Comp_ID_2'].to_numpy(),new['Seq_ID_2'].to_numpy())]
new.drop(columns=['Comp_ID_1','Comp_ID_2','Seq_ID_1','Seq_ID_2'], axis=1, inplace=True)
print(new)

noe_file = "1.noe"
data_dir = "CLN001/*/*/CS_J_NOE"
files = biceps.toolbox.get_files(f"{data_dir}/{noe_file}")

for file in files:
    print(file)
    yes = 0
    no = 0
    df = pd.read_pickle(file)
    for i in range(len(df)):
        exp = df.iloc[[i]]["exp"].to_numpy()
        res1,res2 = df.iloc[[i]]["res1"].to_numpy(),df.iloc[[i]]["res2"].to_numpy()
        atom_name1,atom_name2 = df.iloc[[i]]["atom_name1"].to_numpy(),df.iloc[[i]]["atom_name2"].to_numpy()
        index = np.where((new["res1"].to_numpy()==res1) & (new["res2"].to_numpy()==res2) &\
                (new["Atom_ID_1"].to_numpy()==atom_name1) & (new["Atom_ID_2"].to_numpy()==atom_name2))[0]
        val = new.iloc[index]["Distance_val"].to_numpy()
        if val == exp:
            yes += 1
        else:
            no += 1
            print(new.iloc[index])
            print(exp, val)
    print("yes = ",yes)
    print("no = ",no)
    print()

#    FF = file.split("CLN001/")[0].split("/")[0]
#    indices = np.loadtxt(biceps.toolbox.get_files("CLN001/indices/"+FF+"*NOE_indices.txt")[0])
#    print(data_dir)
#    yes = 0
#    no = 0
#    structure_file = biceps.toolbox.get_files(f"{file.replace('/'+noe_file,'')}/../*cluster0*e0.pdb")[0]
#    t = md.load(structure_file)
#    topology = t.topology
#    table, bonds = topology.to_dataframe()
#    print(table)
#    exit()
#    df = pd.read_pickle(file)
#
#    for k in range(len(indices)):
#        idx1,idx2 = indices[k]
#        table.iloc[idx1]["serial"], table.iloc[idx1]["serial"]
#
#        for i in range(len(df)):
#            exp = df.iloc[[i]]["exp"].to_numpy()
#            res1,res2 = df.iloc[[i]]["res1"].to_numpy(),df.iloc[[i]]["res2"].to_numpy()
#            atom_name1,atom_name2 = df.iloc[[i]]["atom_name1"].to_numpy(),df.iloc[[i]]["atom_name2"].to_numpy()
#
##serial  name element  resSeq resName  chainID segmentID
#
#
#    idx1,idx2 =
#    exit()



    print()
    print()
    print()
    print()






exit()














