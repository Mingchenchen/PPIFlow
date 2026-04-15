import torch

three_to_one_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'UNK': 'X'}

one_to_three_letter = {v:k for k,v in three_to_one_letter.items()}

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}

num_to_letter = {v:k for k, v in letter_to_num.items()}

restype_to_heavyatom_names = {
    "ALA": ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    "ARG": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    "ASN": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    "ASP": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    "CYS": ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    "GLN": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    "GLU": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    "GLY": ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    "HIS": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    "ILE": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    "LEU": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    "LYS": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    "MET": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    "PHE": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    "PRO": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    "SER": ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    "THR": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    "TRP": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    "TYR": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    "VAL": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    "UNK": ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
}

symmetric_atom_names = {
    "ARG": [['NH1','NH2']],
    "ASN": [['OD1','ND2']],
    "ASP": [['OD1','OD2']],
    "GLN": [['OE1','NE2']],
    "GLU": [['OE1','OE2']],
    "HIS": [['ND1','CD2'],['NE2','CE1']],
    "LEU": [['CD1','CD2']],
    "PHE": [['CD1','CD2'],['CE1','CE2']],
    "TYR": [['CD1','CD2'],['CE1','CE2']],
    "VAL": [['CG1','CG2']],
}

# padded to [0,0]
symmetric_atom_index = {
    "ARG": [[9,10], [0,0]],
    "ASN": [[6,7], [0,0]],
    "ASP": [[6,7], [0,0]],
    "GLN": [[7,8], [0,0]],
    "GLU": [[7,8], [0,0]],
    "HIS": [[6,7], [8,9]],
    "LEU": [[6,7], [0,0]],
    "PHE": [[6,7], [9,10]],
    "TYR": [[6,7], [8,9]],
    "VAL": [[5,6], [0,0]],
}

heavyatom_to_label = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'X': 4} # X is null token

van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

vdw_tensor = torch.tensor([1.7, 1.55, 1.52, 1.8, 0.0]) # 0.0 for 'X'

max_num_heavy_atoms = len(restype_to_heavyatom_names["ALA"])

chi_atoms = {
    'ARG': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD'],
        'chi3': ['CB','CG','CD','NE'],
        'chi4': ['CG','CD','NE','CZ'],
        'chi5': ['CD','NE','CZ','NH1']
    },
    'ASN': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','OD1'],
    },
    'ASP': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','OD1'],
    },
    'CYS': {
        'chi1': ['N','CA','CB','SG'],
    },
    'GLN': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD'],
        'chi3': ['CB','CG','CD','OE1']
    },
    'GLU': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD'],
        'chi3': ['CB','CG','CD','OE1'],
    },
    'HIS': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','ND1']
    },
    'ILE': {
        'chi1': ['N','CA','CB','CG1'],
        'chi2': ['CA','CB','CG1','CD1'],
    },
    'LEU': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD1'],
    },
    'LYS': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD'],
        'chi3': ['CB','CG','CD','CE'],
        'chi4': ['CG','CD','CE','NZ']
    },
    'MET': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','SD'],
        'chi3': ['CB','CG','SD','CE'],
    },
    'PHE': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD1'],
    },
    'PRO': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD'],
    },
    'SER': {
        'chi1': ['N','CA','CB','OG'],
    },
    'THR': {
        'chi1': ['N','CA','CB','OG1'],
    },
    'TRP': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD1'],
    },
    'TYR': {
        'chi1': ['N','CA','CB','CG'],
        'chi2': ['CA','CB','CG','CD1'],
    },
    'VAL': {
        'chi1': ['N', 'CA', 'CB', 'CG1'],
    }
}

# ASP chi2, GLU chi3, PHE chi2, TYR chi2
chi_alt_truths = torch.zeros(21,4)
chi_alt_truths[3, 1] = 1
chi_alt_truths[6, 2] = 1
chi_alt_truths[13, 1] = 1
chi_alt_truths[18, 1] = 1

chi_relevant_mask = torch.zeros(21,14).long()
chi_relevant_mask[:,:4] = 1

chi_true_indices = torch.zeros(21,4,14).long()
chi_mask = torch.zeros(21,4).float()

for res_idx, (aa, atom_list) in enumerate(restype_to_heavyatom_names.items()):
    for chi_idx in range(4): # 4 chi atoms, ignoring chi5
        key = f'chi{chi_idx+1}'
        if aa in chi_atoms.keys():
            if key in chi_atoms[aa].keys():
                chi_atom_list = chi_atoms[aa][key]
                chi_mask[res_idx, chi_idx] = 1.0
                for atom_idx, atom in enumerate(chi_atom_list):
                    chi_true_indices[res_idx, chi_idx, atom_list.index(atom)] = 1
                    chi_relevant_mask[res_idx, atom_list.index(atom)] = 1
            else:
                chi_true_indices[res_idx, chi_idx, :4] = 1
        else:
            chi_true_indices[res_idx, :, :4] = 1 # NOTE: we set the N, Ca, C, O atoms as dihedrals to maintain tensor shape, but mask out later

atom_types = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

atom37_to_14_mask = torch.zeros(21,37)
atom14_mask = torch.zeros(21,14)

for aa,atom_list in restype_to_heavyatom_names.items():
    aa_idx = letter_to_num[three_to_one_letter[aa]]
    for atom_idx,atom in enumerate(atom_list):
        if atom == '': continue
        atom37_to_14_mask[aa_idx,atom_types.index(atom)] = 1
        atom14_mask[aa_idx, atom_idx] = 1
