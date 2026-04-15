import os, glob, time
import sys
import pandas as pd
from multiprocessing import Pool

import pyrosetta
from pyrosetta import *
from pyrosetta.teaching import *

from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

init('-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
     -ignore_zero_occupancy false -load_PDB_components false  -relax:default_repeats 2 -no_fconfig')

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pdb':
                L.append(os.path.join(root , file))
    return L

def main(args):
    all_pdb_name = sorted(glob.glob(os.path.join(args.pdb_dir, '*.pdb')))
    print(f"all_pdb_name: {all_pdb_name}")  # 调试输出
    output = []
    if len(all_pdb_name) > 0:
        print(f"Processing {len(all_pdb_name)} pdb")
    else:
        print(f"No pdb available in {args.pdb_dir}")
        return  # 提前返回，避免后续代码报错

    for idx, pdb in enumerate(all_pdb_name):
        pdb_name = os.path.basename(pdb).replace(".pdb","")
        pose = pose_from_pdb(pdb)
        original_pose = pose.clone()
        fr = FastRelax()
        scorefxn = get_score_function()
        fr.set_scorefxn(scorefxn)
        fr.max_iter(170)

        scorefxn_fa_atr = ScoreFunction()
        scorefxn_fa_atr.set_weight(fa_atr, 1.0)
        
        scorefxn_fa_rep = ScoreFunction()
        scorefxn_fa_rep.set_weight(fa_rep, 1.0)

        scorefxn_fa_intra_rep = ScoreFunction()
        scorefxn_fa_intra_rep.set_weight(fa_intra_rep, 1.0)
        
        scorefxn_fa_sol = ScoreFunction()
        scorefxn_fa_sol.set_weight(fa_sol, 1.0)

        scorefxn_lk_ball_wtd = ScoreFunction()
        scorefxn_lk_ball_wtd.set_weight(lk_ball_wtd, 1.0)
        
        scorefxn_fa_intra_sol = ScoreFunction()
        scorefxn_fa_intra_sol.set_weight(fa_intra_sol, 1.0)

        scorefxn_fa_elec = ScoreFunction()
        scorefxn_fa_elec.set_weight(fa_elec, 1.0)
        
        scorefxn_hbond_lr_bb = ScoreFunction()
        scorefxn_hbond_lr_bb.set_weight(hbond_lr_bb, 1.0)

        scorefxn_hbond_sr_bb = ScoreFunction()
        scorefxn_hbond_sr_bb.set_weight(hbond_sr_bb, 1.0)
        
        scorefxn_hbond_bb_sc = ScoreFunction()
        scorefxn_hbond_bb_sc.set_weight(hbond_bb_sc, 1.0)

        scorefxn_hbond_sc = ScoreFunction()
        scorefxn_hbond_sc.set_weight(hbond_sc, 1.0)
        
        scorefxn_dslf_fa13 = ScoreFunction()
        scorefxn_dslf_fa13.set_weight(dslf_fa13, 1.0)

        scorefxn_rama_prepro = ScoreFunction()
        scorefxn_rama_prepro.set_weight(rama_prepro, 1.0)
        
        scorefxn_p_aa_pp = ScoreFunction()
        scorefxn_p_aa_pp.set_weight(p_aa_pp, 1.0)
        
        scorefxn_fa_dun = ScoreFunction()
        scorefxn_fa_dun.set_weight(fa_dun, 1.0)
        
        scorefxn_omega = ScoreFunction()
        scorefxn_omega.set_weight(omega, 1.0)

        scorefxn_pro_close = ScoreFunction()
        scorefxn_pro_close.set_weight(pro_close, 1.0)
        
        scorefxn_yhh_planarity = ScoreFunction()
        scorefxn_yhh_planarity.set_weight(yhh_planarity, 1.0)
        
        scorefxn_ref = ScoreFunction()
        scorefxn_ref.set_weight(ref, 1.0)

        scorefxn_rg = ScoreFunction()
        scorefxn_rg.set_weight(rg , 1 )

        if not os.getenv("DEBUG"):
            fr.apply(pose)

        # interface analysis and energy calculation
        ia = InterfaceAnalyzerMover()
        # ia.set_interface(args.interface)
        ia.set_skip_reporting(True)
        ia.apply(pose)
    
        # 'ia.get_interface_dG()' is related with the defined score function, the same as 'ia.get_separated_interface_energy()'
        temp_dict = {'relaxed':scorefxn(pose),'interface_score':ia.get_interface_dG(), 
        'original':scorefxn(original_pose),'delta':scorefxn(pose) - scorefxn(original_pose),
        'fa_atr':scorefxn_fa_atr(pose),'fa_rep':scorefxn_fa_rep(pose),'fa_intra_rep':scorefxn_fa_intra_rep(pose),
        'fa_sol':scorefxn_fa_sol(pose),'lk_ball_wtd':scorefxn_lk_ball_wtd(pose),'fa_intra_sol':scorefxn_fa_intra_sol(pose),
        'fa_elec':scorefxn_fa_elec(pose),'hbond_lr_bb':scorefxn_hbond_lr_bb(pose),'hbond_sr_bb':scorefxn_hbond_sr_bb(pose),
        'hbond_bb_sc':scorefxn_hbond_bb_sc(pose),'hbond_sc':scorefxn_hbond_sc(pose),'dslf_fa13':scorefxn_dslf_fa13(pose),
        'rama_prepro':scorefxn_rama_prepro(pose),'p_aa_pp':scorefxn_p_aa_pp(pose),'fa_dun':scorefxn_fa_dun(pose),
        'omega':scorefxn_omega(pose),'pro_close':scorefxn_pro_close(pose),'yhh_planarity':scorefxn_yhh_planarity(pose),
        'ref':scorefxn_ref(pose), 'get_complexed_sasa':ia.get_complexed_sasa(),
        'get_interface_delta_sasa':ia.get_interface_delta_sasa()}

        if args.dump_pdb: pose.dump_pdb(os.path.join(args.output_dir, 'relax_' + os.path.basename(pdb)))

        output.append([pdb_name]+[v for k, v in temp_dict.items()])

    score_df = pd.DataFrame(output, columns=['pdb_name']+list(temp_dict.keys()))
    score_df.to_csv(os.path.join(args.output_dir, f'rosetta_complex_{args.batch_idx}.csv'),index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir',type=str, help="input pdb path.", required=True)
    parser.add_argument('--output_dir',type=str, help="output relaxed pdb path", required=True)
    parser.add_argument('--dump_pdb', default= False, type=bool_type, help="dump pdb or not")
    parser.add_argument('--batch_idx', type=str, help="batch index based on folder name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)