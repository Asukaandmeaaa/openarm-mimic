"""OpenArm model/control contract. PD values come from the model, not hardware measurement."""
import numpy as np
from openarm_mimic.robot import ROOT,ARM_NAMES,load_robot,arm_indices
NUM_J=18
NUM_A=14
G1_JOINT_NAMES=ARM_NAMES  # Upstream compatibility name, never a G1 asset.
TRACKED_BODIES=[f'openarm_{side}_{link}' for side in ('left','right') for link in ('link3','link4','ee_base_link')]
REFERENCE=ROOT/'data/processed/wave_right_v1.npz'
def default_joint_pos_list(path=REFERENCE):
    from pathlib import Path
    path=Path(path)
    if path.is_dir():path=path/'wave_right_v1.npz'
    return np.load(path,allow_pickle=False)['qpos'][0].tolist()
def action_scales(path=REFERENCE):
    m,_=load_robot();qi,_,limits=arm_indices(m);q=np.array(default_joint_pos_list(path))[qi]
    return (np.maximum(abs(limits[:,0]-q),abs(limits[:,1]-q))/20).tolist()
def action_scale(name):return action_scales()[ARM_NAMES.index(name)]

def full_position_target(reference_q, arm_target, arm_indices):
    """Preserve fixed grippers in model joint order while the policy controls arms."""
    target=reference_q.clone()
    target[...,arm_indices]=arm_target
    return target
