"""Named OpenArm model interface shared by motion preparation and simulation."""
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'assets/openarm_mujoco/src'))
from openarm_mujoco.v2.joint_resolver import JointResolver

ARM_NAMES = [f'openarm_{side}_joint{i}' for side in ('left', 'right') for i in range(1, 8)]

def load_robot():
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/openarm_mujoco/v2/openarm_bimanual.xml'))
    model.opt.timestep = 0.002
    return model, JointResolver(model)

def arm_indices(model):
    ids = np.array([model.joint(n).id for n in ARM_NAMES])
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids], model.jnt_range[ids].copy()

def set_arms(mapper, qpos, q, grippers=(0., 0.)):
    for k, side in enumerate(('left', 'right')):
        mapper.set_qpos(qpos, np.r_[q[k*7:(k+1)*7], grippers[k]], side)

def set_targets(mapper, ctrl, q, grippers=(0., 0.)):
    for k, side in enumerate(('left', 'right')):
        mapper.set_ctrl(ctrl, np.r_[q[k*7:(k+1)*7], grippers[k]], side)

def nontrivial_contacts(model, data):
    # Report actual negative-distance contacts; no blanket adjacent-link exclusion.
    return [(model.geom(c.geom1).name, model.geom(c.geom2).name, float(c.dist))
            for c in data.contact if c.dist < -1e-5]
