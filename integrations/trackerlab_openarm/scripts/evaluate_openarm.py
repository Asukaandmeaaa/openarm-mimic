"""Evaluate an upstream OpenArm checkpoint in PhysX; no optimizer or training updates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint')
p.add_argument('--motion',default=str(REPO.parents[1]/'data/processed/wave_right_v1.npz'))
p.add_argument('--controller',choices=['policy','reference'],default='policy')
p.add_argument('--output',required=True)
p.add_argument('--num_envs',type=int,default=16)
p.add_argument('--initial_offset',type=float,default=0.,help='Uniform joint reset perturbation in radians')
p.add_argument('--seed',type=int,default=17)
AppLauncher.add_app_launcher_args(p);p.set_defaults(headless=True,device='cuda:0')
args=p.parse_args()
if args.controller=='policy' and not args.checkpoint:p.error('policy requires --checkpoint')
if args.initial_offset<0:p.error('offset must be nonnegative')
if (Path(args.output)/'metrics.json').exists():p.error('use a new output directory to preserve evaluations')
app=AppLauncher(args)

def main():
    import torch
    from trackerlab_supermimic.kinematics import G1Kinematics
    from trackerlab_supermimic.mdp import MDP,MAX_STEP_REWARD,flatten_state,unflatten_state,normalize_xi
    from trackerlab_supermimic.motion import MotionLib,GOAL_FRAME_DIM
    from trackerlab_supermimic.robot import REFERENCE,default_joint_pos_list,TRACKED_BODIES
    from trackerlab_supermimic.networks import Policy,WorldModel
    from trackerlab_supermimic.task import SuperMimicG1TaskCfg
    from trackerlab_supermimic.env import TrackingSim,make_sim
    from openarm_mimic.robot import load_robot,nontrivial_contacts
    import mujoco
    torch.manual_seed(args.seed)
    kin=G1Kinematics(args.device);motion=MotionLib.from_dir(args.motion,kin,args.device)
    mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(motion.files[0]),device=args.device),obs_mode='fixed_base50')
    cfg=SuperMimicG1TaskCfg();cfg.reset_noise=False
    env=TrackingSim(make_sim(args.device,task_cfg=cfg),args.num_envs,motion,mdp,eval_mode=True,task_cfg=cfg)
    env.reset(frame_ids=torch.zeros(args.num_envs,dtype=torch.long,device=args.device))
    ck=None;pi=None;wm=None;H=10;K=10
    if args.checkpoint:
        ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
        from trackerlab_supermimic.contract import fingerprint
        if ck.get('contract')!=fingerprint(motion.files[0]):raise ValueError('checkpoint robot/reference/code fingerprint does not match')
        if ck['args']['obs_mode']!='fixed_base50':raise ValueError('wrong robot/observation contract')
        if ck['args'].get('validate_only'):raise ValueError('validation checkpoint is UNTRAINED; do not report it as a trained policy')
        H,K=ck['args']['hist'],ck['args']['goal_frames']
        pi=Policy(50,GOAL_FRAME_DIM*K,hist=H).to(args.device);pi.load_state_dict(ck['pi']);pi.eval()
        wm=WorldModel().to(args.device);wm.load_state_dict(ck['wm']);wm.eval()
    if args.initial_offset:
        q=env.state()[4].clone();qd=env.state()[5].clone()
        q[:,kin.arm]+=(2*torch.rand_like(q[:,kin.arm])-1)*args.initial_offset
        q=q.clamp(kin.q_lo,kin.q_hi)
        env.robot.write_joint_state_to_sim(q,qd,joint_ids=env.joint_ids)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
    from trackerlab_supermimic.evaluation import record_rollout
    record_rollout(env,pi,wm,args.output,iteration=ck['iteration'] if ck else None,checkpoint=args.checkpoint,
        controller=args.controller,hist=H,goal_frames=K,initial_offset=args.initial_offset,seed=args.seed)

try:main()
except BaseException:
    import traceback
    traceback.print_exc();exit_code=1
else:exit_code=0
finally:
    sys.stdout.flush();sys.stderr.flush()
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance()
    try:app.app.close()
    except SystemExit:pass
raise SystemExit(exit_code)
