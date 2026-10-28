"""Physical evaluation shared by the training callback and standalone command; no parameter updates."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from .mdp import MDP,MAX_STEP_REWARD,flatten_state,unflatten_state
from .robot import REFERENCE,TRACKED_BODIES
from openarm_mimic.robot import load_robot,nontrivial_contacts

def record_rollout(env,pi,wm,output,iteration=None,checkpoint=None,controller='policy',hist=10,goal_frames=10,initial_offset=0.,seed=17,untrained=False):
    args=SimpleNamespace(output=output,checkpoint=checkpoint,controller=controller,num_envs=env.num_envs,device=env.device,initial_offset=initial_offset,seed=seed)
    motion,mdp,kin=env.motion,env.mdp,env.mdp.kin
    H,K=hist,goal_frames
    ck={'iteration':iteration} if pi is not None else None
    hist=mdp.obs(env.state(),env.a_prev,env.xi,env.f)[:,None].expand(-1,H,-1).clone()
    states=[flatten_state(env.state())[0].cpu().numpy()];actions=[]
    squared={key:[] for key in ('joint','velocity','wrist','orientation')}
    rewards=[];fail=[];saturation=[];physx_fk_error=0.;limit_frames=0;coupling=0.
    joint_squared=[];wrist_squared=[];gripper_squared=[];gripper_max=0.;physx_fk_rotation_error=0.
    model_names=env.backend.task.names
    gripper_ids=[model_names.index(f'openarm_{side}_finger_joint{i}') for side in ('left','right') for i in (1,2)]
    b_ids=[env.robot.body_names.index(n) for n in TRACKED_BODIES]
    with torch.no_grad():
        for k in range(motion.num_frames-1):
            if pi is not None and args.controller=='policy':a=pi(hist,motion.goal(env.f,K))
            else:a=(motion.q[env.f][:,kin.arm]-env.default_q[kin.arm])/env.action_scale
            out=env.step(a);S=unflatten_state(out['s1']);rf=k+1
            bp,bR,_,_=kin.forward(*S)
            squared['joint'].append(float((S[4][:,kin.arm]-motion.q[rf,kin.arm]).square().mean()))
            joint_squared.append((S[4][:,kin.arm]-motion.q[rf,kin.arm]).square().mean(0).cpu().numpy())
            wrist_squared.append((bp[:,[2,5]]-motion.body_p[rf,[2,5]]).square().sum(-1).mean(0).cpu().numpy())
            grip_error=S[4][:,gripper_ids]-motion.q[rf,gripper_ids]
            gripper_squared.append(grip_error.square().mean(0).cpu().numpy())
            gripper_max=max(gripper_max,float(grip_error.abs().max()))
            squared['velocity'].append(float((S[5][:,kin.arm]-motion.qd[rf,kin.arm]).square().mean()))
            squared['wrist'].append(float((bp[:,[2,5]]-motion.body_p[rf,[2,5]]).square().sum(-1).mean()))
            from .rotations import rot_angle_sq
            squared['orientation'].append(float(rot_angle_sq(bR[:,[2,5]],motion.body_R[rf,[2,5]]).mean()))
            physical_bp=env.robot.data.body_pos_w[:,b_ids]-env.scene.env_origins[:,None,:]
            physx_fk_error=max(physx_fk_error,float((physical_bp-bp).norm(dim=-1).max()))
            from .rotations import quat_to_mat
            physical_R=quat_to_mat(env.robot.data.body_link_quat_w[:,b_ids])
            physx_fk_rotation_error=max(physx_fk_rotation_error,float((physical_R-bR).abs().max()))
            rewards.append(float(out['r'].mean()/MAX_STEP_REWARD));fail.append(out['terminated'].cpu().numpy())
            sat=env.robot.data.applied_torque[:,env.arm_ids].abs()>=env.robot.data.joint_effort_limits[:,env.arm_ids]-.01
            saturation.append(float(sat.float().mean()))
            violation=(S[4]<kin.q_lo-1e-3)|(S[4]>kin.q_hi+1e-3)
            limit_frames+=int(violation.any(-1).any())
            for i,j in ((7,8),(16,17)):coupling=max(coupling,float((S[4][:,i]-S[4][:,j]).abs().max()))
            states.append(out['s1'][0].cpu().numpy());actions.append(out['a'][0].cpu().numpy())
            obs=mdp.obs(S,out['a'],env.xi,env.f);hist=torch.cat((hist[:,1:],obs[:,None]),1)
            if not torch.isfinite(out['s1']).all():raise FloatingPointError('nonfinite physical rollout')
            if (k+1)%1000==0:print(f'Evaluation progress: {k+1}/{motion.num_frames-1} control steps',flush=True)
    states=np.asarray(states);qpos=states[:,13:31];actions=np.asarray(actions)
    model,_=load_robot();data=mujoco.MjData(model);contact_frames=[];penetration=0.
    for i,q in enumerate(qpos):
        data.qpos[:]=q;mujoco.mj_forward(model,data);cs=nontrivial_contacts(model,data)
        if cs:contact_frames.append(i);penetration=max(penetration,max(-c[2] for c in cs))
    metric={
      'controller':args.controller,'untrained_validation_policy':untrained,'checkpoint':str(Path(args.checkpoint).resolve()) if ck else None,
      'upstream_commit':'606aae160b88635c97eefb05a3e4e4591afd7632',
      'iteration':ck['iteration'] if ck else None,'metrics_num_envs':args.num_envs,
      'recorded_and_geometry_checked_env':0,'frames':len(states),'expected_frames':motion.num_frames,
      'joint_rmse_rad':float(np.sqrt(np.mean(squared['joint']))),
      'joint_velocity_rmse_rad_s':float(np.sqrt(np.mean(squared['velocity']))),
      'wrist_rmse_m':float(np.sqrt(np.mean(squared['wrist']))),
      'wrist_orientation_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(squared['orientation'])))),
      'mean_normalized_reward':float(np.mean(rewards)),
      'failed_envs':int(env.failed_eval.sum()),'completed_without_tracking_failure':not bool(env.failed_eval.any()),
      'effort_saturation_fraction':float(np.mean(saturation)),
      'mujoco_geometry_collision_frames':len(contact_frames),'first_collision_frames':contact_frames[:20],
      'max_mesh_penetration_m':penetration,'joint_limit_violation_frames_any_env':limit_frames,
      'max_physx_vs_fk_body_position_error_m':physx_fk_error,'max_finger_coupling_error_rad':coupling,
      'initial_joint_offset_rad':args.initial_offset,'seed':args.seed,
      'reference_controller_is_not_a_learned_policy':args.controller=='reference',
      'reference':motion.files[0],
      'joint_rmse_by_arm_axis_rad':np.sqrt(np.mean(joint_squared,axis=0)).tolist(),
      'wrist_rmse_left_right_m':np.sqrt(np.mean(wrist_squared,axis=0)).tolist(),
      'gripper_rmse_left1_left2_right1_right2_rad':np.sqrt(np.mean(gripper_squared,axis=0)).tolist(),
      'gripper_max_abs_tracking_error_rad':gripper_max,
      'max_physx_vs_fk_rotation_matrix_error':physx_fk_rotation_error,
      'joint_order_mapping':[{ 'model_joint':n,'physx_index':env.joint_ids[i],
                              'physx_joint':env.robot.joint_names[env.joint_ids[i]]} for i,n in enumerate(model_names)],
      'collision_audit_scope':'MuJoCo mesh penetration audit of recorded PhysX env 0 poses at 50 Hz; not native PhysX contact-force or every-substep monitoring',
    }
    if wm is not None:
        st=torch.tensor(states,device=args.device);ac=torch.tensor(actions,device=args.device)
        starts=torch.arange(0,len(ac)-7,8,device=args.device);pred=unflatten_state(st[starts])
        with torch.no_grad():
            for k in range(8):
                pred=wm.step(pred,ac[starts+k],torch.zeros(len(starts),0,device=args.device))
                if k in (0,7):
                    target=unflatten_state(st[starts+k+1])
                    metric[f'wm_{k+1}step_joint_rmse_rad']=float((pred[4][:,kin.arm]-target[4][:,kin.arm]).square().mean().sqrt())
                    metric[f'wm_{k+1}step_velocity_rmse_rad_s']=float((pred[5][:,kin.arm]-target[5][:,kin.arm]).square().mean().sqrt())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'trajectory.npz',qpos=qpos,states=states,actions=actions,fps=50,controller='trackerlab_'+args.controller+('_UNTRAINED_validation' if untrained else ''),reference=motion.files[0])
    (out/'metrics.json').write_text(json.dumps(metric,indent=2)+'\n')
    print('EVALUATION '+json.dumps(metric),flush=True)

    return metric
