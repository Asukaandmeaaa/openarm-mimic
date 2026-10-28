"""Offline CMU arm trajectory candidates; independent of trained wave sources."""
import argparse
import hashlib
import json
import sys
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
from openarm_mimic.robot import ROOT, ARM_NAMES, load_robot, arm_indices, set_arms, nontrivial_contacts
from openarm_mimic.timing import retime_path
from openarm_mimic.reference import validate_reference
sys.path.insert(0,str(ROOT/'vendor'))
from lafan_vendor.extract import read_bvh
from lafan_vendor.utils import quat_fk


def unit(v):
    norm=np.linalg.norm(v,axis=-1,keepdims=True)
    if np.any(norm<1e-8):raise ValueError('Degenerate source bone/frame')
    return v/norm


def build(config,output):
    cfg=json.loads(config.read_text());output.mkdir(parents=True,exist_ok=False)
    model,mapper=load_robot();data=mujoco.MjData(model);qi,vi,limits=arm_indices(model)
    side=cfg['active_arm'];offset=0 if side=='left' else 7;active=slice(offset,offset+7)
    human_side=side.title();rest=np.r_[cfg['standby_left'],cfg['standby_right']]
    grip=(cfg['gripper_left'],cfg['gripper_right']);dt=1/cfg['control_hz']
    set_arms(mapper,data.qpos,rest,grip);mujoco.mj_forward(model,data)
    shoulder=data.body(f'openarm_{side}_link2').xpos.copy()
    upper=np.linalg.norm(model.body(f'openarm_{side}_link3').pos+model.body(f'openarm_{side}_link4').pos)
    lower=np.linalg.norm(model.body(f'openarm_{side}_link5').pos+model.body(f'openarm_{side}_link6').pos)
    source=ROOT/cfg['source'];anim=read_bvh(str(source));_,human=quat_fk(anim.quats,anim.pos,anim.parents)
    names=anim.bones
    # Fixed-base task: remove torso translation and yaw while respecting BVH Y-up.
    y=human[:,names.index('LeftArm')]-human[:,names.index('RightArm')];y[:,1]=0.;y=unit(y)
    up=np.broadcast_to([0.,1.,0.],y.shape);x=unit(np.cross(y,up));basis=np.stack([x,y,up],axis=-1)
    local=np.einsum('tji,tkj->tki',basis,human-human[:,names.index('Spine1'),None])
    def bone(name):return local[:,names.index(human_side+name)]
    u=unit(bone('ForeArm')-bone('Arm'));v=unit(bone('Hand')-bone('ForeArm'))
    elbow=shoulder+upper*u;wrist=elbow+lower*v
    # Preview convention: fingers extend along forearm. No claims about precise
    # human palm/finger tracking (CMU finger channels are not measured anatomy).
    normal=np.broadcast_to([0.,1.,0.] if cfg['name']=='handshake' else [-1.,0.,0.],v.shape).copy()
    z=-v;normal=normal-z*np.sum(normal*z,axis=1,keepdims=True)
    for i in range(len(normal)):
        if np.linalg.norm(normal[i])<.05:
            fallback=np.array([0.,0.,1.]);normal[i]=fallback-z[i]*np.dot(fallback,z[i])
    xx=unit(normal);yy=unit(np.cross(z,xx));rotation=np.stack([xx,yy,z],axis=-1)
    tin=np.linspace(0,cfg['motion_duration_s'],len(human));time=np.arange(round(cfg['motion_duration_s']/dt)+1)*dt
    wt=PchipInterpolator(tin,savgol_filter(wrist,15,3,axis=0))(time)
    et=PchipInterpolator(tin,savgol_filter(elbow,15,3,axis=0))(time)
    rt=Slerp(tin,Rotation.from_matrix(rotation))(time).as_matrix()
    local_resampled=PchipInterpolator(tin,local)(time)
    dof=4 if cfg.get('neutral_wrist') else 7
    low=limits[active,0][:dof]+cfg['joint_margin_rad'];high=limits[active,1][:dof]-cfg['joint_margin_rad']
    previous=rest[active][:dof].copy();qmotion=[];success=[]
    for frame in range(len(time)):
        def residual(q):
            allq=rest.copy();allq[offset:offset+dof]=q;set_arms(mapper,data.qpos,allq,grip);mujoco.mj_forward(model,data)
            current=data.body(f'openarm_{side}_ee_base_link')
            penetration=np.linalg.norm([min(c.dist,0.) for c in data.contact])
            return np.r_[current.xpos-wt[frame],.5*(data.body(f'openarm_{side}_link4').xpos-et[frame]),
                         cfg.get('orientation_weight',.06)*Rotation.from_matrix(rt[frame].T@current.xmat.reshape(3,3)).as_rotvec(),
                         .012*(q-previous),.002*(q-rest[active][:dof]),cfg.get('collision_weight',5.)*penetration]
        lo=np.maximum(low,previous-.15) if frame else low
        hi=np.minimum(high,previous+.15) if frame else high
        result=least_squares(residual,np.clip(previous,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=100,
                             ftol=1e-7,xtol=1e-7,gtol=1e-7)
        previous=result.x;q=rest.copy();q[offset:offset+dof]=previous;qmotion.append(q);success.append(bool(result.success))
    qmotion=savgol_filter(np.asarray(qmotion),11,3,axis=0)
    qmotion=np.clip(qmotion,limits[:,0]+cfg['joint_margin_rad'],limits[:,1]-cfg['joint_margin_rad'])
    transition=cfg['transition_s'];nt=round(transition/dt);vel=np.gradient(qmotion,dt,axis=0,edge_order=2)
    enter=CubicHermiteSpline([0,transition],[rest,qmotion[0]],[np.zeros(14),vel[0]])(np.arange(nt)*dt)
    leave=CubicHermiteSpline([0,transition],[qmotion[-1],rest],[vel[-1],np.zeros(14)])(np.arange(1,nt+1)*dt)
    path=np.clip(np.concatenate([enter,qmotion,leave]),limits[:,0]+cfg['joint_margin_rad'],limits[:,1]-cfg['joint_margin_rad'])
    review=cfg['review_limits']
    q,source_time=retime_path(path,dt,review['joint_speed_rad_s'],review['joint_acceleration_rad_s2'])
    qvel=np.gradient(q,dt,axis=0,edge_order=2);qacc=np.gradient(qvel,dt,axis=0,edge_order=2)
    body_names=[model.body(i).name for i in range(1,model.nbody)]
    fields={k:[] for k in ('qpos','qpos_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')}
    jacp=np.zeros((3,model.nv));jacr=np.zeros_like(jacp);contacts={};contact_frames=0
    for angles,velocity in zip(q,qvel):
        set_arms(mapper,data.qpos,angles,grip);data.qvel[:]=0;data.qvel[vi]=velocity;mujoco.mj_forward(model,data)
        fields['qpos'].append(data.qpos.copy());fields['qpos_vel'].append(data.qvel.copy())
        fields['body_pos_w'].append(data.xpos[1:].copy());fields['body_quat_w'].append(data.xquat[1:].copy())
        bv=[];bw=[]
        for b in range(1,model.nbody):
            mujoco.mj_jacBody(model,data,jacp,jacr,b);bv.append(jacp@data.qvel);bw.append(jacr@data.qvel)
        fields['body_lin_vel_w'].append(bv);fields['body_ang_vel_w'].append(bw)
        pairs=nontrivial_contacts(model,data);contact_frames+=bool(pairs)
        for a,b,d in pairs:
            key=' | '.join(sorted([a,b]));contacts[key]=min(d,contacts.get(key,0.))
    fields={k:np.asarray(v) for k,v in fields.items()}
    cfg['output']=str((output/'reference.npz').relative_to(ROOT))
    cfg['retarget_note']=('Active source arm only; torso yaw/translation removed, Y-up retained. '+
        ('J1-J4 elbow/wrist position fit; J5-J7 held neutral; source palm orientation NOT a target.' if dof==4 else
         'Forearm-aligned tool with preferred normal; fingers not imitated.'))
    archive=dict(fps=50,time=np.arange(len(q))*dt,joint_names=np.array(ARM_NAMES),joint_pos=q,joint_vel=qvel,joint_acc=qacc,
                 qpos_joint_names=np.array([model.joint(i).name for i in range(model.njnt)]),body_names=np.array(body_names),
                 human_joint_names=np.array(names),human_upper_positions=local_resampled,human_parents=anim.parents,
                 target_wrist=wt,target_elbow=et,target_wrist_rotation=rt,
                 human_source_frame=np.clip((source_time-transition)/dt,0,len(time)-1),source_time=source_time,
                 motion_start_frame=int(np.searchsorted(source_time,transition)),
                 motion_frame_count=int(np.sum((source_time>=transition)&(source_time<=transition+time[-1]))),
                 format_version='openarm_reference_v2',retarget_mode='gesture_forearm_ik',config_json=json.dumps(cfg,sort_keys=True),
                 model_sha256=hashlib.sha256((ROOT/'assets/openarm_mujoco/v2/openarm_bimanual.xml').read_bytes()).hexdigest(),**fields)
    validation=validate_reference(archive)
    np.savez_compressed(output/'reference.npz',**archive)
    active_frames=(source_time>=transition)&(source_time<=transition+time[-1]);phase=source_time[active_frames]-transition
    expected=np.column_stack([np.interp(phase,time,wt[:,j]) for j in range(3)])
    wrist_id=body_names.index(f'openarm_{side}_ee_base_link')
    error=np.linalg.norm(fields['body_pos_w'][active_frames,wrist_id]-expected,axis=1)
    orientation=(Slerp(time,Rotation.from_matrix(rt))(phase).inv()*Rotation.from_quat(fields['body_quat_w'][active_frames,wrist_id][:,[1,2,3,0]])).magnitude()
    flags=[]
    if np.sqrt(np.mean(error**2))>review['wrist_target_rmse_m']:flags.append('wrist_position_fit')
    if cfg.get('orientation_weight',.06)>0 and np.rad2deg(np.sqrt(np.mean(orientation**2)))>35:flags.append('preferred_tool_orientation_fit')
    if contact_frames:flags.append('kinematic_penetrating_contacts')
    full_limits=model.jnt_range
    if np.any(fields['qpos']<full_limits[:,0]-1e-9) or np.any(fields['qpos']>full_limits[:,1]+1e-9):
        flags.append('joint_or_gripper_limit_violation')
    if not all(success):flags.append('ik_not_converged')
    report=dict(name=cfg['name'],title=cfg['title'],active_arm=side,source=cfg['source'],frames=len(q),duration_s=(len(q)-1)*dt,
                wrist_target_rmse_m=float(np.sqrt(np.mean(error**2))),wrist_target_max_error_m=float(error.max()),
                preferred_orientation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(orientation**2)))),
                max_joint_speed_rad_s=float(abs(qvel).max()),max_joint_acceleration_rad_s2=float(abs(qacc).max()),
                contact_frames=contact_frames,contact_pairs=contacts,review_flags=flags,validation=validation,
                training_ready=False,orientation_is_objective=cfg.get('orientation_weight',.06)>0,
                scope='Kinematic candidate, no policy inference or training, no hardware; tool orientation is an explicit preview convention.')
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();build(ROOT/args.config,ROOT/args.output)
