"""CMU upper-body -> fixed-base OpenArm reference. No policy or hardware execution."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
from openarm_mimic.robot import ROOT, ARM_NAMES, load_robot, arm_indices, set_arms, nontrivial_contacts
sys.path.insert(0, str(ROOT/'vendor'))
from lafan_vendor.extract import read_bvh
from lafan_vendor.utils import quat_fk

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-8): raise ValueError('Degenerate human bone/frame')
    return v/n

def derivatives(values, dt):
    vel=np.gradient(values,dt,axis=0,edge_order=2)
    return vel,np.gradient(vel,dt,axis=0,edge_order=2)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/motions/wave_right.json');parser.add_argument('--report');parser.add_argument('--no-overwrite',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());model,mapper=load_robot();data=mujoco.MjData(model)
    report_path=ROOT/(args.report or 'reports/wave_right_reference.json')
    if args.no_overwrite and ((ROOT/cfg['output']).exists() or report_path.exists()):
        raise FileExistsError('Use fresh reference and report paths; existing experiments are preserved')
    qi,vi,limits=arm_indices(model);dt=1/cfg['control_hz'];grip=(cfg['gripper_left'],cfg['gripper_right'])
    rest=np.r_[cfg['standby_left'],cfg['standby_right']]
    set_arms(mapper,data.qpos,rest,grip);mujoco.mj_forward(model,data)
    shoulder=data.body('openarm_right_link2').xpos.copy()
    upper_len=np.linalg.norm(model.body('openarm_right_link3').pos+model.body('openarm_right_link4').pos)
    lower_len=np.linalg.norm(model.body('openarm_right_link5').pos+model.body('openarm_right_link6').pos)
    source=ROOT/cfg['source'];anim=read_bvh(str(source));human_quat,human=quat_fk(anim.quats,anim.pos,anim.parents)
    names=anim.bones
    def bone(name): return human[:,names.index(name)]
    # Body-fixed coordinates: x forward, y left, z up; no root translation or leg targets.
    y=unit(bone('LeftArm')-bone('RightArm'))
    up=unit(bone('Neck')-bone('Spine'))
    x=unit(np.cross(y,up));z=unit(np.cross(x,y));y=unit(np.cross(z,x))
    basis=np.stack([x,y,z],axis=-1)
    local=np.einsum('tji,tkj->tki',basis,human-bone('Spine1')[:,None,:])
    def localbone(name):return local[:,names.index(name)]
    u=unit(localbone('RightForeArm')-localbone('RightArm'))
    v=unit(localbone('RightHand')-localbone('RightForeArm'))
    elbow=shoulder+upper_len*u;wrist=elbow+lower_len*v
    # Tool -Z follows human fingers. Tool Y follows thumb-side direction projected
    # into the palm plane: explicit initial anatomical convention, review in preview.
    zhand=-unit(localbone('RightHandIndex1')-localbone('RightHand'))
    # RThumb origin coincides with wrist in CMU; use its End Site offset.
    import re
    thumb_block=source.read_text().split('JOINT RThumb',1)[1].split('End Site',1)[1]
    thumb_offset=np.array([float(n) for n in re.search(r'OFFSET\s+([^\n]+)',thumb_block).group(1).split()])
    thumb_q=human_quat[:,names.index('RThumb')]
    thumb_world=Rotation.from_quat(thumb_q[:,[1,2,3,0]]).apply(thumb_offset)
    thumb=np.einsum('tji,tj->ti',basis,thumb_world)
    yhand=unit(thumb-zhand*np.sum(thumb*zhand,axis=-1,keepdims=True))
    xhand=unit(np.cross(yhand,zhand));yhand=np.cross(zhand,xhand)
    handrot=np.stack([xhand,yhand,zhand],axis=-1)
    # Optional diagnostic candidate: hold the tool-to-palm alignment constant.
    # In this CMU BVH the zero-offset RightFingerBase carries the hand rotation;
    # RightHand alone contains wrist/forearm twist. Never remove FingerBase motion.
    orientation_mode=cfg.get('hand_orientation_mode','live_thumb_projection')
    wrist_swing=None
    if orientation_mode == 'wrist_swing_j7':
        # Semantic one-DOF retarget: hand rotation RELATIVE TO the forearm.
        # CMU RightHand carries forearm roll; RightFingerBase carries hand flexion.
        wrist_cfg=cfg['wrist_mapping']
        if float(wrist_cfg.get('j6_reference_rad',0.0)) != 0.0:
            raise ValueError('wrist_swing_j7 requires neutral J6=0')
        if wrist_cfg.get('forearm_roll_mode','hold_at_median_raised_pose') != 'hold_at_median_raised_pose':
            raise ValueError('Unsupported J5 setup mode')
        forearm_R=Rotation.from_quat(human_quat[:,names.index('RightHand')][:,[1,2,3,0]])
        palm_R=Rotation.from_quat(human_quat[:,names.index('RightFingerBase')][:,[1,2,3,0]])
        relative=forearm_R.inv()*palm_R
        source_times=np.linspace(0,cfg['motion_duration_s'],len(human))
        a,b=wrist_cfg['neutral_window_s'];active=(source_times>=a)&(source_times<=b)
        if not np.any(active):raise ValueError('Empty wrist neutral calibration window')
        neutral=relative[active].mean()
        delta=(relative*neutral.inv()).as_quat()
        delta=np.where(delta[:,3:4]<0,-delta,delta)
        axis=unit(np.asarray(wrist_cfg['human_relative_axis'],dtype=float))
        wrist_swing=2*np.arctan2(delta[:,:3]@axis,delta[:,3])
        wrist_swing*=float(wrist_cfg.get('gain',1.0))
        if abs(wrist_swing).max()>float(wrist_cfg['max_swing_rad']):
            raise ValueError('Source wrist swing exceeds configured review range')
        # Neutral fingers continue along the forearm, in a torso-frontal waving
        # plane. Discard source constant flexion/twist; do not fit them with J6.
        normal=np.array(wrist_cfg['wave_plane_normal'],dtype=float)
        normal=unit(normal-v*np.sum(v*normal,axis=1,keepdims=True))
        tangent=np.cross(normal,v)
        neutral_tool=np.stack([normal,tangent,-v],axis=-1)
        handrot=neutral_tool@Rotation.from_rotvec(wrist_swing[:,None]*np.array([1.,0,0])).as_matrix()
    elif orientation_mode == 'rigid_palm_first_frame':
        palm_q=human_quat[:,names.index('RightFingerBase')]
        palm_world=Rotation.from_quat(palm_q[:,[1,2,3,0]]).as_matrix()
        palm_local=np.einsum('tji,tjk->tik',basis,palm_world)
        # Explicit preview convention, not an anatomical or hardware calibration.
        alignment=palm_local[0].T@handrot[0]
        handrot=palm_local@alignment
    elif orientation_mode != 'live_thumb_projection':
        raise ValueError(f'Unknown hand_orientation_mode: {orientation_mode}')
    # Retarget convention only: roll the gripper frame about its finger axis.
    # This preserves the target finger direction but assigns the palm's two
    # transverse axes explicitly. Never swap J6/J7 driver channels or MJCF axes.
    tool_roll_deg=float(cfg.get('tool_roll_deg',0.0))
    if tool_roll_deg and orientation_mode != 'rigid_palm_first_frame':
        raise ValueError('tool_roll_deg requires a rigid palm alignment')
    handrot=handrot@Rotation.from_euler('z',tool_roll_deg,degrees=True).as_matrix()
    tin=np.linspace(0,cfg['motion_duration_s'],len(human));time=np.arange(round(cfg['motion_duration_s']/dt)+1)*dt
    et=PchipInterpolator(tin,savgol_filter(elbow,15,3,axis=0))(time)
    wt=PchipInterpolator(tin,savgol_filter(wrist,15,3,axis=0))(time)
    rt=Slerp(tin,Rotation.from_matrix(handrot))(time).as_matrix()
    unconstrained_rt=rt.copy()
    local_resampled=PchipInterpolator(tin,local)(time)
    swing_target=PchipInterpolator(tin,wrist_swing)(time) if wrist_swing is not None else None
    w=cfg['ik'];low=limits[7:,0]+cfg['joint_margin_rad'];high=limits[7:,1]-cfg['joint_margin_rad']
    previous=rest[7:].copy();qmotion=[];errors=[];ik_status=[]
    for frame in range(len(time)):
        if orientation_mode == 'wrist_swing_j7':
            def arm_residual(q4):
                set_arms(mapper,data.qpos,np.r_[rest[:7],q4,0.,0.,0.],grip)
                mujoco.mj_kinematics(model,data)
                return np.r_[w['wrist_position_weight']*(data.body('openarm_right_ee_base_link').xpos-wt[frame]),
                             w['elbow_position_weight']*(data.body('openarm_right_link4').xpos-et[frame]),
                             w['continuity_weight']*(q4-previous[:4]),
                             w['posture_weight']*(q4-rest[7:11])]
            step=w['max_joint_step_rad']
            lo=np.maximum(low[:4],previous[:4]-step) if frame else low[:4]
            hi=np.minimum(high[:4],previous[:4]+step) if frame else high[:4]
            result=least_squares(arm_residual,np.clip(previous[:4],lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=90,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            arm_residual(result.x)
            # J5 establishes the waving plane once the shoulder/elbow pose is
            # known. J6 stays at zero; only J7 receives the human wrist waveform.
            pre=data.body('openarm_right_link5').xmat.reshape(3,3).copy()
            forearm=-pre[:,2]
            n=np.array(cfg['wrist_mapping']['wave_plane_normal'],dtype=float)
            n=unit(n-forearm*np.dot(n,forearm))
            nlocal=pre.T@n
            q5=-np.arctan2(nlocal[1],nlocal[0]) # J5 axis is local -Z
            q7=float(swing_target[frame])
            if not (low[6]<=q7<=high[6]):
                raise ValueError(f'Wrist mapping outside model limits at frame {frame}: J5={q5}, J7={q7}')
            previous=np.r_[result.x,q5,0.,q7]
            qmotion.append(np.r_[rest[:7],previous]);ik_status.append(bool(result.success))
            continue
        def residual(q):
            set_arms(mapper,data.qpos,np.r_[rest[:7],q],grip);mujoco.mj_kinematics(model,data)
            current_R=data.body('openarm_right_ee_base_link').xmat.reshape(3,3)
            return np.r_[w['wrist_position_weight']*(data.body('openarm_right_ee_base_link').xpos-wt[frame]),
                         w['elbow_position_weight']*(data.body('openarm_right_link4').xpos-et[frame]),
                         w['orientation_weight_m_per_rad']*Rotation.from_matrix(rt[frame].T@current_R).as_rotvec(),
                         w['continuity_weight']*(q-previous),w['posture_weight']*(q-rest[7:])]
        step=w['max_joint_step_rad']
        lo=np.maximum(low,previous-step) if frame else low
        hi=np.minimum(high,previous+step) if frame else high
        result=least_squares(residual,np.clip(previous,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=90,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        previous=result.x; qmotion.append(np.r_[rest[:7],previous]);ik_status.append(bool(result.success))
    qmotion=np.array(qmotion)
    if orientation_mode == 'wrist_swing_j7':
        # Establish a comfortable forearm roll from the raised-arm interval,
        # then HOLD it. Re-solving roll every frame can recreate the unwanted
        # twist or flip the palm during entry/exit. J7 alone carries the waveform.
        a,b=cfg['wrist_mapping']['neutral_window_s']
        hold=float(np.median(qmotion[(time>=a)&(time<=b),11]))
        if not low[4]<=hold<=high[4]:raise ValueError(f'J5 setup pose outside range: {hold}')
        qmotion[:,11]=hold
        # This mode requests a single wrist axis with a held forearm roll, not
        # the independently moving world-frontal palm frame. Define its feasible
        # orientation BEFORE smoothing/retiming so fitting error remains measurable.
        for frame,angles in enumerate(qmotion):
            set_arms(mapper,data.qpos,angles,grip);mujoco.mj_kinematics(model,data)
            rt[frame]=data.body('openarm_right_ee_base_link').xmat.reshape(3,3)
    window=int(cfg.get('joint_smoothing_window',11))
    if window<5 or window%2!=1 or window>len(qmotion):raise ValueError('Invalid joint_smoothing_window')
    qmotion=savgol_filter(qmotion,window,3,axis=0)
    qmotion=np.clip(qmotion,limits[:,0]+cfg['joint_margin_rad'],limits[:,1]-cfg['joint_margin_rad'])
    # Transition joints use cubic Hermite with matched endpoint velocity.
    from scipy.interpolate import CubicHermiteSpline
    vel,_=derivatives(qmotion,dt);nt=round(cfg['transition_s']/dt);tt=np.arange(nt)*dt;duration=nt*dt
    enter=CubicHermiteSpline([0,duration],np.stack([rest,qmotion[0]]),np.stack([np.zeros(14),vel[0]]))(tt)
    leave=CubicHermiteSpline([0,duration],np.stack([qmotion[-1],rest]),np.stack([vel[-1],np.zeros(14)]))(np.arange(1,nt+1)*dt)
    q=np.concatenate([enter,qmotion,leave])
    from openarm_mimic.timing import retime_path
    scale=float(cfg.get('playback_time_scale',1.0))
    q,source_time=retime_path(q,dt,cfg['review_limits']['joint_speed_rad_s'],cfg['review_limits']['joint_acceleration_rad_s2'],time_scale=scale)
    human_source_frame=np.clip((source_time-duration)/dt,0,len(time)-1)
    qvel,qacc=derivatives(q,dt)
    target_orientation=Slerp(time,Rotation.from_matrix(rt))
    qfull=[];qdfull=[];bpos=[];bquat=[];bvel=[];bang=[];contacts={};contact_frames=0
    jacp=np.zeros((3,model.nv));jacr=np.zeros_like(jacp)
    body_names=[model.body(i).name for i in range(1,model.nbody)]
    for i,angles in enumerate(q):
        set_arms(mapper,data.qpos,angles,grip);data.qvel[:]=0;data.qvel[vi]=qvel[i];mujoco.mj_forward(model,data)
        qfull.append(data.qpos.copy());bpos.append(data.xpos[1:].copy());bquat.append(data.xquat[1:].copy())
        qdfull.append(data.qvel.copy());frame_v=[];frame_w=[]
        for body in range(1,model.nbody):
            mujoco.mj_jacBody(model,data,jacp,jacr,body)
            frame_v.append(jacp@data.qvel);frame_w.append(jacr@data.qvel)
        bvel.append(frame_v);bang.append(frame_w)
        pairs=nontrivial_contacts(model,data);contact_frames+=bool(pairs)
        for a,b,depth in pairs:contacts[' | '.join(sorted([a,b]))]=min(depth,contacts.get(' | '.join(sorted([a,b])),0))
        if duration-1e-9<=source_time[i]<=duration+time[-1]+1e-9:
            phase=float(np.clip(source_time[i]-duration,0,time[-1]))
            target_p=np.array([np.interp(phase,time,wt[:,k]) for k in range(3)])
            target_e=np.array([np.interp(phase,time,et[:,k]) for k in range(3)])
            target_R=target_orientation(phase).as_matrix()
            errors.append([np.linalg.norm(data.body('openarm_right_ee_base_link').xpos-target_p),np.linalg.norm(data.body('openarm_right_link4').xpos-target_e),Rotation.from_matrix(target_R.T@data.body('openarm_right_ee_base_link').xmat.reshape(3,3)).magnitude()])
    bpos=np.array(bpos);bquat=np.array(bquat);bvel=np.array(bvel);bang=np.array(bang)
    out=ROOT/cfg['output'];out.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out,fps=cfg['control_hz'],time=np.arange(len(q))*dt,joint_names=np.array(ARM_NAMES),joint_pos=q,joint_vel=qvel,joint_acc=qacc,qpos=np.array(qfull),qpos_vel=np.array(qdfull),qpos_joint_names=np.array([model.joint(i).name for i in range(model.njnt)]),body_names=np.array(body_names),body_pos_w=bpos,body_quat_w=bquat,body_lin_vel_w=bvel,body_ang_vel_w=bang,human_joint_names=np.array(names),human_upper_positions=local_resampled,human_parents=anim.parents,target_wrist=wt,target_elbow=et,target_wrist_rotation=rt,diagnostic_unconstrained_wrist_rotation=unconstrained_rt,motion_start_frame=int(np.searchsorted(source_time,duration)),motion_frame_count=int(np.sum((source_time>=duration)&(source_time<=duration+time[-1]))),human_source_frame=human_source_frame,source_time=source_time,format_version='openarm_reference_v2',retarget_mode=orientation_mode,config_json=json.dumps(cfg,sort_keys=True),model_sha256=hashlib.sha256((ROOT/'assets/openarm_mujoco/v2/openarm_bimanual.xml').read_bytes()).hexdigest())
    errors=np.array(errors);review=cfg['review_limits'];speed=float(np.abs(qvel).max());acc=float(np.abs(qacc).max());rmse=float(np.sqrt(np.mean(errors[:,0]**2)));ori=float(np.rad2deg(np.sqrt(np.mean(errors[:,2]**2))))
    flags=[]
    if speed>review['joint_speed_rad_s']:flags.append('speed_review_limit')
    if acc>review['joint_acceleration_rad_s2']:flags.append('acceleration_review_limit')
    if rmse>review['wrist_target_rmse_m']:flags.append('wrist_position_fit')
    if ori>review['orientation_target_rmse_deg']:flags.append('wrist_orientation_fit')
    if contact_frames:flags.append('reference_penetrating_contacts')
    if np.any(q<limits[:,0]) or np.any(q>limits[:,1]): flags.append('joint_limit_violation')
    if not all(ik_status):flags.append('ik_solver_not_converged')
    report=dict(output=str(out.relative_to(ROOT)),source=str(source.relative_to(ROOT)),config=cfg,frames=len(q),duration_s=(len(q)-1)*dt,upper_arm_length_m=float(upper_len),forearm_length_m=float(lower_len),max_joint_speed_rad_s=speed,max_joint_acceleration_rad_s2=acc,wrist_target_rmse_m=rmse,elbow_target_rmse_m=float(np.sqrt(np.mean(errors[:,1]**2))),wrist_orientation_rmse_deg=ori,contact_frames=contact_frames,contact_pairs=contacts,review_flags=flags,training_ready=False,note='Kinematic reference candidate only. Review retarget appearance, dynamic PD test and environment name mapping before training. Source motion slowed by design; no original source FPS claim.')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    if orientation_mode == 'wrist_swing_j7':
        diagnostic_target=Slerp(time,Rotation.from_matrix(unconstrained_rt))
        active=(source_time>=duration)&(source_time<=duration+time[-1])
        actual=Rotation.from_quat(bquat[active,body_names.index('openarm_right_ee_base_link')][:,[1,2,3,0]])
        diagnostic_error=(diagnostic_target(source_time[active]-duration).inv()*actual).magnitude()
        report['diagnostic_unconstrained_orientation_rmse_deg']=float(np.rad2deg(np.sqrt(np.mean(diagnostic_error**2))))
        report['orientation_target_semantics']='Pre-smoothing J1-J4 position IK + held J5 + J6=0 + source-relative J7. Unconstrained world-frontal palm error is retained separately, not a training objective.'
    report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2))
if __name__=='__main__':main()
