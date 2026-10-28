"""Offline checks for robot reference labels; never import a simulator application."""
import hashlib
import json
import numpy as np
import mujoco
from .robot import ROOT, ARM_NAMES, load_robot, arm_indices


def validate_reference(z):
    """Reject misordered/inconsistent labels before a training loader accepts them.

    v1 retains its numerical body velocity convention. v2 additionally requires
    all body states to match FK and Jacobian velocities at the saved q/qd.
    This proves label consistency, not human similarity or hardware feasibility.
    """
    # NpzFile is lazy: repeated indexed reads decompress a whole array each time.
    # Materialize once before the per-frame/per-body verification loop.
    if isinstance(z,np.lib.npyio.NpzFile):z={name:z[name] for name in z.files}
    m,_=load_robot();qi,vi,limits=arm_indices(m)
    def require(condition,message):
        if not condition:raise ValueError('Invalid reference: '+message)
    require(z['joint_names'].tolist()==list(ARM_NAMES),'arm joint order')
    require(z['qpos_joint_names'].tolist()==[m.joint(i).name for i in range(m.njnt)],'full joint order')
    require(z['body_names'].tolist()==[m.body(i).name for i in range(1,m.nbody)],'body order')
    expected_hash=hashlib.sha256((ROOT/'assets/openarm_mujoco/v2/openarm_bimanual.xml').read_bytes()).hexdigest()
    require(str(z['model_sha256'])==expected_hash,'model hash mismatch')
    n=len(z['qpos']);dt=1/float(z['fps'])
    require(n>=3 and np.isfinite(dt) and dt>0,'sample count/fps')
    shapes={'time':(n,),'qpos':(n,m.nq),'joint_pos':(n,14),'joint_vel':(n,14),
            'body_pos_w':(n,m.nbody-1,3),'body_quat_w':(n,m.nbody-1,4),
            'body_lin_vel_w':(n,m.nbody-1,3),'body_ang_vel_w':(n,m.nbody-1,3)}
    for name,shape in shapes.items():
        require(z[name].shape==shape and np.isfinite(z[name]).all(),name+' shape/finite')
    require(np.allclose(z['time'],np.arange(n)*dt,atol=1e-9,rtol=0),'time grid')
    require(np.allclose(z['joint_pos'],z['qpos'][:,qi],atol=1e-9,rtol=0),'qpos versus joint_pos')
    require(np.allclose(z['joint_vel'],np.gradient(z['joint_pos'],dt,axis=0,edge_order=2),atol=1e-8,rtol=0),'joint_vel derivative')
    require(np.all(z['joint_pos']>=limits[:,0]-1e-9) and np.all(z['joint_pos']<=limits[:,1]+1e-9),'joint limits')
    require(np.allclose(np.linalg.norm(z['body_quat_w'],axis=-1),1,atol=1e-8,rtol=0),'quaternion norm')
    passive=np.array([i for i in range(m.nq) if i not in qi])
    require(np.ptp(z['qpos'][:,passive],axis=0).max()<1e-9,'only fixed grippers are supported')
    for side in ('left','right'):
        a,b=[int(m.joint(f'openarm_{side}_finger_joint{i}').qposadr[0]) for i in (1,2)]
        require(np.allclose(z['qpos'][:,a],z['qpos'][:,b],atol=1e-9,rtol=0),'finger mimic')
    stats={'frames':n,'format':str(z['format_version']),'labels_consistent':True}
    if str(z['format_version'])!='openarm_reference_v2':return stats
    cfg=json.loads(str(z['config_json']))
    if str(z['retarget_mode'])=='wrist_swing_j7':
        phase=z['source_time']-cfg['transition_s']
        require(phase.shape==(n,) and np.isfinite(phase).all() and np.all(np.diff(phase)>0),'source phase')
        active=(phase>=0)&(phase<=cfg['motion_duration_s'])
        require(np.any(active),'empty motion interval')
        require(np.ptp(z['joint_pos'][active,11])<1e-8,'J5 must hold during wrist swing')
        require(np.max(abs(z['joint_pos'][:,12]))<1e-9,'J6 must stay neutral')
        for side in ('left','right'):
            i=int(m.joint(f'openarm_{side}_finger_joint1').qposadr[0])
            require(np.max(abs(z['qpos'][:,i]-cfg['gripper_'+side]))<1e-9,'gripper differs from config')
    require(z['qpos_vel'].shape==(n,m.nv) and np.isfinite(z['qpos_vel']).all(),'full velocity')
    require(np.allclose(z['qpos_vel'][:,vi],z['joint_vel'],atol=1e-9,rtol=0),'full/arm velocity')
    require(np.max(abs(z['qpos_vel'][:,passive]))<1e-9,'passive velocity')
    d=mujoco.MjData(m);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    errors=np.zeros(4)
    for f in range(n):
        d.qpos[:]=z['qpos'][f];d.qvel[:]=z['qpos_vel'][f]
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        errors[0]=max(errors[0],np.max(abs(d.xpos[1:]-z['body_pos_w'][f])))
        dots=np.sum(d.xquat[1:]*z['body_quat_w'][f],axis=-1)
        errors[1]=max(errors[1],np.max(1-abs(dots)))
        for b in range(1,m.nbody):
            mujoco.mj_jacBody(m,d,jp,jr,b)
            errors[2]=max(errors[2],np.max(abs(jp@d.qvel-z['body_lin_vel_w'][f,b-1])))
            errors[3]=max(errors[3],np.max(abs(jr@d.qvel-z['body_ang_vel_w'][f,b-1])))
    require(np.max(errors)<1e-8,'FK/body velocity labels disagree: '+str(errors.tolist()))
    stats['max_fk_errors_position_quat_dot_linear_angular']=errors.tolist()
    return stats
