"""Run a trained TrackerLab OpenArm policy CLOSED LOOP in MuJoCo, with optional GUI.

Only simulation: no Isaac application, CAN, camera or robot driver imports.
Existing checkpoint/MDP/action math is reused; training sources are unchanged.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

if '--headless' in sys.argv:
    os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from openarm_mimic.robot import ROOT, ARM_NAMES, JointResolver, load_robot, arm_indices, set_targets, nontrivial_contacts
sys.path.insert(0,str(ROOT/'integrations/trackerlab_openarm/src'))
from trackerlab_supermimic.contract import fingerprint
from trackerlab_supermimic.kinematics import G1Kinematics
from trackerlab_supermimic.motion import MotionLib, CONTROL_DT, GOAL_FRAME_DIM
from trackerlab_supermimic.mdp import MDP, MAX_STEP_REWARD
from trackerlab_supermimic.networks import Policy
from trackerlab_supermimic.robot import action_scales, default_joint_pos_list


def load_presentation_robot():
    """Add visual scenery without changing the trained arm's frame or contacts.

    The official pedestal mounts the arms at z=0.698. Keep the training arm
    origin at zero and place the pedestal at -0.698 instead. These two scenery
    geoms are deliberately visual-only, not a pedestal collision validation.
    Original assets remain untouched so checkpoint fingerprints still match.
    """
    model_dir=ROOT/'assets/openarm_mujoco/v2'
    root=ET.parse(model_dir/'openarm_bimanual.xml').getroot()
    root.find('compiler').set('meshdir',str(model_dir/'assets'))
    asset=root.find('asset');world=root.find('worldbody')
    ET.SubElement(asset,'texture',type='skybox',builtin='gradient',
                  rgb1='0.79 0.85 0.91',rgb2='0.95 0.96 0.98',width='512',height='3072')
    ET.SubElement(asset,'texture',name='viewer_floor_texture',type='2d',builtin='checker',
                  rgb1='0.76 0.79 0.82',rgb2='0.88 0.90 0.92',width='512',height='512')
    ET.SubElement(asset,'material',name='viewer_floor_material',texture='viewer_floor_texture',
                  texuniform='true',texrepeat='4 4',reflectance='0')
    ET.SubElement(asset,'mesh',name='viewer_pedestal_mesh',file='visual/body/body_link0.stl',
                  scale='0.001 0.001 0.001')
    ET.SubElement(world,'geom',name='viewer_pedestal',type='mesh',mesh='viewer_pedestal_mesh',
                  pos='0 0 -0.698',rgba='0.30 0.32 0.35 1',contype='0',conaffinity='0',group='2')
    ET.SubElement(world,'geom',name='viewer_floor',type='plane',pos='0 0 -0.698',size='0 0 0.05',
                  material='viewer_floor_material',contype='0',conaffinity='0',group='2')
    ET.SubElement(world,'light',name='viewer_key',pos='1 -2 3',dir='-1 2 -3',directional='true',
                  diffuse='0.55 0.55 0.55',specular='0.15 0.15 0.15',castshadow='true')
    ET.SubElement(world,'light',name='viewer_fill',pos='-2 1 2',dir='2 -1 -2',directional='true',
                  diffuse='0.25 0.25 0.25',specular='0 0 0',castshadow='false')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1280',offheight='960')
    ET.SubElement(visual,'headlight',ambient='0.35 0.35 0.35',diffuse='0.45 0.45 0.45',specular='0.1 0.1 0.1')
    ET.SubElement(visual,'rgba',haze='0.88 0.91 0.94 1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    model.opt.timestep=.002
    return model,JointResolver(model)


class PolicySimulation:
    def __init__(self,checkpoint,motion_path,device='cpu',show_scene=True):
        self.checkpoint=Path(checkpoint).resolve();self.motion_path=Path(motion_path).resolve()
        self.device=device
        ck=torch.load(self.checkpoint,map_location=device,weights_only=False)
        self.contract=fingerprint(self.motion_path)
        if ck.get('contract')!=self.contract:raise ValueError('Checkpoint robot/reference/source contract mismatch')
        if ck['args'].get('validate_only'):raise ValueError('UNTRAINED validation checkpoint cannot be used as a learned policy')
        if ck['args']['obs_mode']!='fixed_base50':raise ValueError('Unsupported observation contract')
        self.iteration=ck['iteration'];self.H=ck['args']['hist'];self.K=ck['args']['goal_frames']
        self.model,self.mapper=load_presentation_robot() if show_scene else load_robot()
        self.show_scene=show_scene;self.data=mujoco.MjData(self.model)
        if self.model.nq!=18 or self.model.nv!=18:raise ValueError('Expected fixed-base 18-hinge OpenArm')
        self.qi,self.vi,self.limits=arm_indices(self.model)
        self.grips=[int(self.model.joint(f'openarm_{side}_finger_joint1').qposadr[0]) for side in ('left','right')]
        self.kin=G1Kinematics(device);self.motion=MotionLib(self.motion_path,self.kin,device)
        self.default=torch.tensor(default_joint_pos_list(self.motion_path),dtype=torch.float32,device=device)
        self.scale=torch.tensor(action_scales(self.motion_path),dtype=torch.float32,device=device)
        self.mdp=MDP(self.kin,self.motion,self.default,obs_mode='fixed_base50')
        self.policy=Policy(50,GOAL_FRAME_DIM*self.K,hist=self.H).to(device)
        self.policy.load_state_dict(ck['pi']);self.policy.eval();self.policy.requires_grad_(False)
        # Leave AMP_DTYPE=None: explicit float32 inference on CPU or CUDA.
        self.substeps=round(CONTROL_DT/self.model.opt.timestep)
        if abs(self.substeps*self.model.opt.timestep-CONTROL_DT)>1e-9:raise ValueError('Noninteger control period')
        with np.load(self.motion_path,allow_pickle=False) as archive:self.reference=dict(archive)
        self.wrists=[self.model.body(f'openarm_{side}_ee_base_link').id for side in ('left','right')]
        self.ref_wrists=[self.reference['body_names'].tolist().index(f'openarm_{side}_ee_base_link') for side in ('left','right')]
        self.reset()

    def state(self):
        q=torch.tensor(self.data.qpos.copy(),dtype=torch.float32,device=self.device)[None]
        qd=torch.tensor(self.data.qvel.copy(),dtype=torch.float32,device=self.device)[None]
        zero=torch.zeros(1,3,device=self.device)
        return zero,torch.eye(3,device=self.device)[None],zero.clone(),zero.clone(),q,qd

    @torch.inference_mode()
    def reset(self):
        # State initialization is the only reference -> qpos write. During a
        # rollout only MuJoCo integrates qpos; reference data only conditions goals.
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]=self.reference['qpos'][0]
        self.data.qvel[:]=self.motion.qd[0].cpu().numpy()
        set_targets(self.mapper,self.data.ctrl,self.data.qpos[self.qi],self.data.qpos[self.grips])
        mujoco.mj_forward(self.model,self.data)
        self.f=torch.zeros(1,dtype=torch.long,device=self.device)
        self.previous=torch.zeros(1,14,device=self.device)
        obs=self.mdp.obs(self.state(),self.previous,None,self.f)
        self.history=obs[:,None].expand(-1,self.H,-1).clone()
        self.qpos=[self.data.qpos.copy()];self.qvel=[self.data.qvel.copy()]
        self.actions=[];self.targets=[];self.rewards=[];self.wrist_errors=[];self.orientation_errors=[]
        self.contacts={};self.contact_steps=0;self.saturation_steps=0;self.physics_steps=0
        self.limit_violation=0.;self.inference_ms=[];self.reason=None

    @torch.inference_mode()
    def step(self,physics_callback=None):
        if self.reason:raise RuntimeError('Reset before stepping a completed episode')
        f=int(self.f.item())
        # Refresh the latest observation so mouse-applied state changes are also
        # sensed immediately, not replayed from stored PhysX states.
        self.history[:,-1]=self.mdp.obs(self.state(),self.previous,None,self.f)
        start=time.perf_counter()
        a=self.policy(self.history,self.motion.goal(self.f,self.K)).clamp(-20,20)
        if not bool(torch.isfinite(a).all()):raise FloatingPointError('Nonfinite policy action')
        target=(self.default[self.kin.arm]+a*self.scale).clamp(self.kin.q_lo[self.kin.arm],self.kin.q_hi[self.kin.arm])
        self.inference_ms.append((time.perf_counter()-start)*1000)
        target_np=target[0].cpu().numpy()
        set_targets(self.mapper,self.data.ctrl,target_np,self.reference['qpos'][f,self.grips])
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():raise FloatingPointError('Nonfinite MuJoCo state')
            self.physics_steps+=1
            pairs=nontrivial_contacts(self.model,self.data);self.contact_steps+=bool(pairs)
            for x,y,depth in pairs:
                key=' | '.join(sorted([x,y]));self.contacts[key]=min(depth,self.contacts.get(key,0.))
            capped=self.model.actuator_forcelimited.astype(bool)
            cap=np.max(abs(self.model.actuator_forcerange),axis=1)
            self.saturation_steps+=bool(np.any(abs(self.data.actuator_force[capped])>=.99*cap[capped]))
            q=self.data.qpos[self.qi]
            self.limit_violation=max(self.limit_violation,float(np.maximum(np.maximum(q-self.limits[:,1],self.limits[:,0]-q),0).max()))
            if physics_callback is not None:physics_callback()
        self.f+=1;nf=f+1;S=self.state()
        self.rewards.append(float(self.mdp.reward(S,self.f,a,self.previous)/MAX_STEP_REWARD))
        self.previous=a.clone()
        self.history=torch.cat((self.history[:,1:],self.mdp.obs(S,a,None,self.f)[:,None]),1)
        self.qpos.append(self.data.qpos.copy());self.qvel.append(self.data.qvel.copy())
        self.actions.append(a[0].cpu().numpy());self.targets.append(target_np)
        err=np.linalg.norm(self.data.xpos[self.wrists]-self.reference['body_pos_w'][nf,self.ref_wrists],axis=-1)
        self.wrist_errors.append(err)
        expected=Rotation.from_quat(self.reference['body_quat_w'][nf,self.ref_wrists][:,[1,2,3,0]])
        actual=Rotation.from_matrix(self.data.xmat[self.wrists].reshape(2,3,3))
        self.orientation_errors.append((expected.inv()*actual).magnitude())
        if np.max(err)>self.mdp.cfg.term_height:self.reason='tracking_failure'
        elif nf>=self.motion.num_frames-1:self.reason='completed'
        return a.clone()

    def save(self,path,reason=None):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        n=len(self.qpos);q=np.array(self.qpos);qd=np.array(self.qvel)
        np.savez_compressed(path/'trajectory.npz',qpos=q,qvel=qd,actions=np.array(self.actions),
                            joint_targets=np.array(self.targets),fps=50,time=np.arange(n)*CONTROL_DT,
                            reference=str(self.motion_path),checkpoint=str(self.checkpoint),
                            controller='MuJoCo closed-loop trained policy',wrist_error=np.array(self.wrist_errors))
        joint_err=q[1:,self.qi]-self.reference['joint_pos'][1:n]
        grip_err=q[:,self.grips]-self.reference['qpos'][:n,self.grips]
        report={'controller':'trained policy, closed-loop MuJoCo physics','checkpoint':str(self.checkpoint),
                'checkpoint_sha256':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                'iteration':self.iteration,'reference':str(self.motion_path),'contract':self.contract,
                'stop_reason':reason or self.reason or 'window_closed','frames':n,'expected_frames':self.motion.num_frames,
                'duration_s':(n-1)*CONTROL_DT,'control_hz':50,'physics_hz':1/self.model.opt.timestep,
                'policy_evaluations':len(self.actions),'physics_steps':self.physics_steps,
                'joint_rmse_by_arm_axis_rad':np.sqrt(np.mean(joint_err**2,axis=0)).tolist() if n>1 else None,
                'wrist_rmse_left_right_m':np.sqrt(np.mean(np.array(self.wrist_errors)**2,axis=0)).tolist() if n>1 else None,
                'wrist_orientation_rmse_left_right_deg':np.rad2deg(np.sqrt(np.mean(np.array(self.orientation_errors)**2,axis=0))).tolist() if n>1 else None,
                'mean_normalized_reward':float(np.mean(self.rewards)) if n>1 else None,
                'gripper_max_error_left_right_rad':np.max(abs(grip_err),axis=0).tolist(),
                'penetrating_contact_substeps':self.contact_steps,'penetrating_contact_pairs':self.contacts,
                'effort_saturation_fraction':self.saturation_steps/max(self.physics_steps,1),
                'joint_limit_violation_rad':self.limit_violation,
                'policy_inference_mean_ms':float(np.mean(self.inference_ms)) if n>1 else None,
                'precision':'float32','device':self.device,
                'visual_scene':self.show_scene,
                'scope':'Simulation-to-simulation test. Native MuJoCo dynamics and robot contact checks every 2 ms. Added pedestal/floor are visual-only and excluded from contact checks. No retraining or hardware. Viewer interactions, if any, affect this episode.'}
        (path/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return report


def camera_setup(camera,show_human=False):
    # Operator-selected front (reference image 2): view from -X, raised right arm on screen left.
    camera.lookat[:]=[0,0,-.18];camera.distance=2.05;camera.azimuth=180;camera.elevation=-12
    if show_human:camera.lookat[:]=[0,.4,-.05];camera.distance=3.0


class Video:
    def __init__(self,sim,path,speed,stride,prediction=None,layout='comparison',fps=None,
                 reference=None,show_reference=False,show_human=False):
        import imageio.v2 as imageio
        self.sim=sim;self.stride=stride;self.speed=speed;self.prediction=prediction
        self.layout=layout
        self.reference=reference;self.show_reference=show_reference;self.show_human=show_human
        self.physics_fps=fps;self.next_frame=0;self.path=Path(path);self.sample_steps=[]
        self.width,self.height=(960,720) if layout=='single' else (640,480)
        self.renderer=mujoco.Renderer(sim.model,height=self.height,width=self.width)
        self.refdata=mujoco.MjData(sim.model)
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera);camera_setup(self.camera,show_human)
        self.opt=mujoco.MjvOption();self.opt.geomgroup[3]=0;self.opt.sitegroup[:]=0
        self.writer=imageio.get_writer(path,fps=(fps or 50/stride)*speed,codec='libx264',quality=8)
    def capture_substep(self):
        # Use nearest 500 Hz physical state for each output timestamp. At 60 fps
        # this is 8/9 physics substeps per frame, at most 1 ms timing error.
        target=round(self.next_frame/(self.physics_fps*self.sim.model.opt.timestep))
        if self.sim.physics_steps<target:return
        self.frame(substep=True);self.sample_steps.append(self.sim.physics_steps);self.next_frame+=1
    def frame(self,substep=False):
        from PIL import Image,ImageDraw
        s=self.sim;f=int(s.f.item())
        if not substep and f%self.stride:return
        display_time=float(s.data.time) if substep else f*CONTROL_DT
        if substep:
            frame=min(display_time/CONTROL_DT,s.motion.num_frames-1)
            lo=int(frame);hi=min(lo+1,s.motion.num_frames-1);weight=frame-lo
            self.refdata.qpos[:]=(1-weight)*s.reference['qpos'][lo]+weight*s.reference['qpos'][hi]
        else:self.refdata.qpos[:]=s.reference['qpos'][f]
        mujoco.mj_forward(s.model,self.refdata)
        if self.reference is not None:self.reference.update(display_time)
        panels=[]
        for d in ((s.data,) if self.layout=='single' else (self.refdata,s.data)):
            self.renderer.update_scene(d,camera=self.camera,scene_option=self.opt)
            if d is s.data and self.prediction is not None:
                if not substep:self.prediction.update()
                self.prediction.add_to(self.renderer.scene)
            if self.reference is not None:
                self.reference.add_to(self.renderer.scene,self.show_reference and d is s.data,self.show_human)
            panels.append(self.renderer.render().copy())
        im=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(im)
        if self.show_human and self.reference is not None and self.reference.human_available:
            for k in range(len(panels)):
                im.paste(Image.fromarray(self.reference.inset_image()),(k*self.width+12,64))
        labels=('TRAINED POLICY + MuJoCo dynamics',) if self.layout=='single' else ('Reference (kinematic)','TRAINED POLICY + MuJoCo dynamics')
        if self.show_human and self.reference is not None and self.reference.source is not None:
            labels=tuple(label+' | Right: source BVH' for label in labels)
        if self.layout=='single':
            from PIL import ImageFont
            try:draw.font=ImageFont.truetype('DejaVuSans.ttf',16)
            except OSError:pass
        for k,label in enumerate(labels):
            x=k*self.width;draw.rectangle((x,0,x+self.width,52),fill='white')
            draw.text((x+8,6),label,fill='black')
            draw.text((x+8,26),f't={display_time:.2f}s | {self.speed:g}x display | NO HARDWARE',fill='black')
        if self.prediction is not None:
            x=0 if self.layout=='single' else self.width
            y=self.height-100
            draw.rectangle((x,y,x+self.width,self.height),fill='white')
            draw.text((x+8,y+6),self.prediction.label(display_time),fill='black')
            draw.text((x+8,y+28),'Light blue: WM future | Cyan crosses: future reference wrists',fill='black')
        if self.reference is not None:
            x=0 if self.layout=='single' else self.width
            y=self.height-48
            draw.rectangle((x,y,x+self.width,self.height),fill='white')
            draw.text((x+8,y+4),self.reference.label(self.show_reference,self.show_human),fill='black')
        self.writer.append_data(np.asarray(im))
    def close(self):
        self.writer.close();self.renderer.close()
        if self.physics_fps:
            np.savez_compressed(self.path.with_suffix('.timing.npz'),physics_steps=self.sample_steps,
                                sample_times=np.asarray(self.sample_steps)*self.sim.model.opt.timestep,
                                scheduled_times=np.arange(len(self.sample_steps))/self.physics_fps,
                                capture_fps=self.physics_fps,playback_speed=self.speed)


class PlaybackClock:
    """Schedule all fixed control steps independently of GUI/forecast refresh."""
    def __init__(self):self.reset()

    def reset(self,now=None):
        self.last=time.perf_counter() if now is None else now
        self.credit=0.

    def due(self,speed,paused=False,now=None):
        now=time.perf_counter() if now is None else now
        elapsed=max(0,now-self.last);self.last=now
        if paused:self.credit=0.;return 0
        self.credit+=elapsed*speed
        # Bound work per UI refresh; retain debt instead of skipping states.
        steps=min(25,int((self.credit+1e-10)/CONTROL_DT))
        self.credit=max(0.,self.credit-steps*CONTROL_DT)
        return steps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='runs/trackerlab_wave_j7_v5_run01/ckpt_002000.pt')
    p.add_argument('--motion',default='data/processed/wave_right_j7_v5.npz')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--headless',action='store_true');p.add_argument('--output')
    p.add_argument('--max-steps',type=int);p.add_argument('--exit-on-end',action='store_true')
    p.add_argument('--speed',type=float,default=2.,help='Viewer wall-clock speed (default: 2x); physics/control dt unchanged')
    p.add_argument('--video',action='store_true',help='Record headless MP4')
    p.add_argument('--video-layout',choices=('comparison','single'),default='comparison',help='Reference comparison or single simulation view')
    p.add_argument('--video-fps',type=int,help='Capture nearest native physics states at this fps (1..500); overrides video-stride')
    p.add_argument('--video-speed',type=float,default=2.);p.add_argument('--video-stride',type=int,default=2)
    p.add_argument('--no-prediction',action='store_true',help='Disable read-only world-model visualization')
    p.add_argument('--prediction-stride',type=int,default=5,help='Refresh GUI forecast every N control steps')
    p.add_argument('--prediction-seconds',type=float,choices=(.16,.3,.4),default=.4,
                   help='World-model forecast duration from prediction origin (default: 0.4 seconds)')
    p.add_argument('--reference',action=argparse.BooleanOptionalAction,default=False,
                   help='Show orange reference robot and wrist path (default: hidden)')
    p.add_argument('--human',action=argparse.BooleanOptionalAction,default=True,
                   help='Show source BVH on right and torso-local inset on top left (default: on)')
    p.add_argument('--anchor-human',action='store_true',help='Anchor source pelvis XY for comparison; preserve pose, heading and vertical motion')
    args=p.parse_args()
    if args.threads<1 or args.speed<=0 or not np.isfinite(args.speed) or args.video_stride<1 or args.video_speed<=0 or not np.isfinite(args.video_speed):p.error('Invalid threads/speed/stride')
    if args.max_steps is not None and args.max_steps<1:p.error('max-steps must be positive')
    if args.video and not args.headless:p.error('--video requires --headless')
    if args.video_fps is not None and (not args.video or not 1<=args.video_fps<=500):p.error('video-fps requires --video and must be 1..500')
    if args.prediction_stride<1:p.error('prediction-stride must be positive')
    prediction_steps=round(args.prediction_seconds/CONTROL_DT)
    torch.set_num_threads(args.threads)
    sim=PolicySimulation(ROOT/args.checkpoint,ROOT/args.motion,args.device)
    out=ROOT/(args.output or 'reports/mujoco_policy_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    (out/'session.json').write_text(json.dumps({'args':vars(args),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prediction_module_sha256':hashlib.sha256((ROOT/'scripts/mujoco_prediction_preview.py').read_bytes()).hexdigest(),
        'reference_module_sha256':hashlib.sha256((ROOT/'scripts/mujoco_reference_preview.py').read_bytes()).hexdigest(),
        'source_human_module_sha256':hashlib.sha256((ROOT/'scripts/mujoco_source_human.py').read_bytes()).hexdigest(),
        'prediction_horizon_steps':prediction_steps,'prediction_horizon_s':prediction_steps*CONTROL_DT,
        'prediction_controls_physics':False,'reference_controls_physics':False,
        'human_timing':'saved human_source_frame; held during transitions; display-normalized length',
        'mujoco_version':mujoco.__version__},indent=2)+'\n')
    print(f'LIVE policy -> position servos -> MuJoCo. Output: {out}',flush=True)
    prediction=None
    if not args.no_prediction and (not args.headless or args.video):
        from mujoco_prediction_preview import WorldModelPreview
        prediction=WorldModelPreview(sim,horizon=prediction_steps)
        prediction.update()
    reference=None
    if not args.headless or args.video:
        from mujoco_reference_preview import ReferencePreview
        reference=ReferencePreview(sim,anchor_human=args.anchor_human)
        if args.human and not reference.human_available:p.error('Motion archive has no aligned human skeleton data')
        session=json.loads((out/'session.json').read_text())
        session['source_human']=reference.source.metadata if reference.source is not None else None
        (out/'session.json').write_text(json.dumps(session,indent=2)+'\n')
    def advance(physics_callback=None):
        sim.step(physics_callback)
        if args.max_steps and int(sim.f.item())>=args.max_steps and not sim.reason:sim.reason='step_limit'
        if int(sim.f.item())%1000==0:print(f'MuJoCo policy: {int(sim.f.item())}/{sim.motion.num_frames-1}',flush=True)
    if args.headless:
        video_path=out/('simulation.mp4' if args.video_layout=='single' else 'comparison.mp4')
        video=Video(sim,video_path,args.video_speed,args.video_stride,prediction,args.video_layout,args.video_fps,
                    reference,args.reference,args.human) if args.video else None
        try:
            if video:
                if args.video_fps:video.capture_substep()
                else:video.frame()
            while not sim.reason:
                if video and args.video_fps:
                    if prediction is not None and int(sim.f.item())-prediction.origin>=args.prediction_stride:
                        prediction.update()
                    advance(video.capture_substep)
                else:
                    advance()
                    if video:video.frame()
        except BaseException:
            sim.reason='exception'
            raise
        finally:
            if video:video.close()
            sim.save(out/'episode_000')
        return
    from mujoco import viewer as mujoco_viewer
    events=queue.SimpleQueue();paused=False;episode=0;saved=False;quit_requested=False;speed=args.speed
    show_prediction=prediction is not None
    show_reference=args.reference;show_human=args.human
    print('P: pause | R: reset | F: front camera | W: world model | T: reference | H: human | +/-: speed | Q/Esc: close.',flush=True)
    with mujoco_viewer.launch_passive(sim.model,sim.data,key_callback=events.put) as viewer:
        with viewer.lock():
            camera_setup(viewer.cam,show_human);viewer.opt.geomgroup[3]=0;viewer.opt.sitegroup[:]=0
        playback=PlaybackClock()
        while viewer.is_running() and not quit_requested:
            start=time.perf_counter()
            while not events.empty():
                key=events.get()
                if key in (81,256):quit_requested=True
                elif key==80:paused=not paused;playback.reset()
                elif key==70:
                    with viewer.lock():camera_setup(viewer.cam,show_human)
                elif key==84:show_reference=not show_reference
                elif key==72:
                    if reference.human_available:
                        show_human=not show_human
                        with viewer.lock():camera_setup(viewer.cam,show_human)
                    else:print('Motion archive has no aligned human skeleton data.',flush=True)
                elif key==87 and prediction is not None:
                    show_prediction=not show_prediction
                    if show_prediction:prediction.update()
                elif key in (61,334):speed=min(speed*2,8.);print(f'Display speed {speed:g}x',flush=True)
                elif key in (45,333):speed=max(speed/2,.125);print(f'Display speed {speed:g}x',flush=True)
                elif key==82:
                    if not saved:sim.save(out/f'episode_{episode:03d}',reason='user_reset')
                    episode+=1
                    with viewer.lock():sim.reset()
                    if prediction is not None:prediction.update()
                    saved=False;paused=False;playback.reset()
            if quit_requested:break
            steps=playback.due(speed,paused or bool(sim.reason))
            if steps:
                with viewer.lock():
                    for _ in range(steps):
                        if sim.reason:break
                        advance()
            if show_prediction and int(sim.f.item())-prediction.origin>=args.prediction_stride:
                prediction.update()
            with viewer.lock():
                reference.update()
                viewer.user_scn.ngeom=0
                if show_prediction:prediction.add_to(viewer.user_scn)
                reference.add_to(viewer.user_scn,show_reference,show_human)
            if show_human and reference.human_available:
                vp=viewer.viewport
                width=min(240,max(1,vp.width-24));height=min(155,max(1,vp.height-24))
                viewer.set_images((mujoco.MjrRect(vp.left+12,vp.bottom+vp.height-height-12,width,height),
                                   reference.inset_image(width,height)))
            else:viewer.clear_images()
            status=prediction.label() if show_prediction else 'WM preview OFF'
            texts=[(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                             f't={int(sim.f.item())*CONTROL_DT:.2f}s | {speed:g}x playback | '+status+'\n'
                             'Light blue ghost/path: WM forecast | Cyan crosses: future reference wrists\n'
                             +reference.label(show_reference,show_human)+'\n'
                             'W: forecast | T: reference | H: human | F: front | P: pause | R: reset | Q: quit',None)]
            if show_human and reference.source is not None:
                texts.append((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                              'Source BVH (before joint mapping)',None))
            viewer.set_texts(texts)
            viewer.sync()
            if sim.reason and not saved:
                sim.save(out/f'episode_{episode:03d}');saved=True;paused=True
                print(f'Episode ended: {sim.reason}. R resets; Q closes.',flush=True)
                if args.exit_on_end:break
            time.sleep(max(0,1/60-(time.perf_counter()-start)))
    if not saved:sim.save(out/f'episode_{episode:03d}',reason='window_closed')


if __name__=='__main__':main()
