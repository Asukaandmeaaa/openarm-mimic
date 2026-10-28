"""IsaacLab PhysX collection. Import only AFTER AppLauncher is running."""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.utils import configclass
from openarm_mimic.robot import ROOT,ARM_NAMES

class TrackingSim:
    def __init__(self,task,cfg):
        self.task=task;self.cfg=cfg;self.device=task.device;self.N=cfg['num_envs']
        self.dt=cfg['physics_dt'];self.decimation=round(cfg['control_dt']/self.dt)
        if abs(self.dt*self.decimation-cfg['control_dt'])>1e-8:raise ValueError('Non-integer control decimation')
        self.sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt,render_interval=self.decimation,device=self.device))
        m=task.model;actuators={}
        for name in task.names:
            j=m.joint(name).id
            ids=[i for i in range(m.nu) if m.actuator_trnid[i,0]==j]
            if ids:
                i=ids[0];kp=float(m.actuator_gainprm[i,0]);kd=float(-m.actuator_biasprm[i,2])+float(m.dof_damping[j]);cap=float(m.actuator_forcerange[i,1])
            else:kp=0.;kd=float(m.dof_damping[j]);cap=10.
            actuators[name]=ImplicitActuatorCfg(joint_names_expr=[name],stiffness=kp,damping=kd,effort_limit_sim=cap,armature=float(m.dof_armature[j]),friction=0.)
        robot_cfg=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT/'assets/generated/usd/openarm.usd'),activate_contact_sensors=True,articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=8,solver_velocity_iteration_count=2)),actuators=actuators)
        # Independent references allow explicit per-environment fixed-base anchors.
        from isaaclab.assets import Articulation
        self.scene=InteractiveScene(InteractiveSceneCfg(num_envs=self.N,env_spacing=2.,replicate_physics=False))
        for i in range(self.N):
            robot_cfg.spawn.func(f'/World/envs/env_{i}/Robot',robot_cfg.spawn)
        robot_cfg.spawn=None
        robot_cfg.prim_path='/World/envs/env_.*/Robot'
        self.scene.articulations['robot']=Articulation(robot_cfg)
        self.scene.filter_collisions()

        # MJCF importer anchors its fixed joint in WORLD coordinates. Cloning an
        # Xform does not relocate that anchor; set every clone's anchor explicitly.
        from pxr import UsdPhysics,PhysxSchema,Gf
        for i,origin in enumerate(self.scene.env_origins.cpu().tolist()):
            joint=UsdPhysics.FixedJoint.Get(self.sim.stage,f'/World/envs/env_{i}/Robot/joints/rootJoint_openarm_root')
            if not joint:raise RuntimeError('Missing imported fixed-base joint')
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*origin))
            for side in ('left','right'):
                follower=self.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot/joints/openarm_{side}_finger_joint2')
                mimic=PhysxSchema.PhysxMimicJointAPI.Apply(follower,'rotX')
                mimic.CreateReferenceJointRel().SetTargets([f'/World/envs/env_{i}/Robot/joints/openarm_{side}_finger_joint1'])
                mimic.CreateGearingAttr(-1.);mimic.CreateOffsetAttr(0.)
        self.robot=self.scene['robot'];self.sim.reset();self.scene.reset();self.scene.update(self.dt)
        if set(self.robot.joint_names)!=set(task.names):raise ValueError(f'USD joint mismatch: {self.robot.joint_names}')
        self.ids=[self.robot.joint_names.index(n) for n in task.names]
        self.armids=[self.robot.joint_names.index(n) for n in ARM_NAMES]
        actual=self.robot.data.joint_pos_limits[0,self.armids]
        if not torch.allclose(actual,torch.stack((task.low,task.high),-1),atol=1e-4):raise ValueError('Imported joint limits differ from MuJoCo')
        self.f=torch.zeros(self.N,device=self.device,dtype=torch.long);self.ep=torch.zeros_like(self.f)
        self.prev=torch.zeros(self.N,14,device=self.device);self.returns=torch.zeros(self.N,device=self.device)
        self.bin_returns=torch.zeros((task.length+49)//50,device=self.device);self.bin_count=torch.zeros_like(self.bin_returns)
        self.reset_bins=torch.zeros_like(self.f)
        self.reset(torch.arange(self.N,device=self.device))
        from openarm_mimic.kinematics import TRACKED_BODIES
        self.bodyids=[self.robot.body_names.index(n) for n in TRACKED_BODIES]
        # Joint writes invalidate joint caches, but PhysX body transforms update on a step.

    def state(self):return torch.cat((self.robot.data.joint_pos[:,self.ids],self.robot.data.joint_vel[:,self.ids]),-1).clone()
    @torch.no_grad()
    def reset(self,ids,frame=None):
        if len(ids)==0:return
        if frame is None:
            # Return-based adaptive phase sampling plus 50% uniform coverage.
            weights=.5/self.bin_returns.numel()+.5*torch.softmax(-5*self.bin_returns,0)
            bins=torch.multinomial(weights,len(ids),replacement=True)
            f=(bins*50+torch.randint(50,(len(ids),),device=self.device)).clamp_max(self.task.length-2)
        else:f=torch.full((len(ids),),frame,device=self.device,dtype=torch.long)
        self.f[ids]=f;self.ep[ids]=0;self.returns[ids]=0;self.reset_bins[ids]=f//50
        q=self.robot.data.default_joint_pos[ids].clone();qd=torch.zeros_like(q)
        q[:,self.ids]=self.task.fullq[f];qd[:,self.ids]=self.task.fullqd[f]
        self.robot.write_joint_state_to_sim(q,qd,env_ids=ids);self.robot.reset(ids)
        self.prev[ids]=self.task.q[f]
        targets=q.clone();self.robot.set_joint_position_target(targets,env_ids=ids)
        self.scene.write_data_to_sim();self.scene.update(self.dt)
    @torch.no_grad()
    def step(self,a,auto_reset=True):
        s=self.state();f=self.f.clone();prev=self.prev.clone();ep=self.ep.clone()
        target=torch.zeros_like(self.robot.data.joint_pos)
        target[:,self.ids]=self.task.fullq[f]
        target[:,self.armids]=a
        self.robot.set_joint_position_target(target)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.dt)
        s1=self.state();self.f=(self.f+1).clamp_max(self.task.length-1);self.ep+=1
        if not torch.isfinite(s1).all():raise FloatingPointError(f'Non-finite PhysX state at steps {self.ep.tolist()}, q={s1[0].tolist()}')
        r,_,terms,error=self.task.reward(s1,a,prev,self.f)
        if not torch.isfinite(r).all():raise FloatingPointError(f'Non-finite reward: {terms}')
        failed=(error>self.cfg['reward']['termination_distance'])|(~torch.isfinite(s1).all(-1))
        ended=(self.f>=self.task.length-1)|(self.ep>=self.cfg['episode_steps'])
        done=failed|ended;self.returns+=r;self.prev.copy_(a)
        transition=dict(s=s,s1=s1,a=a,prev=prev,f=f,ep=ep,done=done)
        info={'reward':r,'wrist_error':error,'failed':failed,'terms':terms,'saturation':(self.robot.data.applied_torque[:,self.armids].abs() >= self.robot.data.joint_effort_limits[:,self.armids]-.01).float().mean(-1)}
        if auto_reset and done.any():
            ids=done.nonzero().flatten();bins=self.reset_bins[ids]
            sums=torch.zeros_like(self.bin_returns);counts=torch.zeros_like(sums)
            sums.index_add_(0,bins,self.returns[ids]/self.ep[ids].clamp_min(1));counts.index_add_(0,bins,torch.ones_like(self.returns[ids]))
            seen=counts>0;self.bin_returns[seen]=.9*self.bin_returns[seen]+.1*sums[seen]/counts[seen];self.bin_count+=counts
            self.reset(ids)
        return transition,info
