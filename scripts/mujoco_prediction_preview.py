"""Read-only world-model rollout and MuJoCo scene overlays (no physics stepping)."""
import time
import mujoco
import numpy as np
import torch
from trackerlab_supermimic.networks import WorldModel
from trackerlab_supermimic.motion import CONTROL_DT
from mujoco_source_human import dashed_segments


class WorldModelPreview:
    def __init__(self, sim, horizon=20):
        self.sim=sim;self.horizon=horizon
        checkpoint=torch.load(sim.checkpoint,map_location=sim.device,weights_only=False)
        self.wm=WorldModel().to(sim.device)
        self.wm.load_state_dict(checkpoint['wm']);self.wm.eval();self.wm.requires_grad_(False)
        self.data=mujoco.MjData(sim.model)
        self.scene=mujoco.MjvScene(sim.model,maxgeom=2000)
        self.option=mujoco.MjvOption();self.option.geomgroup[3]=0;self.option.sitegroup[:]=0
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.pert=mujoco.MjvPerturb()
        self.origin=-1;self.qpos=None;self.elapsed_ms=0.

    @torch.inference_mode()
    def update(self):
        s=self.sim;start=time.perf_counter()
        self.origin=int(s.f.item())
        count=min(self.horizon,s.motion.num_frames-1-self.origin)
        state=tuple(x.clone() for x in s.state())
        history=s.history.clone();f=s.f.clone();previous=s.previous.clone()
        history[:,-1]=s.mdp.obs(state,previous,None,f)
        xi=torch.zeros(1,0,device=s.device)
        qs=[state[4][0].cpu().numpy().copy()];actions=[]
        for _ in range(count):
            action=s.policy(history,s.motion.goal(f,s.K)).clamp(-20,20)
            state=self.wm.step(state,action,xi)
            if not all(bool(torch.isfinite(x).all()) for x in state):
                raise FloatingPointError('Nonfinite world-model prediction')
            f=f+1
            history=torch.cat((history[:,1:],s.mdp.obs(state,action,None,f)[:,None]),1)
            qs.append(state[4][0].cpu().numpy().copy());actions.append(action[0].cpu().numpy().copy())
        self.qpos=np.asarray(qs);self.actions=np.asarray(actions)
        positions=[]
        for q in self.qpos:
            self.data.qpos[:]=q;mujoco.mj_forward(s.model,self.data)
            positions.append(self.data.xpos[s.wrists].copy())
        self.wrists=np.asarray(positions)
        self.target_frame=self.origin+count
        self.reference_wrists=s.reference['body_pos_w'][self.target_frame,s.ref_wrists]
        mujoco.mjv_updateScene(s.model,self.data,self.option,self.pert,self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        self.elapsed_ms=(time.perf_counter()-start)*1000

    def add_to(self, scene):
        """Append predicted robot meshes and wrist traces to an existing scene."""
        if self.qpos is None:return
        model=self.sim.model
        for i in range(self.scene.ngeom):
            src=self.scene.geoms[i]
            if src.objtype!=mujoco.mjtObj.mjOBJ_GEOM or src.objid<0:continue
            # Only arm visual meshes; never duplicate floor, pedestal or collision meshes.
            if model.geom_bodyid[src.objid]==0 or model.geom_group[src.objid]!=2:continue
            if scene.ngeom>=scene.maxgeom:raise RuntimeError('Prediction overlay scene capacity exceeded')
            dst=scene.geoms[scene.ngeom]
            for name in ('type','dataid','objtype','objid','category','matid','texcoord',
                         'segid','modelrbound','specular','shininess','reflectance','emission'):
                setattr(dst,name,getattr(src,name))
            for name in ('size','pos','mat'):getattr(dst,name)[:]=getattr(src,name)
            dst.rgba[:]=[.45,.75,1.,.32];dst.matid=-1;dst.transparent=1
            dst.emission=.15;dst.reflectance=0.;dst.label=''
            scene.ngeom+=1
        def marker(kind,size,pos,rgba,end=None):
            if scene.ngeom>=scene.maxgeom:raise RuntimeError('Prediction overlay scene capacity exceeded')
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),
                               np.eye(3).ravel(),np.asarray(rgba,dtype=np.float32))
            if end is not None:mujoco.mjv_connector(geom,kind,size[0],pos,end)
            scene.ngeom+=1
        for arm in range(2):
            for a,b in dashed_segments(self.wrists[:,arm],dash=.012,gap=.008):
                if np.linalg.norm(a-b)>1e-8:
                    marker(mujoco.mjtGeom.mjGEOM_CAPSULE,[.003,0,0],a,[.45,.75,1,1],b)
            marker(mujoco.mjtGeom.mjGEOM_SPHERE,[.009]*3,self.wrists[-1,arm],[.45,.75,1,1])
            # Wrist origins are inside the gripper mesh: a cross extends beyond
            # that surface so the reference marker remains visible.
            for axis in np.eye(3)*.045:
                marker(mujoco.mjtGeom.mjGEOM_CAPSULE,[.002,0,0],self.reference_wrists[arm]-axis,
                       [0,.75,.9,1],self.reference_wrists[arm]+axis)

    def label(self, display_time=None):
        now=self.sim.data.time if display_time is None else display_time
        return (f'WM: {self.origin*CONTROL_DT:.2f} -> {self.target_frame*CONTROL_DT:.2f}s '
                f'(+{(self.target_frame-self.origin)*CONTROL_DT:.2f}s; '
                f'{self.target_frame*CONTROL_DT-now:+.2f}s from now; {self.elapsed_ms:.0f} ms compute)')
