"""Read-only reference and phase-aligned human overlays for the MuJoCo viewer."""
import json
import mujoco
import numpy as np
from openarm_mimic.robot import ROOT
from mujoco_source_human import SourceHuman, dashed_segments
from trackerlab_supermimic.motion import CONTROL_DT


def sample(values, frame):
    frame=float(np.clip(frame,0,len(values)-1))
    lo=int(frame);hi=min(lo+1,len(values)-1)
    return values[lo]*(1-(frame-lo))+values[hi]*(frame-lo)


class ReferencePreview:
    def __init__(self, sim, anchor_human=False):
        self.sim=sim
        self.anchor_human=anchor_human
        self.data=mujoco.MjData(sim.model)
        self.scene=mujoco.MjvScene(sim.model,maxgeom=2000)
        self.option=mujoco.MjvOption();self.option.geomgroup[3]=0;self.option.sitegroup[:]=0
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.pert=mujoco.MjvPerturb()
        r=sim.reference
        self.path=r['body_pos_w'][np.linspace(0,len(r['qpos'])-1,min(200,len(r['qpos'])),dtype=int)][:,sim.ref_wrists]
        self.dashes=[list(dashed_segments(self.path[:,arm])) for arm in range(2)]
        self.source=None
        self.human_available=all(k in r for k in ('human_joint_names','human_upper_positions','human_source_frame'))
        if self.human_available:
            names=r['human_joint_names'].tolist()
            edges=[('Spine1','Neck'),('Neck','LeftArm'),('LeftArm','LeftForeArm'),
                   ('LeftForeArm','LeftHand'),('LeftHand','LeftHandIndex1'),
                   ('Neck','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),
                   ('RightHand','RightHandIndex1')]
            self.edges=[(names.index(a),names.index(b),'Right' in b) for a,b in edges]
            h=r['human_upper_positions']
            arm=sum(np.linalg.norm(h[:,names.index(a)]-h[:,names.index(b)],axis=1)
                    for a,b in [('RightArm','RightForeArm'),('RightForeArm','RightHand')])
            # Source BVH length units are unknown. Normalize ONLY the display
            # to a 0.55 m arm length; this is not a metric human measurement.
            self.human_scale=.55/float(np.median(arm))
            # Fixed projection bounds make the inset stable through playback.
            selected=sorted({i for a,b,_ in self.edges for i in (a,b)})
            projected=h[:,selected][:,:,[1,2]]
            self.inset_low=projected.min(axis=(0,1));self.inset_high=projected.max(axis=(0,1))
            cfg=json.loads(str(r['config_json']))
            path=ROOT/cfg['source']
            if path.is_file():
                self.source=SourceHuman(path)
                if self.anchor_human:
                    self.source.metadata['display_root']='Pelvis XY anchored; articulation, heading and vertical motion retained'
                self.source_duration=float(cfg['motion_duration_s'])
                self.transition=float(cfg['transition_s'])
        self.update()

    def update(self, display_time=None):
        s=self.sim
        self.frame=float(np.clip((s.data.time if display_time is None else display_time)/CONTROL_DT,
                                 0,s.motion.num_frames-1))
        self.data.qpos[:]=sample(s.reference['qpos'],self.frame)
        mujoco.mj_forward(s.model,self.data)
        mujoco.mjv_updateScene(s.model,self.data,self.option,self.pert,self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        if self.human_available:
            self.human_frame=float(sample(s.reference['human_source_frame'],self.frame))
            self.human=sample(s.reference['human_upper_positions'],self.human_frame)*self.human_scale
            if self.source is not None:
                phase=float(sample(s.reference['source_time'],self.frame))-self.transition
                self.source_frame=np.clip(phase/self.source_duration,0,1)*(len(self.source.positions)-1)
                self.source_pose=sample(self.source.positions,self.source_frame)
                if self.anchor_human:
                    self.source_pose[:,:2]+=self.source.positions[0,0,:2]-self.source_pose[0,:2]

    def inset_image(self,width=240,height=155):
        """Screen-fixed small front view of the existing torso-local skeleton."""
        from PIL import Image,ImageDraw
        im=Image.new('RGB',(width,height),(245,248,251));draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,width-1,height-1),outline=(140,155,170))
        draw.text((8,7),'Upper body (torso-local)',fill=(35,45,55))
        points=self.human[:,[1,2]]/self.human_scale
        center=(self.inset_low+self.inset_high)/2
        scale=min((width-28)/max(self.inset_high[0]-self.inset_low[0],1e-8),
                  (height-48)/max(self.inset_high[1]-self.inset_low[1],1e-8))
        points=(points-center)*[scale,-scale]+[width/2,(height+24)/2]
        for a,b,right in self.edges:
            color=(205,48,52) if right else (45,130,65)
            draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=3)
            x,y=points[b];draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        return np.asarray(im)

    @staticmethod
    def marker(scene, start, end=None, color=(1,.48,.08,1), radius=.003):
        if scene.ngeom>=scene.maxgeom:raise RuntimeError('Reference overlay scene capacity exceeded')
        geom=scene.geoms[scene.ngeom]
        kind=mujoco.mjtGeom.mjGEOM_SPHERE if end is None else mujoco.mjtGeom.mjGEOM_CAPSULE
        mujoco.mjv_initGeom(geom,kind,np.full(3,radius),np.asarray(start),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
        if end is not None:mujoco.mjv_connector(geom,kind,radius,start,end)
        scene.ngeom+=1

    def add_to(self, scene, show_reference=True, show_human=False):
        if show_reference:
            model=self.sim.model
            for i in range(self.scene.ngeom):
                src=self.scene.geoms[i]
                if src.objtype!=mujoco.mjtObj.mjOBJ_GEOM or src.objid<0:continue
                if model.geom_bodyid[src.objid]==0 or model.geom_group[src.objid]!=2:continue
                if scene.ngeom>=scene.maxgeom:raise RuntimeError('Reference overlay scene capacity exceeded')
                dst=scene.geoms[scene.ngeom]
                for name in ('type','dataid','objtype','objid','category','matid','texcoord',
                             'segid','modelrbound','specular','shininess','reflectance','emission'):
                    setattr(dst,name,getattr(src,name))
                for name in ('size','pos','mat'):getattr(dst,name)[:]=getattr(src,name)
                dst.rgba[:]=[1,.48,.08,.28];dst.matid=-1;dst.transparent=1
                dst.emission=.15;dst.reflectance=0.;dst.label=''
                scene.ngeom+=1
            for arm in range(2):
                for a,b in self.dashes[arm]:self.marker(scene,a,b,radius=.0015)
                self.marker(scene,self.data.xpos[self.sim.wrists[arm]],radius=.012)
        if show_human and self.source is not None:
            marker_scale=self.source.display_height/1.25
            for b,a in enumerate(self.source.parents):
                if a<0:continue
                name=self.source.names[b]
                color=(.85,.18,.22,1) if name.startswith(('Right','RThumb','RHip')) else (
                    (.22,.55,.25,1) if name.startswith(('Left','LThumb','LHip')) else (.25,.32,.43,1))
                if np.linalg.norm(self.source_pose[b]-self.source_pose[a])<1e-8:continue
                self.marker(scene,self.source_pose[a],self.source_pose[b],color,radius=.009*marker_scale)
                self.marker(scene,self.source_pose[b],color=color,radius=.014*marker_scale)
            head=self.source_pose[self.source.names.index('Head')]
            self.marker(scene,head,color=(.25,.32,.43,1),radius=.04*marker_scale)

    def label(self, show_reference=True, show_human=False):
        parts=['Orange: reference + dashed wrist path' if show_reference else 'Reference OFF']
        if show_human and self.human_available:
            phase=self.human_frame*CONTROL_DT
            held=self.human_frame<=0 or self.human_frame>=len(self.sim.reference['human_upper_positions'])-1
            native=f' | BVH t={self.source_frame*self.source.frame_time:.2f}s' if self.source is not None else ' | BVH unavailable'
            parts.append(f'Human phase {phase:.2f}s'+(' (held)' if held else '')+native+' | normalized size'+(' | root XY anchored' if self.anchor_human else ''))
        return '\n'.join(parts)
