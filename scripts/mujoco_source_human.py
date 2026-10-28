"""Source BVH display before robot retargeting; no robot control or hardware."""
import hashlib
import re
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from openarm_mimic.robot import ROOT


def dashed_segments(points, dash=.018, gap=.012):
    """Dashes with continuous arc-length phase across polyline vertices."""
    distance=0.;period=dash+gap
    for a,b in zip(points[:-1],points[1:]):
        length=float(np.linalg.norm(b-a))
        if length<1e-10:continue
        offset=0.
        while offset<length-1e-10:
            phase=(distance+offset)%period
            visible=phase<dash
            remaining=(dash if visible else period)-phase
            if remaining<1e-10:remaining=gap if visible else dash;visible=not visible
            end=min(length,offset+remaining)
            if visible:yield a+(b-a)*(offset/length),a+(b-a)*(end/length)
            offset=end
        distance+=length


class SourceHuman:
    def __init__(self, path):
        sys.path.insert(0,str(ROOT/'vendor'))
        from lafan_vendor.extract import read_bvh
        from lafan_vendor.utils import quat_fk
        self.path=path.resolve();text=path.read_text()
        anim=read_bvh(str(path));quats,positions=quat_fk(anim.quats,anim.pos,anim.parents)
        self.names=list(anim.bones);self.parents=list(anim.parents)
        self.frame_time=float(re.search(r'Frame Time:\s*(\S+)',text).group(1))
        # Keep End Sites (head/toes/fingertips) omitted by the vendor reader.
        stack=[];pending=None
        for line in text.split('MOTION')[0].splitlines():
            words=line.strip().split()
            if not words:continue
            if words[0] in ('ROOT','JOINT'):pending=self.names.index(words[1])
            elif words[:2]==['End','Site']:pending='end'
            elif words[0]=='{':stack.append(pending)
            elif words[0]=='}':stack.pop()
            elif words[0]=='OFFSET' and stack[-1]=='end':
                parent=stack[-2];offset=np.array(words[1:],dtype=float)
                tip=positions[:,parent]+Rotation.from_quat(quats[:,parent][:,[1,2,3,0]]).apply(offset)
                positions=np.concatenate([positions,tip[:,None]],axis=1)
                self.names.append(self.names[parent]+'Tip');self.parents.append(parent)
        # One FIXED rigid viewing transform. Preserve all source articulation,
        # root displacement and torso rotation over time (no per-frame alignment).
        def unit(v):return v/np.linalg.norm(v)
        p=positions[0]
        # CMU BVH is Y-up. Initial torso/shoulder tilt must NOT redefine
        # gravity: align heading only, preserving the source's vertical axis.
        z=np.array([0.,1.,0.])
        shoulder=p[self.names.index('LeftArm')]-p[self.names.index('RightArm')]
        y=unit(shoulder-z*np.dot(shoulder,z))
        x=unit(np.cross(y,z));y=np.cross(z,x)
        self.basis=np.stack([x,y,z],axis=-1)
        self.raw_positions=positions
        display=(positions-positions[0,0])@self.basis
        self.display_height=.85
        self.scale=self.display_height/np.ptp(display[:,:,2])
        self.offset=np.array([0,1.05,-.698-display[:,:,2].min()*self.scale])
        self.positions=display*self.scale+self.offset
        self.metadata={'path':str(self.path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'frames':len(positions),'frame_time_s':self.frame_time,
                       'display_scale':self.scale,'display_height':self.display_height,
                       'display_transform':'BVH Y-up to display Z-up; fixed initial heading only, uniform scale and offset; root motion retained',
                       'timing':'phase aligned with reference source_time; not native BVH wall-clock playback'}
