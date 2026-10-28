"""OpenArm reference NPZ adapter; retain upstream goal/ref-state/end-frame semantics."""
import numpy as np
import torch
from .robot import REFERENCE
from .rotations import mat_to_6d,yaw_mat
CONTROL_DT=.02
BIN_FRAMES=50
GOAL_FRAME_DIM=51  # Original 15 root features (constant) plus 18 q and 18 qd.
class MotionLib:
    def __init__(self,path,kin,device):
        z=np.load(path,allow_pickle=False)
        from openarm_mimic.reference import validate_reference
        self.validation=validate_reference(z)
        if float(z['fps'])!=50:raise ValueError('Reference must be 50 Hz')
        from pathlib import Path
        self.files=[str(Path(path).resolve())];self.names=['wave_right'];self.q=torch.tensor(z['qpos'],device=device,dtype=torch.float32)
        self.qd=torch.zeros_like(self.q);self.qd[:,kin.arm]=torch.tensor(z['joint_vel'],device=device,dtype=torch.float32)
        n=len(self.q);self.num_frames=n
        self.clip_start=torch.tensor([0],device=device);self.clip_len=torch.tensor([n],device=device);self.clip_end=self.clip_len-1
        self.clip_id=torch.zeros(n,device=device,dtype=torch.long)
        self.root_p=torch.zeros(n,3,device=device);self.root_R=torch.eye(3,device=device).expand(n,3,3)
        self.root_v=torch.zeros(n,3,device=device);self.root_w=torch.zeros(n,3,device=device)
        self.root_v_local=self.root_v;self.root_w_local=self.root_w
        with torch.no_grad():self.body_p,self.body_R,self.body_v,self.body_w=kin.forward(*self.ref_state(torch.arange(n,device=device)))
        self.bin_start=torch.arange(0,n-1,BIN_FRAMES,device=device);self.num_bins=len(self.bin_start)
        self.bin_clip=torch.zeros_like(self.bin_start);self.frame_bin=(torch.arange(n,device=device)//BIN_FRAMES).clamp_max(self.num_bins-1)
    @staticmethod
    def from_dir(path,kin,device):
        from pathlib import Path
        path=Path(path)
        if path.is_dir():path=path/'wave_right_v1.npz'
        return MotionLib(path,kin,device)
    def future_frames(self,f,K):
        end=self.clip_end[self.clip_id[f]]
        return torch.minimum(f.unsqueeze(-1)+torch.arange(1,K+1,device=f.device),end.unsqueeze(-1))
    def goal(self,f,K):
        fut=self.future_frames(f,K)
        H=yaw_mat(self.root_R[f]);Ht=H.transpose(-1,-2).unsqueeze(1)
        dp=(Ht@(self.root_p[fut]-self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot=mat_to_6d(Ht@self.root_R[fut])
        return torch.cat((dp,rot,self.root_v_local[fut],self.root_w_local[fut],self.q[fut],self.qd[fut]*.1),-1).reshape(f.shape[0],-1)
    def ref_state(self,f):return self.root_p[f],self.root_R[f],self.root_v[f],self.root_w[f],self.q[f],self.qd[f]
