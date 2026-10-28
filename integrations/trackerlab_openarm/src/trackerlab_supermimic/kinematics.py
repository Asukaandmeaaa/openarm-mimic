"""OpenArm FK with the upstream state tuple. Root is fixed, fingers remain in state."""
import torch
from openarm_mimic.kinematics import OpenArmKinematics
from openarm_mimic.robot import load_robot,arm_indices
class G1Kinematics:
    """Compatibility name only: all assets and joint names are OpenArm."""
    def __init__(self,device='cpu'):
        m,_=load_robot();self.arm=arm_indices(m)[0].tolist();self.fk=OpenArmKinematics(device)
        self.q_lo=torch.tensor(m.jnt_range[:,0],device=device,dtype=torch.float32)
        self.q_hi=torch.tensor(m.jnt_range[:,1],device=device,dtype=torch.float32)
        self.motor_mask=torch.zeros(18,device=device);self.motor_mask[self.arm]=1
    def forward(self,p,R,v,w,q,qd):return self.fk.forward(q[...,self.arm],qd[...,self.arm])
