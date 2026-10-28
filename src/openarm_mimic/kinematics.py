"""OpenArm fixed-base adaptation of TrackerLab differentiable FK.
See third_party/NOTICE.txt and retained MIT licenses for attribution.
"""
import mujoco
import numpy as np
import torch
from .robot import ARM_NAMES, load_robot
from .third_party.rotations import axis_angle_to_mat, quat_to_mat

TRACKED_BODIES = [f'openarm_{side}_{link}' for side in ('left','right')
                  for link in ('link3','link4','ee_base_link')]

class OpenArmKinematics:
    def __init__(self, device='cpu', dtype=torch.float32, tracked=TRACKED_BODIES):
        m,_=load_robot()
        self.body_names=[m.body(i).name for i in range(m.nbody)]
        self.body_names[0]='world'
        self.parent=[-1]+m.body_parentid[1:].tolist()
        self.joint_idx=[];axes=[];anchors=[];qrefs=[]
        for b in range(m.nbody):
            count=int(m.body_jntnum[b]);j=int(m.body_jntadr[b])
            if count>1 or (count and m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError('Only fixed and single-hinge bodies supported')
            name=m.joint(j).name if count else ''
            if count and name not in ARM_NAMES and 'finger_joint' not in name:
                raise ValueError(f'Unexpected uncontrolled joint {name}')
            self.joint_idx.append(ARM_NAMES.index(name) if name in ARM_NAMES else -1)
            axes.append(m.jnt_axis[j].copy() if count else np.zeros(3))
            anchors.append(m.jnt_pos[j].copy() if count else np.zeros(3))
            qrefs.append(m.qpos0[m.jnt_qposadr[j]] if count else 0.)
        self.axes=torch.tensor(np.array(axes),dtype=dtype,device=device)
        self.anchors=torch.tensor(np.array(anchors),dtype=dtype,device=device)
        self.qrefs=torch.tensor(qrefs,dtype=dtype,device=device)
        self.off_p=torch.tensor(m.body_pos.copy(),dtype=dtype,device=device)
        self.off_R=quat_to_mat(torch.tensor(m.body_quat.copy(),dtype=dtype,device=device))
        assert sorted(j for j in self.joint_idx if j>=0)==list(range(14))
        self.tracked_idx=[self.body_names.index(n) for n in tracked]
        self.num_bodies=m.nbody

    def forward(self, q, qd, all_bodies: bool = False):
        """Positions, orientations, origin linear velocities and angular velocities.

        q, qd: (..., 14), left joints 1..7 then right joints 1..7.
        Grippers are fixed at zero for this FK; tracked arm bodies are unaffected.
        """
        p = torch.zeros(q.shape[:-1]+(3,), dtype=q.dtype, device=q.device)
        R = torch.eye(3, dtype=q.dtype, device=q.device).expand(q.shape[:-1]+(3,3))
        v, w = torch.zeros_like(p), torch.zeros_like(p)
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            if j < 0:
                Ri, wi = R_off, W[par]
                vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            else:
                Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j] - self.qrefs[i])
                anchor_w = pi + (R_off @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                arm_w = (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                pi = anchor_w - arm_w
                axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                wi = W[par] + axis_w * qd[..., j : j + 1]
                vi = (
                    V[par] + torch.cross(W[par], anchor_w - P[par], dim=-1)
                    - torch.cross(wi, arm_w, dim=-1)
                )
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
