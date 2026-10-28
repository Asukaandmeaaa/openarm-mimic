"""Compare reused torch FK and Jacobians against MuJoCo before using BPTT rewards."""
import json
import numpy as np
import torch,mujoco
from openarm_mimic.robot import ROOT,load_robot,arm_indices,set_arms
from openarm_mimic.kinematics import OpenArmKinematics
m,mapper=load_robot();d=mujoco.MjData(m);qi,vi,limits=arm_indices(m);rng=np.random.default_rng(23)
q=rng.uniform(limits[:,0]+.03,limits[:,1]-.03,size=(24,14));qd=rng.normal(0,.3,size=q.shape)
fk=OpenArmKinematics('cpu',torch.float64);qt=torch.tensor(q,requires_grad=True);vt=torch.tensor(qd)
p,r,v,w=fk.forward(qt,vt)
errors=[]
for i in range(len(q)):
 set_arms(mapper,d.qpos,q[i]);d.qvel[:]=0;d.qvel[vi]=qd[i];mujoco.mj_forward(m,d)
 for k,b in enumerate(fk.tracked_idx):
  jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,jp,jr,b)
  errors.append([np.max(abs(p[i,k].detach().numpy()-d.xpos[b])),np.max(abs(r[i,k].detach().numpy()-d.xmat[b].reshape(3,3))),np.max(abs(v[i,k].detach().numpy()-jp@d.qvel)),np.max(abs(w[i,k].detach().numpy()-jr@d.qvel))])
errors=np.max(errors,axis=0);assert np.max(errors)<1e-8,errors
one=torch.tensor(q[0],requires_grad=True)
j=torch.autograd.functional.jacobian(lambda x:fk.forward(x[None],torch.zeros_like(x)[None])[0][0,-1],one).numpy()
set_arms(mapper,d.qpos,q[0]);mujoco.mj_forward(m,d);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,fk.tracked_idx[-1])
jerr=float(np.max(abs(j-jp[:,vi])));assert jerr<1e-8
# Verify the float32 path, including finite gradients at zero angles. Use CUDA
# when available, while keeping this check runnable on a CPU-only workstation.
device='cuda' if torch.cuda.is_available() else 'cpu'
fkg=OpenArmKinematics(device);z=torch.zeros((4,14),device=device,requires_grad=True)
gp,gr,gv,gw=fkg.forward(z,torch.zeros_like(z));(gp.square().sum()+gr.sum()).backward();assert torch.isfinite(z.grad).all()
report=dict(random_poses=24,max_position_error_m=float(errors[0]),max_rotation_matrix_error=float(errors[1]),max_linear_velocity_error=float(errors[2]),max_angular_velocity_error=float(errors[3]),max_wrist_jacobian_error=jerr,float32_zero_angle_gradients_finite=True,gradient_device=device)
(ROOT/'reports').mkdir(parents=True,exist_ok=True)
(ROOT/'reports/openarm_fk_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
