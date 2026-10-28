"""OpenArm physics/task adapter. Sampling, episode accounting and transition contract are upstream.
Only the existing asset/PD backend is reused; its old reward and step loop are not called.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from isaaclab.sim import SimulationContext
from .mdp import MAX_STEP_REWARD, NUM_A, XI_DIM, flatten_state
from .motion import CONTROL_DT
from .robot import action_scales, default_joint_pos_list, full_position_target

@contextmanager
def launch_simulation(sim_cfg,args,app_launcher=None):
    try:
        yield sim_cfg  # The existing asset backend owns the single SimulationContext.
    finally:
        import sys
        active_error = sys.exc_info()[0] is not None
        SimulationContext.clear_instance()
        if app_launcher is not None:
            try:app_launcher.app.close()
            except SystemExit:
                if not active_error:pass

def make_sim(device,args=None,task_cfg=None):
    return SimpleNamespace(device=device,dt=task_cfg.physics_dt,render_interval=task_cfg.control_decimation)

class TrackingSim:
    def __init__(self,sim_cfg,num_envs,motion,mdp,gamma=.99,eval_mode=False,task_cfg=None,**kwargs):
        from openarm_mimic.robot import load_robot
        from openarm_mimic.supermimic.env import TrackingSim as AssetBackend
        self.motion,self.mdp=motion,mdp
        self.num_envs,self.device=num_envs,sim_cfg.device
        self.physics_dt,self.control_decimation=sim_cfg.dt,sim_cfg.render_interval
        self.eval_mode=eval_mode;self.reset_noise=getattr(task_cfg,'reset_noise',False) and not eval_mode
        self.gamma=gamma
        if not 0<gamma<1:raise ValueError('gamma must lie between zero and one')
        self.max_episode_steps=2**31-1 if eval_mode else getattr(task_cfg,'max_episode_steps',500)
        model,_=load_robot()
        names=[model.joint(i).name for i in range(model.njnt)]
        task=SimpleNamespace(device=self.device,model=model,names=names,low=mdp.kin.q_lo[mdp.kin.arm],high=mdp.kin.q_hi[mdp.kin.arm],fullq=motion.q,fullqd=motion.qd,q=motion.q[:,mdp.kin.arm],length=motion.num_frames)
        cfg=dict(num_envs=num_envs,physics_dt=self.physics_dt,control_dt=CONTROL_DT,episode_steps=self.max_episode_steps)
        self.backend=AssetBackend(task,cfg)
        self.sim,self.scene,self.robot=self.backend.sim,self.backend.scene,self.backend.robot
        self.joint_ids,self.arm_ids=self.backend.ids,self.backend.armids
        self.default_q=torch.tensor(default_joint_pos_list(motion.files[0]),device=self.device)
        self.action_scale=torch.tensor(action_scales(motion.files[0]),device=self.device)
        self.f=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.motion_id=torch.zeros_like(self.f);self.ep_step=torch.zeros_like(self.f,dtype=torch.int32)
        self.ep_return=torch.zeros(num_envs,device=self.device);self.ep_discounted_return=torch.zeros_like(self.ep_return)
        self.a_prev=torch.zeros(num_envs,NUM_A,device=self.device);self.xi=torch.zeros(num_envs,XI_DIM,device=self.device)
        self.done_eval=torch.zeros(num_envs,dtype=torch.bool,device=self.device);self.failed_eval=torch.zeros_like(self.done_eval)
        self.bin_returns=torch.zeros(motion.num_bins,device=self.device)
        self.max_bin_return=MAX_STEP_REWARD/(1-gamma)
        self.bin_past_weights=torch.full_like(self.bin_returns,1/motion.num_bins)
        self.reset_bin=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.reset()

    def state(self):
        x=self.backend.state();q,qd=x.split(18,-1)
        z=torch.zeros(self.num_envs,3,device=self.device)
        return z,torch.eye(3,device=self.device).expand(self.num_envs,3,3),z.clone(),z.clone(),q,qd

    @torch.no_grad()
    def _reset(self,env_ids,frame_ids=None):
        if not env_ids.numel():return
        if frame_ids is None:clip_ids,frame_ids=self._sample_frames(env_ids)
        else:
            frame_ids=frame_ids.to(self.device,dtype=torch.long)
            clip_ids=self.motion.clip_id[frame_ids]
        self.motion_id[env_ids]=clip_ids;self.f[env_ids]=frame_ids
        self.reset_bin[env_ids]=self.motion.frame_bin[frame_ids]
        self.ep_step[env_ids]=0;self.ep_return[env_ids]=0;self.ep_discounted_return[env_ids]=0
        self.a_prev[env_ids]=0;self.done_eval[env_ids]=False;self.failed_eval[env_ids]=False
        q=self.motion.q[frame_ids].clone();qd=self.motion.qd[frame_ids].clone()
        if self.reset_noise:
            q[:,self.mdp.kin.arm]+=(2*torch.rand_like(q[:,self.mdp.kin.arm])-1)*.1
        q=q.clamp(self.mdp.kin.q_lo,self.mdp.kin.q_hi)
        self.robot.write_joint_state_to_sim(q,qd,joint_ids=self.joint_ids,env_ids=env_ids)
        self.robot.reset(env_ids)
        self.robot.set_joint_position_target(q,joint_ids=self.joint_ids,env_ids=env_ids)
        self.scene.write_data_to_sim();self.sim.forward();self.scene.update(self.physics_dt)

    def reset(self,env_ids=None,frame_ids=None):
        if env_ids is None:env_ids=torch.arange(self.num_envs,device=self.device)
        self._reset(env_ids,frame_ids)

    def action_target(self,a):
        arm=self.mdp.kin.arm
        return (self.default_q[arm]+a.clamp(-20,20)*self.action_scale).clamp(self.mdp.kin.q_lo[arm],self.mdp.kin.q_hi[arm])

    def _sample_frames(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(env_ids)
        soft_probs = torch.softmax(-25.0 * self.bin_returns / self.max_bin_return, dim=0)
        self.bin_past_weights.mul_(0.999).add_(soft_probs, alpha=0.001)
        past_probs = self.bin_past_weights / self.bin_past_weights.sum().clamp_min(1e-8)
        probabilities = 0.5 / self.motion.num_bins + 0.4 * soft_probs + 0.1 * past_probs
        bins = torch.multinomial(probabilities, count, replacement=True)

        starts = self.motion.bin_start[bins]
        clip_ids = self.motion.bin_clip[bins]
        clip_ends = self.motion.clip_end[clip_ids]
        starts = torch.minimum(starts, clip_ends - 1)
        end_exclusive = torch.minimum(self.motion.bin_start[bins] + 50, clip_ends)
        spans = (end_exclusive - starts).clamp_min(1)
        offsets = torch.floor(torch.rand(count, device=self.device) * spans).long()
        return clip_ids, starts + offsets

    def _update_bin_returns(self, finished: torch.Tensor, terminated: torch.Tensor):
        lengths = self.ep_step[finished].float()
        # Clip ends and time limits retain a tail value; physical failures do not.
        bootstrap = torch.where(
            terminated[finished],
            0.0,
            (self.gamma**lengths)
            * (self.ep_return[finished] / lengths.clamp_min(1))
            / (1.0 - self.gamma),
        )
        returns = self.ep_discounted_return[finished] + bootstrap
        bins = self.reset_bin[finished]
        sums = torch.zeros_like(self.bin_returns).index_add_(0, bins, returns)
        counts = torch.bincount(bins, minlength=self.motion.num_bins).to(self.bin_returns.dtype)
        observed = counts > 0
        means = sums / counts.clamp_min(1)
        self.bin_returns[observed] = 0.8 * self.bin_returns[observed] + 0.2 * means[observed]

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        S0 = self.state()
        s = flatten_state(S0)
        a_prev = self.a_prev.clone()
        policy_action = action.clone()
        a = torch.clamp(action, -20.0, 20.0)
        if self.eval_mode:
            a = torch.where(self.done_eval[:, None], torch.zeros_like(a), a)
        active = ~self.done_eval if self.eval_mode else torch.ones_like(self.done_eval)
        target = self.action_target(a)
        full_target = full_position_target(self.motion.q[self.f],target,self.mdp.kin.arm)
        self.robot.set_joint_position_target(full_target,joint_ids=self.joint_ids)
        for _ in range(self.control_decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

        S1 = self.state()
        end = self.motion.clip_end[self.motion.clip_id[self.f]]
        f_next = torch.minimum(self.f + 1, end)
        terminated = self.mdp.termination(S1, f_next) & active
        timeout = self.ep_step + 1 >= self.max_episode_steps
        truncated = ((self.f + 1 >= end) | timeout) & ~terminated & active
        done = terminated | truncated
        if self.eval_mode:
            done = done & ~self.done_eval
        reward, reward_terms = self.mdp.reward(S1, f_next, a, a_prev, terms=True)
        reward = torch.where(self.done_eval, torch.zeros_like(reward), reward)
        self.ep_return += reward
        self.ep_discounted_return += (self.gamma**self.ep_step.float()) * reward
        self.ep_step += (~self.done_eval).to(self.ep_step.dtype)
        self.f = f_next
        self.a_prev = a
        transition = {
            "s": s,
            "s1": flatten_state(S1),
            "a": a,
            "policy_action": policy_action,
            "a_prev": a_prev,
            "xi": self.xi.clone(),
            "r": reward,
            "reward_terms": reward_terms,
            "f": f_next - 1,
            "motion_id": self.motion_id.clone(),
            "ep_step": self.ep_step - 1,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
        }

        if self.eval_mode:
            self.done_eval |= done
            self.failed_eval |= terminated
        else:
            finished = done.nonzero(as_tuple=False).squeeze(-1)
            if finished.numel():
                self._update_bin_returns(finished, terminated)
                transition["ep_len"] = torch.where(done, self.ep_step.float(), torch.zeros_like(self.ep_return))
                transition["ep_ret_norm"] = torch.where(
                    done,
                    self.ep_return / (self.ep_step.float() * MAX_STEP_REWARD).clamp_min(1e-8),
                    torch.zeros_like(self.ep_return),
                )
                transition["term"] = terminated
                self._reset(finished)
                transition["done"] = done
        return transition
