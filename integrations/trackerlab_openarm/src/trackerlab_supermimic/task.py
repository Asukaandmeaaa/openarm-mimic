"""OpenArm task configuration; compatibility class name retained for upstream training."""

from isaaclab.utils import configclass


@configclass
class SuperMimicG1TaskCfg:
    num_envs: int = 4096
    env_spacing: float = 2.0
    physics_dt: float = 0.002
    control_decimation: int = 10
    control_dt: float = 0.02
    max_episode_steps: int = 500
    reset_noise: bool = True
    gpu_max_rigid_patch_count: int = 2**21
    observation_mode: str = "fixed_base50"
    observation_history: int = 10
    goal_frames: int = 10
    imagination_horizon: int = 32
    replay_buffer_steps: int = 1000
    world_model_window: int = 8
    world_model_updates_per_iteration: int = 10
    world_model_batch_size: int = 2048
    policy_batch_size: int = 4096
    friction_range: tuple[float, float] = (1.0, 1.0)
    joint_offset_range: float = 0.0
    torso_com_offset_range: tuple[float, float, float] = (0.0, 0.0, 0.0)
