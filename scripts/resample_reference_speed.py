"""Uniformly slow a validated reference and regenerate all robot state labels.

Offline only: no policy training, hardware, or change to the original reference.
"""
import argparse
import hashlib
import json

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from openarm_mimic.reference import validate_reference
from openarm_mimic.robot import ROOT, arm_indices, load_robot, nontrivial_contacts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--speed', required=True, type=float, help='Fraction of original speed, in (0, 1)')
    p.add_argument('--output', required=True, help='New directory')
    a = p.parse_args()
    if not 0 < a.speed < 1:
        p.error('This preparation tool supports slowdown only: 0 < speed < 1')
    source = (ROOT / a.source).resolve()
    with np.load(source, allow_pickle=False) as z:
        old = dict(z)
    validate_reference(old)
    if str(old['format_version']) != 'openarm_reference_v2':
        p.error('A validated v2 source is required')
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=False)
    dt = 1 / float(old['fps'])
    intervals = int(np.ceil(float(old['time'][-1]) / a.speed / dt))
    time = np.arange(intervals + 1) * dt
    speed = float(old['time'][-1] / time[-1])
    old_time = time * speed
    old_time[-1] = old['time'][-1]
    path = PchipInterpolator(old['time'], old['joint_pos'], axis=0)
    q = path(old_time)
    qd = np.gradient(q, dt, axis=0, edge_order=2)
    qdd = np.gradient(qd, dt, axis=0, edge_order=2)
    ref = dict(old)
    cfg = json.loads(str(old['config_json']))
    provenance = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      requested_speed=a.speed, effective_speed=speed,
                      original_duration_s=float(old['time'][-1]), duration_s=float(time[-1]),
                      interpolation='Shape-preserving PCHIP of existing joint path; no IK remapping',
                      rounding='Round duration up to next control interval; retain both endpoints',
                      source_phase='Original source_time and human arrays remain on their original source time axis')
    cfg['temporal_resampling'] = provenance
    cfg['title'] = cfg.get('title', 'Reference') + f' ({a.speed:.0%} speed)'
    ref.update(time=time, joint_pos=q, joint_vel=qd, joint_acc=qdd,
               source_time=np.interp(old_time, old['time'], old['source_time']),
               human_source_frame=np.interp(old_time, old['time'], old['human_source_frame']),
               config_json=np.array(json.dumps(cfg)))
    phase = ref['source_time'] - cfg['transition_s']
    active = (phase >= 0) & (phase <= cfg['motion_duration_s'])
    ref['motion_start_frame'] = np.array(np.flatnonzero(active)[0])
    ref['motion_frame_count'] = np.array(np.count_nonzero(active))
    m, _ = load_robot()
    qi, vi, _ = arm_indices(m)
    d = mujoco.MjData(m)
    ref['qpos'] = np.repeat(old['qpos'][0:1], len(time), axis=0)
    ref['qpos'][:, qi] = q
    ref['qpos_vel'] = np.zeros((len(time), m.nv))
    ref['qpos_vel'][:, vi] = qd
    for key, width in [('body_pos_w', 3), ('body_quat_w', 4),
                       ('body_lin_vel_w', 3), ('body_ang_vel_w', 3)]:
        ref[key] = np.empty((len(time), m.nbody - 1, width))
    jp = np.zeros((3, m.nv))
    jr = np.zeros_like(jp)
    collision_frames = []
    for i in range(len(time)):
        d.qpos[:] = ref['qpos'][i]
        d.qvel[:] = ref['qpos_vel'][i]
        mujoco.mj_forward(m, d)
        ref['body_pos_w'][i] = d.xpos[1:]
        ref['body_quat_w'][i] = d.xquat[1:]
        for b in range(1, m.nbody):
            mujoco.mj_jacBody(m, d, jp, jr, b)
            ref['body_lin_vel_w'][i, b-1] = jp @ d.qvel
            ref['body_ang_vel_w'][i, b-1] = jr @ d.qvel
        if nontrivial_contacts(m, d):
            collision_frames.append(i)
    validation = validate_reference(ref)
    # Check sampled resampling error by reconstructing old samples from the new clip.
    reconstructed = PchipInterpolator(old_time, q, axis=0)(old['time'])
    arm = cfg['active_arm']
    wrist = ref['body_names'].tolist().index(f'openarm_{arm}_ee_base_link')
    inactive = slice(7, 14) if arm == 'left' else slice(0, 7)
    report = dict(provenance, frames=len(time), control_hz=float(ref['fps']), active_arm=arm,
                  validation=validation, sampled_geometry_collision_frames=len(collision_frames),
                  first_collision_frames=collision_frames[:20],
                  endpoint_error_rad=float(np.max(abs(q[[0, -1]] - old['joint_pos'][[0, -1]]))),
                  reconstructed_original_samples_max_error_rad=float(np.max(abs(reconstructed-old['joint_pos']))),
                  inactive_arm_range_rad=float(np.max(np.ptp(q[:, inactive], axis=0))),
                  peak_joint_speed_rad_s=float(np.max(abs(qd))),
                  peak_joint_acceleration_rad_s2=float(np.max(abs(qdd))),
                  peak_joint_speed_by_axis_rad_s=np.max(abs(qd), axis=0).tolist(),
                  peak_active_wrist_speed_m_s=float(np.max(np.linalg.norm(ref['body_lin_vel_w'][:, wrist], axis=1))),
                  original_peak_joint_speed_rad_s=float(np.max(abs(old['joint_vel']))),
                  original_peak_joint_acceleration_rad_s2=float(np.max(abs(old['joint_acc']))),
                  original_peak_active_wrist_speed_m_s=float(np.max(np.linalg.norm(old['body_lin_vel_w'][:, wrist], axis=1))),
                  hardware_validated=False, policy_trained=False)
    if collision_frames or report['endpoint_error_rad'] > 1e-10 or report['inactive_arm_range_rad'] > 1e-8:
        raise RuntimeError(f'Resampling audit failed: {report}')
    np.savez_compressed(out / 'reference.npz', **ref)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    report['reference_sha256'] = hashlib.sha256((out / 'reference.npz').read_bytes()).hexdigest()
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
