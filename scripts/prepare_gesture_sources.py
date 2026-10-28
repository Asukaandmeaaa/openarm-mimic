"""Prepare pinned CMU gesture clips without changing any original motion data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from openarm_mimic.robot import ROOT

GESTURES={
    'wipe_window':dict(clip='143_26',side='left',crop=None,duration=8.,title='Window wiping (left arm)'),
    'draw_circle':dict(clip='143_27',side='right',crop=[350,600],duration=5.,title='Circular wiping / drawing loops (right arm)'),
    'handshake':dict(clip='141_23',side='right',crop=None,duration=4.,title='Reach and shake hands'),
    'high_five':dict(clip='141_22',side='right',crop=None,duration=4.,title='Raise hand for high five'),
}


def prepare(raw,output):
    output.mkdir(parents=True,exist_ok=False)
    configs=[]
    for name,description in GESTURES.items():
        clip=description['clip'];source=raw/f'{clip}.bvh';text=source.read_text()
        lines=text.splitlines();header=next(i for i,l in enumerate(lines) if l.startswith('Frame Time:'))
        rows=lines[header+1:];channels=np.array([[float(v) for v in r.split()] for r in rows])
        amc=raw/f'{clip}.amc'
        if not amc.exists():raise FileNotFoundError(f'Need original AMC to verify reference-pose removal: {amc}')
        roots=np.array([[float(v) for v in l.split()[1:]] for l in amc.read_text().splitlines() if l.startswith('root ')])
        if len(channels)!=len(roots)+1:raise ValueError('Unexpected BVH/AMC frame count')
        error=float(np.max(abs(channels[1:,:6]-roots[:,[0,1,2,5,4,3]])))
        if error>=.001:raise ValueError('BVH/AMC root alignment failed')
        first,last=description['crop'] or [0,len(roots)]
        selected=rows[1+first:1+last]
        if not 0<=first<last<=len(roots):raise ValueError('Invalid crop')
        clean=output/f'{name}_source.bvh'
        prefix=lines[:header+1];prefix[header-1]=f'Frames: {len(selected)}'
        clean.write_text('\n'.join(prefix+selected)+'\n')
        cfg=dict(name=name,title=description['title'],source=str(clean.relative_to(ROOT)),
                 original_bvh=str(source.relative_to(ROOT)),original_amc=str(amc.relative_to(ROOT)),
                 original_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 source_sha256=hashlib.sha256(clean.read_bytes()).hexdigest(),
                 original_frame_time_s=float(re.search(r'Frame Time:\s*(\S+)',text)[1]),
                 crop_clean_frames_exclusive=[first,last],removed_leading_reference_frames=1,
                 original_bvh_frames_inclusive=[first+1,last],amc_root_max_error=error,
                 active_arm=description['side'],control_hz=50,motion_duration_s=description['duration'],
                 transition_s=1.2,playback_time_scale=1.,gripper_left=.6 if description['side']=='left' else 0.,
                 gripper_right=-.6,standby_left=[0.,-.12,0.,.35,0.,0.,0.],
                 standby_right=[0.,.12,0.,.35,0.,0.,0.],joint_margin_rad=.02,
                 review_limits=dict(joint_speed_rad_s=1.5,joint_acceleration_rad_s2=8.,wrist_target_rmse_m=.04),
                 note='Human source candidate; no learned policy. Circle is a cropped real circular-wiping segment, not an ideal synthesized circle.')
        (output/f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n');configs.append(cfg)
    (output/'source_preparation.json').write_text(json.dumps(configs,indent=2)+'\n')
    return configs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',default='data/raw/cmu/gesture_candidates_20261027')
    p.add_argument('--output',default='data/processed/gesture_candidates_20261027/sources')
    args=p.parse_args();print(json.dumps(prepare(ROOT/args.raw,ROOT/args.output),indent=2))
