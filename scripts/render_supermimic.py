"""Render recorded physical evaluation, without running or resetting a policy."""
import argparse
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
from evaluate_wave_pd import render
from openarm_mimic.robot import load_robot
p=argparse.ArgumentParser(description=__doc__);p.add_argument('trajectory');p.add_argument('--output-prefix');args=p.parse_args()
z=np.load(args.trajectory,allow_pickle=False);ref=np.load(str(z['reference']),allow_pickle=False)
model,_=load_robot();out=args.output_prefix or str(Path(args.trajectory).with_name('comparison'))
render(ref,z['qpos'],model,output_prefix=out,controller_label='PhysX '+str(z['controller'])+' (recorded dynamics)')
print(out+'.mp4')
