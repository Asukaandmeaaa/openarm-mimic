"""Physical MuJoCo PD baseline, separate from kinematic reference playback."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from openarm_mimic.robot import ROOT,load_robot,arm_indices,set_arms,set_targets,nontrivial_contacts

def main():
    p=argparse.ArgumentParser();p.add_argument('--motion',default='data/processed/wave_right_v1.npz');p.add_argument('--render',action='store_true');p.add_argument('--output-dir');args=p.parse_args()
    out=ROOT/args.output_dir if args.output_dir else ROOT/'reports'
    if args.output_dir:out.mkdir(parents=True,exist_ok=False)
    ref=np.load(ROOT/args.motion,allow_pickle=False);model,mapper=load_robot();data=mujoco.MjData(model);qi,vi,limits=arm_indices(model)
    dt=1/float(ref['fps']);substeps=round(dt/model.opt.timestep);assert abs(substeps*model.opt.timestep-dt)<1e-9
    # Preserve reference gripper targets; previously all PD replays forced zero.
    grip_idx=[int(model.joint(f'openarm_{s}_finger_joint1').qposadr[0]) for s in ('left','right')]
    grippers=ref['qpos'][:,grip_idx]
    data.qpos[:]=ref['qpos'][0];set_targets(mapper,data.ctrl,ref['joint_pos'][0],grippers[0]);mujoco.mj_forward(model,data)
    # Settle under the same plain PD controller; no gravity compensation or state teleportation.
    for _ in range(500):mujoco.mj_step(model,data)
    recorded=[];wrist_errors=[];ori_errors=[];contacts={};contact_steps=0;sat_count=0;steps=0
    for i,target in enumerate(ref['joint_pos']):
        if i:
            set_targets(mapper,data.ctrl,ref['joint_pos'][i-1],grippers[i-1])
            for _ in range(substeps):
                mujoco.mj_step(model,data);mujoco.mj_forward(model,data);steps+=1
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                pairs=nontrivial_contacts(model,data);contact_steps+=bool(pairs)
                for a,b,depth in pairs:
                    key=' | '.join(sorted([a,b]));contacts[key]=min(depth,contacts.get(key,0))
                force=np.abs(data.actuator_force);cap=np.max(np.abs(model.actuator_forcerange),axis=1)
                sat_count+=bool(np.any(force>=cap*.99))
        recorded.append(data.qpos.copy())
        body=ref['body_names'].tolist().index('openarm_right_ee_base_link')
        wrist_errors.append(np.linalg.norm(data.body('openarm_right_ee_base_link').xpos-ref['body_pos_w'][i,body]))
        rr=Rotation.from_quat(ref['body_quat_w'][i,body][[1,2,3,0]]).as_matrix()
        ori_errors.append(Rotation.from_matrix(rr.T@data.body('openarm_right_ee_base_link').xmat.reshape(3,3)).magnitude())
    qactual=np.array(recorded);err=qactual[:,qi]-ref['joint_pos'];wrist_errors=np.array(wrist_errors);ori_errors=np.array(ori_errors)
    np.savez_compressed(out/'wave_right_pd_rollout.npz',time=ref['time'],qpos=qactual,reference_qpos=ref['qpos'],joint_error=err,wrist_error=wrist_errors,orientation_error=ori_errors,reference=str(ROOT/args.motion))
    report=dict(controller='Existing MJCF position servos, plain PD, 50 Hz command hold, 500 Hz simulation, no gravity compensation',frames=len(qactual),duration_s=float(ref['time'][-1]),joint_rmse_rad=float(np.sqrt(np.mean(err**2))),joint_max_error_rad=float(np.abs(err).max()),wrist_rmse_m=float(np.sqrt(np.mean(wrist_errors**2))),wrist_max_error_m=float(wrist_errors.max()),wrist_orientation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(ori_errors**2)))),contact_substeps=contact_steps,total_substeps=steps,penetrating_contact_pairs=contacts,actuator_saturation_substep_fraction=sat_count/max(steps,1),actual_joint_limit_violation_rad=float(np.maximum(np.maximum(qactual[:,qi]-limits[:,1],limits[:,0]-qactual[:,qi]),0).max()),scope='Dynamic baseline only; no learned policy. Contact report includes all nontrivial mesh contacts.')
    report['gripper_rmse_rad']=np.sqrt(np.mean((qactual[:,grip_idx]-grippers)**2,axis=0)).tolist()
    report['reference']=str(ROOT/args.motion)
    (out/'wave_right_pd.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if args.render:render(ref,qactual,model,output_prefix=out/'wave_right_comparison')

def render(ref,qactual,model,output_prefix=None,controller_label="OpenArm plain PD (physical simulation)"):
    prefix=Path(output_prefix) if output_prefix else ROOT/"reports/wave_right_comparison"
    prefix.parent.mkdir(parents=True,exist_ok=True)
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model.vis.headlight.ambient[:]=[0.6,0.6,0.6]
    model.vis.headlight.diffuse[:]=[0.8,0.8,0.8]
    renderer=mujoco.Renderer(model,height=480,width=480);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.lookat[:]=[0,0,-.08];camera.distance=1.35;camera.azimuth=25;camera.elevation=-12
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
    names=ref['human_joint_names'].tolist();human=ref['human_upper_positions'];parents=ref['human_parents']
    selected=[names.index(n) for n in ['Spine1','Neck','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand']]
    edges=[('Spine1','Neck'),('Neck','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),('Neck','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand')]
    fig=plt.figure(figsize=(4.8,4.8),dpi=100);ax=fig.add_subplot(111,projection='3d');fig.subplots_adjust(0,0,1,.90)
    v=human[:,selected];lo=v.min(axis=(0,1));hi=v.max(axis=(0,1));mid=(lo+hi)/2;rad=(hi-lo).max()/2*1.15
    ax.set_xlim(mid[0]-rad,mid[0]+rad);ax.set_ylim(mid[1]-rad,mid[1]+rad);ax.set_zlim(mid[2]-rad,mid[2]+rad);ax.set_box_aspect((1,1,1));ax.view_init(elev=12,azim=25);ax.set_axis_off()
    lines=[ax.plot([],[],[],lw=3,color='#da4646' if 'Right' in b else '#287cbc')[0] for a,b in edges]
    frames=[];start=int(ref['motion_start_frame']);stride=2;fps=float(ref['fps'])/stride
    writer=imageio.get_writer(prefix.with_suffix('.mp4'),fps=fps,codec='libx264',quality=7)
    try:
        for i in range(0,len(qactual),stride):
            j=int(np.rint(ref['human_source_frame'][i]))
            for line,(a,b) in zip(lines,edges):
                v=human[j,[names.index(a),names.index(b)]];line.set_data(v[:,0],v[:,1]);line.set_3d_properties(v[:,2])
            fig.canvas.draw();hp=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
            panels=[hp]
            for q in [ref['qpos'][i],qactual[i]]:
                data.qpos[:]=q;mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option);panels.append(renderer.render().copy())
            canvas=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(canvas)
            texts=['Human upper body (source; held during transitions)','OpenArm reference (kinematic)',controller_label]
            for k,title in enumerate(texts):
                draw.rectangle((k*480,0,(k+1)*480,46),fill='white');draw.text((k*480+8,8),title,fill='black');draw.text((k*480+8,26),f't={ref["time"][i]:.2f}s',fill='black')
            writer.append_data(np.asarray(canvas));frames.append(canvas.resize((960,320)))
        frames[0].save(prefix.with_suffix('.gif'),save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0)
        frames[len(frames)//2].save(prefix.with_suffix('.png'))
    finally:
        writer.close();renderer.close();plt.close(fig)
if __name__=='__main__':main()
