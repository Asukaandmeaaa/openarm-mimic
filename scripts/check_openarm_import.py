"""Build/reuse the derived USD and repair articulation/mimic metadata."""
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--force",action="store_true");args=p.parse_args()
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True)
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.mjcf")
    launcher.app.update()
    from prepare_openarm_isaac import prepare
    import isaaclab.sim as sim
    from isaaclab.assets import Articulation,ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from openarm_mimic.robot import ROOT
    import json
    path=prepare()
    from isaaclab.sim.converters import MjcfConverter,MjcfConverterCfg
    from pxr import Usd,UsdPhysics,PhysxSchema
    converter=MjcfConverter(MjcfConverterCfg(asset_path=str(path),usd_dir=str(ROOT/'assets/generated/usd'),usd_file_name='openarm.usd',fix_base=True,make_instanceable=False,self_collision=True,force_usd_conversion=args.force))
    stage=Usd.Stage.Open(converter.usd_path)
    for prim in stage.Traverse():
        if prim.GetName()=='worldBody':
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):print('ROOT',prim.GetPath(),flush=True)
    joint_prims={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    for side in ('left','right'):
        ref=joint_prims[f'openarm_{side}_finger_joint1']
        follower=joint_prims[f'openarm_{side}_finger_joint2']
        mimic=PhysxSchema.PhysxMimicJointAPI.Apply(follower,'rotX')
        mimic.CreateReferenceJointRel().SetTargets([ref.GetPath()])
        mimic.CreateGearingAttr(-1.0);mimic.CreateOffsetAttr(0.0)
    stage.GetRootLayer().Save()
    ctx=sim.SimulationContext(sim.SimulationCfg(dt=.002,device='cuda:0'))
    cfg=ArticulationCfg(prim_path='/World/Robot',spawn=sim.UsdFileCfg(usd_path=converter.usd_path),actuators={'all':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=30.,damping=2.,effort_limit_sim=40.)})
    robot=Articulation(cfg)
    ctx.reset();robot.update(.002)
    report={'joints':robot.joint_names,'bodies':robot.body_names,'q':robot.data.joint_pos.tolist(),'limits':robot.data.joint_pos_limits.tolist(),'mimic_gearing':-1.0,'mimic_constraint':'q_finger2 - q_finger1 = 0'}
    (ROOT/'reports').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/openarm_isaac_import.json').write_text(json.dumps(report,indent=2))
    print('OPENARM_IMPORT_OK',json.dumps(report),flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    exit_code=1
else:exit_code=0
finally:
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance()
    try:launcher.app.close()
    except SystemExit:pass
raise SystemExit(exit_code)
