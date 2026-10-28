"""Bind an experiment to exact robot, reference and adaptation sources."""
import hashlib
from pathlib import Path
from openarm_mimic.robot import ROOT
COMMIT='606aae160b88635c97eefb05a3e4e4591afd7632'
ADAPTER=Path(__file__).resolve().parents[2]
def fingerprint(reference=None):
    from .robot import REFERENCE
    reference=Path(reference) if reference is not None else REFERENCE
    if reference.is_dir():reference=reference/'wave_right_v1.npz'
    groups={
        'reference':[reference.resolve()],
        'robot':list((ROOT/'assets/openarm_mujoco/v2').rglob('*.xml'))+list((ROOT/'assets/openarm_mujoco/v2').rglob('*.stl'))+list((ROOT/'assets/generated/usd').rglob('*.usd')),
        'code':list((ADAPTER/'src').rglob('*.py'))+[ADAPTER/'scripts/train.py',ROOT/'src/openarm_mimic/supermimic/env.py',ROOT/'src/openarm_mimic/kinematics.py',ROOT/'src/openarm_mimic/robot.py',ROOT/'src/openarm_mimic/reference.py'],
    }
    result={'upstream_commit':COMMIT,'contract':'OpenArm18state14action-fixedbase-v1'}
    for name,files in groups.items():
        h=hashlib.sha256()
        for file in sorted(files):h.update(str(file.relative_to(ROOT)).encode());h.update(file.read_bytes())
        result[name]=h.hexdigest()
    return result
