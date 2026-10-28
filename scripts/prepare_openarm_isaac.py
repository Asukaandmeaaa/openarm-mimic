"""Prepare a derived MJCF with one fixed articulation root (original stays intact)."""
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import shutil
import numpy as np
import trimesh
from openarm_mimic.robot import ROOT

def prepare():
    source=ROOT/'assets/openarm_mujoco/v2/openarm_bimanual.xml'
    tree=ET.parse(source); root=tree.getroot()
    compiler=root.find('compiler')
    if compiler is None: compiler=ET.SubElement(root,'compiler')
    meshdir=compiler.get('meshdir','')
    meshroot=(source.parent/meshdir).resolve()
    target=ROOT/'assets/generated/meshes';target.mkdir(parents=True,exist_ok=True)
    for mesh in root.findall('asset/mesh'):
        src=meshroot/mesh.attrib['file']
        dest=target/(mesh.attrib['name']+src.suffix)
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        if np.any(scale<0):
            # PhysX convex mesh instancing with negative scale produces spurious
            # contacts. Bake the reflection and face winding into derived geometry.
            geometry=trimesh.load(src,force='mesh',process=False)
            transform=np.eye(4);transform[:3,:3]=np.diag(scale)
            geometry.apply_transform(transform);geometry.export(dest)
            mesh.set('scale','1 1 1')
        else:shutil.copy2(src,dest)
        mesh.set('file',dest.name)
    compiler.set('meshdir','meshes')
    wb=root.find('worldbody')
    bodies=list(wb.findall('body'))
    base=ET.SubElement(wb,'body',name='openarm_root')
    ET.SubElement(base,'inertial',pos='0 0 0',mass='1',diaginertia='0.01 0.01 0.01')
    for b in bodies: wb.remove(b);base.append(b)
    dest=ROOT/'assets/generated/openarm_isaac.xml'
    ET.indent(tree);tree.write(dest,encoding='utf-8',xml_declaration=True)
    return dest
if __name__=='__main__': print(prepare())
