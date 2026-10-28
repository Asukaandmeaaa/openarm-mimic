"""Fetch a hash-pinned CMU wave BVH and remove its leading reference pose."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

from openarm_mimic.robot import ROOT

URL = ('https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/'
       '09a07f54f3bbb58797325f009282d0b2048a2871/data/141/141_16.bvh')
SHA256 = 'd428d2c4fa8873d077537567ad32c95b9687a479a10522d314b203fddea37daf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Use a previously downloaded original 141_16.bvh')
    args = parser.parse_args()
    source = args.source or ROOT / 'data/raw/cmu/141_16.bvh'
    if source.exists():
        content = source.read_bytes()
    elif args.source:
        raise FileNotFoundError(source)
    else:
        with urlopen(Request(URL, headers={'User-Agent': 'OpenArm-Mimic'}), timeout=45) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError('Source SHA-256 differs from the pinned CMU 141_16 BVH')
    lines = content.decode('utf-8').splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('Frame Time:'))
    frames = int(lines[header - 1].split(':', 1)[1])
    rows = [line for line in lines[header + 1:] if line.strip()]
    if frames != 300 or len(rows) != frames:
        raise ValueError('Expected 300 BVH frames including the leading reference pose')
    prefix = lines[:header + 1]
    prefix[header - 1] = f'Frames: {frames - 1}'
    output = ROOT / 'data/processed/141_16_no_reference_frame.bvh'
    cleaned = ('\n'.join(prefix + rows[1:]) + '\n').encode('utf-8')
    if output.exists() and output.read_bytes() != cleaned:
        raise FileExistsError(f'Refusing to replace different processed source: {output}')
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cleaned)
    print(f'Verified source SHA-256: {SHA256}')
    print(f'Prepared {frames - 1} frames: {output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
