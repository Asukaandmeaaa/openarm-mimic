# OpenArm Mimic

Human-motion retargeting and model-based motion tracking for a fixed-base,
bimanual OpenArm. The training path adapts TrackerLab SuperMimic at commit
`606aae160b88635c97eefb05a3e4e4591afd7632` to 18 physical joints and 14 arm
controls. Training uses IsaacLab/PhysX; reference generation, PD comparison
and independent policy playback use MuJoCo.

This is an unofficial research implementation for no-contact simulation.
Each policy tracks one reference. Hardware deployment, contact tasks and
multi-motion generalization have not been validated.

## Installation

Use Linux, Python 3.11 and a separate Python environment. Training and USD
conversion require Isaac Sim 5.1 / IsaacLab 2.3.2, an NVIDIA GPU and compatible
drivers. The pinned IsaacLab revision is in
`integrations/trackerlab_openarm/resources.lock.json`. Install IsaacLab using
its own installation instructions before installing these Python dependencies:

```bash
git clone https://github.com/Asukaandmeaaa/openarm-mimic.git
cd openarm-mimic
python -m pip install -r integrations/trackerlab_openarm/requirements.txt
```

The wrapper uses the active `python3`. Alternatively set
`MIMIC_CONDA_PREFIX=/absolute/path/to/your/environment` to select its Python.
The wrapper adds the source packages to `PYTHONPATH`; editable installs are
not required. CPU checks and reference generation do not require Isaac Sim.

## Reproduce the wave example

Raw motion, generated references and checkpoints are downloaded or generated
locally. Review the source terms in `NOTICE.txt` and the retained component
licenses before using external resources.

```bash
# Download the pinned CMU 141_16 BVH, verify SHA-256 and remove its first pose.
./scripts/python.sh scripts/prepare_wave_source.py
# Offline alternative: append --source /path/to/original/141_16.bvh

# Build the default 50 Hz reference.
./scripts/python.sh scripts/build_wave_reference.py

# Validate FK against MuJoCo and run the non-learning physical PD baseline.
./scripts/python.sh scripts/check_openarm_kinematics.py
./scripts/python.sh scripts/evaluate_wave_pd.py --output-dir reports/pd_wave

# In an IsaacLab environment, generate the fixed-root USD and validate import.
./scripts/python.sh scripts/check_openarm_import.py

# Check forward/backward execution with all optimizer updates suppressed.
./scripts/python.sh integrations/trackerlab_openarm/scripts/train.py \
  --config configs/trackerlab_openarm_wave.json --validate_only \
  --iterations 1 --num_envs 4 --buffer_steps 64 --warmup_steps 64 \
  --wm_only_iters 0 --wm_batches 1 --wm_batch 8 --pi_batch 8 \
  --eval_every 0 --log_dir reports/preflight

# Train into a new directory; the first 500 iterations fit the world model only.
./scripts/train_trackerlab_openarm.sh runs/wave

# Evaluate a trained checkpoint in PhysX.
./scripts/python.sh integrations/trackerlab_openarm/scripts/evaluate_openarm.py \
  --checkpoint runs/wave/ckpt_002000.pt \
  --motion data/processed/wave_right_v1.npz --output reports/wave_eval

# Run that policy in an independent MuJoCo closed loop.
./scripts/python.sh scripts/play_trackerlab_mujoco.py \
  --checkpoint runs/wave/ckpt_002000.pt \
  --motion data/processed/wave_right_v1.npz --speed 1
```

Reports, videos, generated USD/meshes and trained weights remain local.
Choose new output directories for repeat runs. The model's servo settings and
the retiming review limits are simulation settings, not hardware calibration.

Other wave presets are in `configs/motions/`; pass `--config` to the reference
builder and the matching `--motion_dir` to training. For other gestures,
`prepare_gesture_sources.py` and `build_gesture_reference.py` accept locally
supplied CMU BVH/AMC files. Gesture training presets require the corresponding
reference NPZ; they are not bundled datasets.

## Offline tests

```bash
./scripts/python.sh -m pytest integrations/trackerlab_openarm/tests -q
./scripts/python.sh scripts/check_openarm_kinematics.py
```

## Source layout and licenses

- `src/openarm_mimic/`: robot mapping, reference validation, differentiable FK
  and the PhysX asset backend.
- `integrations/trackerlab_openarm/`: adapted world model, policy, critic,
  replay, MDP, training and full-reference evaluation.
- `assets/openarm_mujoco/`: required OpenArm model, meshes and named resolver.
- `configs/`, `scripts/`: motion preparation, training and playback.
- `vendor/lafan_vendor/`: unchanged BVH/FK utility with its own license.

Component licenses and attribution are retained in `NOTICE.txt`,
`assets/openarm_mujoco/LICENSE`, `src/openarm_mimic/third_party/`,
`integrations/trackerlab_openarm/LICENSE` and `NOTICE.md`, and
`vendor/lafan_vendor/license.txt`. No repository-wide license overrides these
terms. This release contains source and required robot assets; it excludes
experimental records, the workspace documentation folder, datasets and weights.
