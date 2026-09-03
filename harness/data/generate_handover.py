"""franka-sonic copy of `mimic/scripts/generate_handover.py` (franka-bimanual-isaac-sim @14f0d8a).

Runs Isaac Lab's `generate_dataset.py` (MimicGen) with OUR task registrations injected, the
per-worker `MIMIC_DATAGEN_SEED` override upstream already had, and two pod-specific changes:

1. Isaac Lab checkout from `$ISAACLAB_ROOT` (default `/workspace/isaaclab`) instead of the
   franka repo's missing `.isaaclab/` clone (that repo is upstream: never edited).
2. `$MIMIC_RATE_HZ` (rev 3c: 50) overrides the Mimic env cfg's hard-coded 30 Hz stepping so the
   generated action sequences are 50 Hz sequences, like the sources and the export.

    MIMIC_RATE_HZ=50 MIMIC_DATAGEN_SEED=100 PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh \
        harness/data/generate_handover.py --task Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0 \
        --input_file <annotated> --output_file <out> --generation_num_trials 20 --headless
"""

import os
import sys

from frankas_assets.rig import check_rig, log_rig

# Fail loudly if the annotated source dataset's rig differs from the active FR3_RIG.
log_rig("generate")
if "--input_file" in sys.argv:
    check_rig(sys.argv[sys.argv.index("--input_file") + 1])

_root = os.environ.get("ISAACLAB_ROOT", "/workspace/isaaclab")
_target = os.path.join(_root, "scripts", "imitation_learning", "isaaclab_mimic", "generate_dataset.py")
if not os.path.exists(_target):
    raise SystemExit(f"[generate] Isaac Lab script not found: {_target} (set ISAACLAB_ROOT)")

INJECT = '''
import mimic  # noqa: F401  (registers the Mimic env + core)

# Per-worker datagen seed override (generate_dataset seeds random/np/torch from
# datagen_config.seed — parallel workers MUST differ or they generate duplicates), plus the
# franka-sonic MIMIC_RATE_HZ override (rev 3c: 50 Hz stepping instead of the cfg's 30 Hz).
import os as _os

from mimic.env_cfg import DualFrankaHandoverMimicEnvCfg as _MC

_seed = _os.environ.get("MIMIC_DATAGEN_SEED")
_rate = float(_os.environ.get("MIMIC_RATE_HZ", "0") or 0)
_orig_pi = _MC.__post_init__


def _patched_pi(self):
    _orig_pi(self)
    if _seed is not None:
        self.datagen_config.seed = int(_seed)
    if _rate > 0:
        self.decimation = max(1, round(100.0 / _rate))
        self.sim.dt = 1.0 / (_rate * self.decimation)
        self.sim.render_interval = self.decimation
        print(f"[mimic-rate] env stepped at {_rate:g} Hz (decimation {self.decimation}, "
              f"physics dt {self.sim.dt:.4f}), datagen seed {self.datagen_config.seed}", flush=True)


_MC.__post_init__ = _patched_pi
'''

src = open(_target).read()
anchor = "import isaaclab_mimic.envs  # noqa: F401"
assert anchor in src, f"anchor not found in {_target}"
src = src.replace(anchor, anchor + "\n" + INJECT, 1)
sys.argv[0] = _target
exec(compile(src, _target, "exec"), {"__name__": "__main__", "__file__": _target})
