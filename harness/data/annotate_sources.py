"""franka-sonic copy of `mimic/scripts/annotate_sources.py` (franka-bimanual-isaac-sim @14f0d8a).

Runs Isaac Lab's `annotate_demos.py` with OUR task registrations injected. Two changes to the
upstream wrapper, both pod-specific and both deliberate:

1. The Isaac Lab checkout comes from `$ISAACLAB_ROOT` (default `/workspace/isaaclab`, the 2.3.2
   tree the sim user-site was built from). Upstream expects a `.isaaclab/` clone inside the franka
   repo, which this pod does not have — and that repo is upstream: never edited, never committed.
2. `$MIMIC_RATE_HZ` (rev 3c: 50) overrides the Mimic env cfg's hard-coded 30 Hz stepping
   (`DualFrankaHandoverMimicEnvCfg.__post_init__` pins decimation 3 / dt 1/90), so annotate,
   generate and export all step the SAME 50 Hz / 100 Hz-physics loop the sources were recorded at.

    MIMIC_RATE_HZ=50 PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh harness/data/annotate_sources.py \
        --task Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0 --input_file <run>/out/sources.hdf5 \
        --output_file <run>/out/sources_annotated.hdf5 --auto --headless
"""

import os
import sys

from frankas_assets.rig import check_rig, log_rig

# Fail loudly if the source dataset was recorded on a different rig than FR3_RIG selects.
log_rig("annotate")
if "--input_file" in sys.argv:
    check_rig(sys.argv[sys.argv.index("--input_file") + 1])

_root = os.environ.get("ISAACLAB_ROOT", "/workspace/isaaclab")
_target = os.path.join(_root, "scripts", "imitation_learning", "isaaclab_mimic", "annotate_demos.py")
if not os.path.exists(_target):
    raise SystemExit(f"[annotate] Isaac Lab script not found: {_target} (set ISAACLAB_ROOT)")

RATE_INJECT = '''
# --- franka-sonic: step the Mimic env at MIMIC_RATE_HZ instead of the cfg's hard-coded 30 Hz ---
import os as _os

_rate = float(_os.environ.get("MIMIC_RATE_HZ", "0") or 0)
if _rate > 0:
    from mimic.env_cfg import DualFrankaHandoverMimicEnvCfg as _MC

    _orig_pi_rate = _MC.__post_init__

    def _rate_pi(self):
        _orig_pi_rate(self)
        self.decimation = max(1, round(100.0 / _rate))
        self.sim.dt = 1.0 / (_rate * self.decimation)
        self.sim.render_interval = self.decimation
        print(f"[mimic-rate] env stepped at {_rate:g} Hz (decimation {self.decimation}, "
              f"physics dt {self.sim.dt:.4f})", flush=True)

    _MC.__post_init__ = _rate_pi
'''

src = open(_target).read()
# Register our envs right after Isaac's own (post-app-launch import section).
anchor = "import isaaclab_mimic.envs  # noqa: F401"
assert anchor in src, f"anchor not found in {_target}"
src = src.replace(
    anchor, anchor + "\n\nimport mimic  # noqa: F401  (registers the Mimic env + core)\n" + RATE_INJECT, 1
)
sys.argv[0] = _target
exec(compile(src, _target, "exec"), {"__name__": "__main__", "__file__": _target})
