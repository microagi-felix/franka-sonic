#!/usr/bin/env bash
# Per-fresh-pod bootstrap for the FR3 handover bake-off on pod `franka-sonic`.
# Idempotent: every step prints [step] then [ok] or [skip]. Safe to re-run.
#
# The container overlay is lost on every fresh pod, so the group membership and
# the apt install recur; everything else lives on Lustre under ~ and persists.
#
# Exit codes: 0 = everything done · 3 = Isaac steps skipped, re-login (or just
# re-run this script) needed · non-zero otherwise = a step failed hard.
set -euo pipefail

KIT_PY=/isaac-sim/kit/python/bin/python3
PYSH=/isaac-sim/python.sh
FR3_REPO="$HOME/code/franka-bimanual-isaac-sim"
FR3_COMMIT=14f0d8a38cb0258f8ad955d2ecde55175ee4fd09
FR3_URL=git@github.com:MicroAGI-Labs/franka-bimanual-isaac-sim.git
FR3_URL_HTTPS=https://github.com/MicroAGI-Labs/franka-bimanual-isaac-sim.git
USERBASE_FR3="$HOME/env/pyuser-fr3"
USERBASE_SONIC="$HOME/env/pyuser-sonic"
GR00T_PY="$HOME/Isaac-GR00T/.venv/bin/python"
WORKLOG="$HOME/agents/2026-09-01_franka-sonic/WORKLOG.md"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { echo; echo "[step] $*"; }
ok()   { echo "[ok] $*"; }
skip() { echo "[skip] $*"; }
warn() { echo "[warn] $*" >&2; }

worklog() {  # append a timestamped line to the campaign WORKLOG (rule h)
  mkdir -p "$(dirname "$WORKLOG")"
  [ -f "$WORKLOG" ] || printf '# WORKLOG — 2026-09-01_franka-sonic\n\nPod-state mutations outside git (apt, usermod, symlinks, installs, downloads).\n\n' > "$WORKLOG"
  printf -- '- %s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$WORKLOG"
}

# ---------------------------------------------------------------------------
# 1. isaac-sim group membership. /isaac-sim is drwxr-x--- isaac-sim:isaac-sim,
#    so without it neither the Kit interpreter nor python.sh is even visible.
# ---------------------------------------------------------------------------
step "1/10 isaac-sim group membership"
if id -nG | grep -qw isaac-sim; then
  skip "already in group isaac-sim"
else
  sudo usermod -aG isaac-sim "$USER"
  worklog "sudo usermod -aG isaac-sim $USER (overlay-only, recurs per fresh pod)"
  ok "added $USER to isaac-sim — re-login required for the group to take effect"
  # A new login is the clean way, but `sg` reads /etc/group directly, so the
  # rest of this run can proceed immediately by re-execing under the group.
  if [ "${FR3_SG_REEXEC:-0}" != "1" ] && command -v sg >/dev/null 2>&1 \
     && getent group isaac-sim | grep -qw "$USER"; then
    echo "[step] re-exec under \`sg isaac-sim\` so this run can finish without a re-login"
    export FR3_SG_REEXEC=1
    exec sg isaac-sim -c "bash '${BASH_SOURCE[0]}'"
  fi
fi

ISAAC_OK=1
if [ ! -x "$KIT_PY" ] || [ ! -x "$PYSH" ]; then
  ISAAC_OK=0
  warn "/isaac-sim is not readable in this session yet — Isaac steps (5, 6, 9) will be skipped"
fi

# ---------------------------------------------------------------------------
# 2. tmux (not in the image; overlay-only, so it recurs per fresh pod)
# ---------------------------------------------------------------------------
step "2/10 tmux"
if command -v tmux >/dev/null 2>&1; then
  skip "tmux present at $(command -v tmux)"
else
  sudo apt-get update -qq && sudo apt-get install -y -qq tmux
  worklog "sudo apt-get install -y tmux (overlay-only, recurs per fresh pod)"
  ok "tmux installed at $(command -v tmux)"
fi

# ---------------------------------------------------------------------------
# 3. directory layout + the two gitignored symlinks into it
# ---------------------------------------------------------------------------
step "3/10 directory layout and symlinks"
mkdir -p "$HOME/code" "$HOME/code/upstream" "$HOME/env" \
         "$HOME/runs/franka-sonic" "$HOME/data/franka-sonic" \
         "$HOME/agents/2026-09-01_franka-sonic"
link() {  # link <target> <linkname>
  if [ -L "$2" ]; then
    [ "$(readlink "$2")" = "$1" ] || { rm -f "$2"; ln -s "$1" "$2"; worklog "re-pointed symlink $2 -> $1"; }
  elif [ -e "$2" ]; then
    warn "$2 exists and is not a symlink — leaving it alone"
  else
    ln -s "$1" "$2"; worklog "symlink $2 -> $1"
  fi
}
link "$HOME/runs/franka-sonic" "$REPO_ROOT/runs"
link "$HOME/data/franka-sonic" "$REPO_ROOT/data"
ok "layout ready (runs -> ~/runs/franka-sonic, data -> ~/data/franka-sonic)"

# ---------------------------------------------------------------------------
# 4. the franka sim repo, pinned
# ---------------------------------------------------------------------------
step "4/10 franka-bimanual-isaac-sim @ ${FR3_COMMIT:0:7}"
if [ -d "$FR3_REPO/.git" ]; then
  git -C "$FR3_REPO" fetch --quiet origin main || warn "fetch failed — using what is on disk"
else
  git clone --quiet "$FR3_URL" "$FR3_REPO" \
    || git clone --quiet "$FR3_URL_HTTPS" "$FR3_REPO"
  worklog "cloned franka-bimanual-isaac-sim -> $FR3_REPO"
fi
FR3_CHANGED=0  # step 5 re-installs the editable package only when this flips
if [ "$(git -C "$FR3_REPO" rev-parse HEAD)" = "$FR3_COMMIT" ]; then
  skip "already at $FR3_COMMIT"
else
  git -C "$FR3_REPO" checkout --quiet "$FR3_COMMIT"
  worklog "checked out franka-bimanual-isaac-sim @ $FR3_COMMIT (detached)"
  ok "checked out $FR3_COMMIT"
  FR3_CHANGED=1
fi
git -C "$FR3_REPO" --no-pager log -1 --format='      HEAD %h %s'

# ---------------------------------------------------------------------------
# 5. the sim stack, installed "the cluster.Dockerfile way" but into ~ .
#
#    Every install is --no-deps: pip cannot see Kit's prebundled torch/numpy
#    and would shadow them (numba 0.59 hard-requires numpy<1.27, so a numpy 2
#    pulled in as a dependency breaks `import isaacsim` outright).
#
#    Group 1 = the eval-path leaves from cluster.Dockerfile, at the exact
#    versions requirements.lock froze. pydantic has to list its own four
#    leaves because --no-deps would otherwise install a package that raises on
#    first import.
#    Group 2 = what mimic/ and teleop/ need; the cluster image omits both
#    packages, so this list has no upstream to mirror — it is the import set
#    (h5py: teleop/recorder + generate_parallel; pyzmq: teleop/ipc +
#    rollout_client; pynput: rollout_client; scipy: retarget/geometry paths).
#    avp-stream is the Vision Pro teleop device and is best-effort only.
# ---------------------------------------------------------------------------
step "5/10 sim stack into PYTHONUSERBASE=$USERBASE_FR3"
if [ "$ISAAC_OK" = "0" ]; then
  skip "/isaac-sim not accessible yet"
else
  mkdir -p "$USERBASE_FR3"
  export PYTHONUSERBASE="$USERBASE_FR3"
  "$KIT_PY" -m pip install --user --no-deps --disable-pip-version-check -q \
      plyfile \
      ninja \
      "wheel==0.47.0" \
      "openpi-client==0.1.2" \
      "pydantic==2.11.10" \
      "pydantic-core==2.33.2" \
      "annotated-types==0.7.0" \
      "typing-inspection==0.4.2"
  ok "eval-path leaves installed"

  # "Already there" is a metadata check for the exact pinned version, not an
  # import probe: the bare Kit interpreter has no numpy on its path (so
  # `import h5py` fails there even when h5py is installed), pynput refuses to
  # import without an X display, and avp-stream is --no-deps by design — an
  # import test re-installed all of them on every re-run.
  installed() {  # installed <dist> <version>
    "$KIT_PY" -c 'import sys; from importlib.metadata import version; sys.exit(version(sys.argv[1]) != sys.argv[2])' \
      "$1" "$2" >/dev/null 2>&1
  }
  importable() { ( cd "$HOME" && "$PYSH" -c "import $1" >/dev/null 2>&1 ); }
  for spec in "h5py==3.16.0" "scipy==1.15.3" "pyzmq==27.1.0" "pynput==1.8.2"; do
    if installed "${spec%%==*}" "${spec##*==}"; then
      skip "$spec already installed"
    else
      "$KIT_PY" -m pip install --user --no-deps --disable-pip-version-check -q "$spec"
      worklog "pip --user install $spec into $USERBASE_FR3"
      ok "$spec"
    fi
  done
  # Hardware teleop only (AVP); never on the eval path. Do not fail the run.
  if installed avp-stream 2.51; then
    skip "avp-stream==2.51 already installed"
  else
    { "$KIT_PY" -m pip install --user --no-deps --disable-pip-version-check -q "avp-stream==2.51" \
        && worklog "pip --user install avp-stream==2.51 into $USERBASE_FR3" && ok "avp-stream==2.51"; } \
      || warn "avp-stream not installed (teleop-only, not needed for the bake-off)"
  fi

  # The editable install is re-done only when the pinned checkout moved (step 4)
  # or the package does not import; otherwise a re-run is a no-op here too.
  if [ "$FR3_CHANGED" = "0" ] && importable evaluation && importable tasks; then
    skip "franka-bimanual-isaac-sim already installed editable"
  else
    "$KIT_PY" -m pip install --user --no-deps --no-build-isolation --disable-pip-version-check -q \
        -e "$FR3_REPO"
    worklog "pip --user install of the sim stack into $USERBASE_FR3 (editable $FR3_REPO)"
    ok "franka-bimanual-isaac-sim installed editable"
  fi
  unset PYTHONUSERBASE
fi

# ---------------------------------------------------------------------------
# 6. verify the sim stack through python.sh (the only interpreter that can
#    import isaacsim — setup_python_env.sh is what puts it on PYTHONPATH).
#    Run from $HOME on purpose: from the repo, `import evaluation` would
#    resolve via cwd and hide a broken editable install.
# ---------------------------------------------------------------------------
step "6/10 verify the sim stack"
if [ "$ISAAC_OK" = "0" ]; then
  skip "/isaac-sim not accessible yet"
else
  ( cd "$HOME" && PYTHONUSERBASE="$USERBASE_FR3" "$PYSH" -c \
      "import isaacsim, isaaclab; import evaluation, tasks; print('sim ok')" )
  ok "isaacsim + isaaclab + evaluation + tasks import"
fi

# ---------------------------------------------------------------------------
# 7. GR00T venv (already on the pod: py3.10, torch 2.7.1+cu128)
# ---------------------------------------------------------------------------
step "7/10 GR00T venv"
"$GR00T_PY" -c "import gr00t; print('gr00t', getattr(gr00t, '__version__', 'ok'))"
ok "$GR00T_PY imports gr00t"

# ---------------------------------------------------------------------------
# 8. GR00T N1.7-3B base weights (~6 GB) into the existing HF cache
# ---------------------------------------------------------------------------
step "8/10 nvidia/GR00T-N1.7-3B in the HF cache"
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [ -d "$HF_HUB/models--nvidia--GR00T-N1.7-3B/snapshots" ] \
   && [ -n "$(ls -A "$HF_HUB/models--nvidia--GR00T-N1.7-3B/snapshots" 2>/dev/null)" ]; then
  skip "already cached at $HF_HUB/models--nvidia--GR00T-N1.7-3B"
else
  echo "      downloading ~6 GB into $HF_HUB (Lustre is 99% full — this is the one allowed big fetch)"
  "$HOME/Isaac-GR00T/.venv/bin/huggingface-cli" download nvidia/GR00T-N1.7-3B \
    || "$GR00T_PY" -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.7-3B')"
  worklog "downloaded nvidia/GR00T-N1.7-3B into $HF_HUB (~6 GB)"
  ok "N1.7-3B cached"
fi

# ---------------------------------------------------------------------------
# 9. gear_sonic into its OWN user-site so its pins cannot fight the sim's.
#    Installed WITH deps on purpose (it is a normal python package with a real
#    dependency set); if that resolution fights Kit's prebundle, the failure is
#    reported here and gate P0 records it as a WARN rather than a stop.
#
#    Two pod facts shape the command (measured 2026-09-02):
#    - pip runs through python.sh, NOT the bare Kit interpreter. Only python.sh
#      (via setup_python_env.sh) puts the prebundled torch/scipy/transformers on
#      sys.path, so pip sees them as satisfied and installs ~25 small wheels.
#      Under the bare interpreter the same command downloads a second torch
#      (~5 GB of CUDA wheels) into ~ — rule k, Lustre is 99 % full.
#    - git cannot reach github.com over https from this pod (every repo, public
#      or not, fails with "could not read Username"); ssh works. gear_sonic's
#      pyproject pins smpl_sim as a git+https URL, so the https->ssh rewrite is
#      scoped to this ONE pip call through GIT_CONFIG_COUNT — never a global
#      git config, which would silently change every other clone on the pod.
# ---------------------------------------------------------------------------
step "9/10 gear_sonic into PYTHONUSERBASE=$USERBASE_SONIC"
if [ "$ISAAC_OK" = "0" ]; then
  skip "/isaac-sim not accessible yet"
elif PYTHONUSERBASE="$USERBASE_SONIC" "$PYSH" -c "import gear_sonic" >/dev/null 2>&1; then
  skip "gear_sonic already importable in $USERBASE_SONIC"
else
  mkdir -p "$USERBASE_SONIC"
  if PYTHONUSERBASE="$USERBASE_SONIC" \
     GIT_CONFIG_COUNT=1 \
     GIT_CONFIG_KEY_0="url.git@github.com:.insteadOf" \
     GIT_CONFIG_VALUE_0="https://github.com/" \
     "$PYSH" -m pip install --user --disable-pip-version-check \
        -e "$HOME/GR00T-WholeBodyControl/gear_sonic[training]"; then
    worklog "pip --user install gear_sonic[training] into $USERBASE_SONIC (through python.sh so the prebundled torch is reused; git https->ssh rewrite scoped to this pip call)"
    ok "gear_sonic installed"
  else
    warn "gear_sonic[training] install FAILED (see the pip output above) — continuing; gate P0 reports this as WARN"
  fi
fi

# ---------------------------------------------------------------------------
# 10. brain repo up to date (read-only reference on the pod)
# ---------------------------------------------------------------------------
step "10/10 brain repo"
git -C "$HOME/microagi-felix-brain" pull --ff-only || warn "brain pull skipped/failed (non-fatal)"
ok "brain repo at $(git -C "$HOME/microagi-felix-brain" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo
if [ "$ISAAC_OK" = "0" ]; then
  cat <<'MSG'
================================================================================
INCOMPLETE: the isaac-sim group is not active in this shell yet.
Open a NEW ssh session (or run `newgrp isaac-sim`) and re-run:

    bash env/bootstrap.sh

Steps 5, 6 and 9 were skipped; everything else is done and idempotent.
================================================================================
MSG
  exit 3
fi
echo "[done] bootstrap complete — next: bash harness/gates/p0.sh"
