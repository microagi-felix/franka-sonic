"""Closed-loop error of a B-oracle trace in the handover env vs the demo's recorded joints (FR3 convention).
usage: PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh p5_trace_fk.py <trace.npz> <export dir with demos_shard*.hdf5>"""
import os, sys, numpy as np, h5py, mujoco
XML = os.path.expanduser("~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/dual_fr3.xml")
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_fr3_link7", "right_fr3_link7")]
J6 = 2.5307
def flanges(q_fr3_14):  # [L1..7, R1..7] FR3 convention
    q = q_fr3_14.astype(np.float64).copy(); q[5] -= J6; q[12] -= J6
    d.qpos[:14] = q; mujoco.mj_forward(m, d)
    return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, 0, 0.107]) for b in bodies])
z = np.load(sys.argv[1]); name = str(z["name"]); shard, demo = name.split(":")
with h5py.File(os.path.join(sys.argv[2], shard), "r") as f:
    g = f["data"][demo]
    ql = np.asarray(g["obs/joint_pos_left"]); qr = np.asarray(g["obs/joint_pos_right"])
    gl = np.asarray(g["actions"])[:, 7]; 
q_demo = np.concatenate([ql, qr], 1); T = len(q_demo)
st = z["state16"]; wire = z["wire"]
q_meas = np.concatenate([st[:, 0:7], st[:, 8:15]], 1)[:T]
q_wire = np.concatenate([wire[:, 0:7], wire[:, 8:15]], 1)[:T]
names = [f"L{i}" for i in range(1, 8)] + [f"R{i}" for i in range(1, 8)]
close = int(np.argmax(gl < 0)) if (gl < 0).any() else -1
print(f"{name}: T_demo={T} trace={len(st)} left grip closes at demo frame {close}")
for lab, q in (("measured", q_meas), ("target", q_wire)):
    e = np.abs(q - q_demo)
    print(f"  {lab:8s}-demo |dq| mean all {e.mean():.3f} rad; per joint", {n: round(float(v), 3) for n, v in zip(names, e.mean(0))})
    fe = np.array([np.linalg.norm(flanges(q[i]) - flanges(q_demo[i]), axis=1) for i in range(T)]) * 100
    w = slice(max(0, close - 100), close + 20) if close > 0 else slice(100, 250)
    print(f"  {lab:8s} flange cm: L mean {fe[:, 0].mean():.1f} max {fe[:, 0].max():.1f} | L grasp window {w} mean {fe[w, 0].mean():.1f} max {fe[w, 0].max():.1f} | R mean {fe[:, 1].mean():.1f}")
# lag estimate: best time shift of measured vs demo (left arm) in the first 300 frames
best = min(range(0, 40), key=lambda s: np.abs(q_meas[s:300 + s, :7] - q_demo[:300, :7]).mean())
print(f"  best lag (measured lags demo by) {best} frames = {best / 50:.2f} s; err at that lag {np.abs(q_meas[best:300 + best, :7] - q_demo[:300, :7]).mean():.3f}")
# where is the hand at the demo's close frame, in the demo vs measured
if close > 0:
    fd = flanges(q_demo[close]); fm = flanges(q_meas[close]); fw = flanges(q_wire[close])
    print(f"  at close frame {close}: demo L flange {fd[0].round(3)} measured {fm[0].round(3)} target {fw[0].round(3)}  |meas-demo| {np.linalg.norm(fm[0]-fd[0])*100:.1f} cm |tgt-demo| {np.linalg.norm(fw[0]-fd[0])*100:.1f} cm")
    # 'settling': measured vs demo 25 frames later
    fm2 = flanges(q_meas[min(T-1, close + 25)])
    print(f"  measured 0.5 s after close vs demo at close: {np.linalg.norm(fm2[0]-fd[0])*100:.1f} cm")
