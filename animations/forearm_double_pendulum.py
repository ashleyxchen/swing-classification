import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ------------- Config -------------

WRIST_CSV = "../raw_sessions/wrist_angle_test/angle_test_wrist.csv"
RACKET_CSV = "../raw_sessions/wrist_angle_test/angle_test_racket.csv"
MAX_FRAMES = 700
OUT_GIF = "forearm_racket_double_pendulum.gif"

# Forearm: elbow (0) -> wrist (L_FOREARM)
L_FOREARM = 0.27          # m

# Racket link: wrist (0) -> along racket IMU +X (L_RACKET)
L_RACKET = 0.26           # m distance from wrist watch to racket watch

# ------------- Math utils -------------

def normalize_rows(M):
    M = np.asarray(M, float)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n

def quat_mul(q, r):
    """
    Hamilton product of quaternions q ⊗ r
    q, r: (..., 4) with [w, x, y, z]
    """
    q = np.asarray(q, float)
    r = np.asarray(r, float)
    w1, x1, y1, z1 = q.T
    w2, x2, y2, z2 = r.T
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    q = np.asarray(q, float)
    w, x, y, z = q.T
    return np.column_stack([w, -x, -y, -z])

def quat_to_R(q):
    """
    Quaternion [w,x,y,z] -> 3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

def slerp(q0, q1, u):
    """
    Spherical linear interpolation between unit quaternions q0, q1.
    u in [0,1]
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(min(dot, 1.0), -1.0)

    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * u
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0*q0 + s1*q1
    return q / np.linalg.norm(q)

def interp_quats(times_src, quats_src, times_target):
    """
    Time-warp quaternions onto a new time base using SLERP.
    times_* in seconds, sorted.
    """
    times_src = np.asarray(times_src, float)
    quats_src = np.asarray(quats_src, float)
    out = np.empty((len(times_target), 4), float)

    idxs = np.searchsorted(times_src, times_target)
    n = len(times_src)

    for k, (t, idx) in enumerate(zip(times_target, idxs)):
        if idx <= 0:
            out[k] = quats_src[0]
        elif idx >= n:
            out[k] = quats_src[-1]
        else:
            t0 = times_src[idx - 1]
            t1 = times_src[idx]
            if t1 <= t0:
                out[k] = quats_src[idx]
            else:
                u = float((t - t0) / (t1 - t0))
                out[k] = slerp(quats_src[idx - 1], quats_src[idx], u)

    return normalize_rows(out)

def integrate_gyro(t, omega):
    """
    Fallback: integrate angular velocity to orientation.
    t: (N,) seconds (monotonic)
    omega: (N,3) rad/s
    """
    qs = [np.array([1.0, 0.0, 0.0, 0.0])]
    for k in range(len(t) - 1):
        dt = float(t[k+1] - t[k])
        if dt <= 0:
            qs.append(qs[-1])
            continue
        w = omega[k]
        w_norm = float(np.linalg.norm(w))
        if w_norm < 1e-12:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            theta = w_norm * dt
            axis = w / w_norm
            half = 0.5 * theta
            dq = np.concatenate(([math.cos(half)], axis * math.sin(half)))
        qs.append(normalize_rows(quat_mul(qs[-1][None, :], dq[None, :]))[0])
    return np.vstack(qs)

def rotate_points(R, P):
    """
    R: (3,3), P: (N,3)
    """
    return (R @ P.T).T

# ------------- Load & align SensorLogger data -------------

def load_orientation_from_csv(path):
    """
    Load orientation from SensorLogger CSV.
    - Uses quaternionW/X/Y/Z if present.
    - Falls back to integrating rotationRateX/Y/Z.
    - Returns (t_ns, quats) with t_ns as int64 nanoseconds.
    """
    df = pd.read_csv(path)

    quat_cols = ["quaternionW", "quaternionX", "quaternionY", "quaternionZ"]
    has_quat = all(c in df.columns for c in quat_cols)

    # Time base: prefer 'time' from SensorLogger
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
        t_ns = df["time"].astype("int64").to_numpy()
    elif "seconds_elapsed" in df.columns:
        df = df.sort_values("seconds_elapsed").reset_index(drop=True)
        t = df["seconds_elapsed"].to_numpy(float)
        t_ns = (t * 1e9).astype("int64")
    else:
        # Fallback: synthetic 100 Hz
        dt = 0.01
        t = np.arange(len(df)) * dt
        t_ns = (t * 1e9).astype("int64")

    if has_quat:
        quats = df[quat_cols].to_numpy(float)
        quats = normalize_rows(quats)
    else:
        gyro_cols = ["rotationRateX", "rotationRateY", "rotationRateZ"]
        if not all(c in df.columns for c in gyro_cols):
            raise ValueError(f"{path}: no quaternion or gyro columns.")
        t_sec = (t_ns - t_ns[0]) * 1e-9
        omega = df[gyro_cols].to_numpy(float)
        quats = integrate_gyro(t_sec, omega)

    return t_ns, quats

# Wrist (forearm) IMU
t_ns_wrist, quats_wrist_raw = load_orientation_from_csv(WRIST_CSV)
# Racket IMU
t_ns_racket, quats_racket_raw = load_orientation_from_csv(RACKET_CSV)

# Relative times in seconds from common zero (to help interpolation)
t0 = min(t_ns_wrist[0], t_ns_racket[0])
t_wrist = (t_ns_wrist - t0) * 1e-9
t_racket = (t_ns_racket - t0) * 1e-9

# Overlapping time window
t_start = max(t_wrist[0], t_racket[0])
t_end = min(t_wrist[-1], t_racket[-1])
if t_end <= t_start:
    raise RuntimeError("No overlapping time between wrist and racket recordings.")

# Shared time base (this is what the animation runs over)
num_frames = min(MAX_FRAMES, len(quats_wrist_raw), len(quats_racket_raw))
t_common = np.linspace(t_start, t_end, num_frames)

# Interpolate both orientation streams onto the common timeline
quats_wrist_common = interp_quats(t_wrist, quats_wrist_raw, t_common)
quats_racket_common = interp_quats(t_racket, quats_racket_raw, t_common)

# Rebase so first wrist orientation = identity, and apply same rebase to racket.
# This preserves the *relative* orientation between wrist and racket.
q0 = quats_wrist_common[0:1, :]
q0_inv = quat_conj(q0)

quats_wrist = normalize_rows(quat_mul(q0_inv.repeat(num_frames, axis=0),
                                      quats_wrist_common))
quats_racket = normalize_rows(quat_mul(q0_inv.repeat(num_frames, axis=0),
                                       quats_racket_common))

# ------------- Geometry: Forearm -------------

R_ELBOW = 0.045
R_WRIST = 0.03
N_THETA = 24
N_LEN = 2

theta = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
x = np.linspace(0, L_FOREARM, N_LEN)

def radius_at(xval):
    return R_ELBOW + (R_WRIST - R_ELBOW) * (xval / L_FOREARM)

# forearm cylinder vertices in local coords (elbow at x=0)
verts_forearm = []
for xi in x:
    r = radius_at(xi)
    ring = np.column_stack([
        np.full_like(theta, xi),
        r * np.cos(theta),
        r * np.sin(theta)
    ])
    verts_forearm.append(ring)
verts_forearm = np.stack(verts_forearm, axis=0)  # (N_LEN, N_THETA, 3)

# quad faces on cylinder
faces_forearm = []
for i in range(N_LEN - 1):
    for j in range(N_THETA):
        jn = (j + 1) % N_THETA
        faces_forearm.append((i, j, i, jn, i+1, jn, i+1, j))

# watch plate near the wrist
PLATE_OFFSET = 0.01
PLATE_SIZE = 0.04
plate_local = np.array([
    [L_FOREARM - PLATE_OFFSET, -PLATE_SIZE/2, -PLATE_SIZE/2],
    [L_FOREARM - PLATE_OFFSET,  PLATE_SIZE/2, -PLATE_SIZE/2],
    [L_FOREARM - PLATE_OFFSET,  PLATE_SIZE/2,  PLATE_SIZE/2],
    [L_FOREARM - PLATE_OFFSET, -PLATE_SIZE/2,  PLATE_SIZE/2],
])

# ------------- Geometry: Racket (second rigid body) -------------

# Shaft: cylinder from wrist (0) to L_RACKET along racket +X
N_THETA_R = 16
N_LEN_R = 2
theta_r = np.linspace(0, 2*np.pi, N_THETA_R, endpoint=False)
x_r = np.linspace(0, L_RACKET, N_LEN_R)
R_SHAFT = 0.015

verts_racket = []
for xi in x_r:
    ring = np.column_stack([
        np.full_like(theta_r, xi),
        R_SHAFT * np.cos(theta_r),
        R_SHAFT * np.sin(theta_r)
    ])
    verts_racket.append(ring)
verts_racket = np.stack(verts_racket, axis=0)

faces_racket = []
for i in range(N_LEN_R - 1):
    for j in range(N_THETA_R):
        jn = (j + 1) % N_THETA_R
        faces_racket.append((i, j, i, jn, i+1, jn, i+1, j))

# Simple "racket head" plate at distal end of shaft
HEAD_LEN = 0.18
HEAD_W = 0.12
HEAD_OFFSET = L_RACKET
racket_head_local = np.array([
    [HEAD_OFFSET,        -HEAD_W/2, 0.0],
    [HEAD_OFFSET,         HEAD_W/2, 0.0],
    [HEAD_OFFSET+HEAD_LEN, HEAD_W/2, 0.0],
    [HEAD_OFFSET+HEAD_LEN,-HEAD_W/2, 0.0],
])

# ------------- Elbow sphere -------------

def make_sphere(center, radius, n_u=16, n_v=10):
    u = np.linspace(0, 2*np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v)
    pts = []
    for vv in v:
        for uu in u:
            x = center[0] + radius * np.sin(vv) * np.cos(uu)
            y = center[1] + radius * np.sin(vv) * np.sin(uu)
            z = center[2] + radius * np.cos(vv)
            pts.append([x, y, z])
    return np.array(pts)

ELBOW_WORLD = np.array([0.0, 0.0, 0.0])

# ------------- Matplotlib setup -------------

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

cyl_poly = Poly3DCollection([])
ax.add_collection3d(cyl_poly)

plate_poly = Poly3DCollection([])
ax.add_collection3d(plate_poly)

racket_poly = Poly3DCollection([])
ax.add_collection3d(racket_poly)

racket_head_poly = Poly3DCollection([])
ax.add_collection3d(racket_head_poly)

elbow_pts = make_sphere(ELBOW_WORLD, 0.02)
elbow_scatter = ax.scatter(elbow_pts[:, 0], elbow_pts[:, 1], elbow_pts[:, 2], s=1)

# local axes for visualization
axis_lines_forearm = [ax.plot([], [], [], linewidth=2)[0] for _ in range(3)]
axis_lines_racket = [ax.plot([], [], [], linewidth=1)[0] for _ in range(3)]

lim = 0.7
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Forearm + Racket (Double Rigid Body from Apple Watch IMUs)")

# equal aspect if available
try:
    ax.set_box_aspect([1, 1, 1])
except Exception:
    pass

# ------------- Animation update -------------

def update(frame_idx):
    q_w = quats_wrist[frame_idx]
    q_r = quats_racket[frame_idx]

    R_w = quat_to_R(q_w)  # forearm orientation
    R_r = quat_to_R(q_r)  # racket orientation (already in same rebased world)

    # ---- Forearm ----
    forearm_polys = []
    for (i0, j0, i1, j1, i2, j2, i3, j3) in faces_forearm:
        pts_local = np.vstack([
            verts_forearm[i0, j0],
            verts_forearm[i1, j1],
            verts_forearm[i2, j2],
            verts_forearm[i3, j3],
        ])
        pts_world = rotate_points(R_w, pts_local) + ELBOW_WORLD
        forearm_polys.append(pts_world)
    cyl_poly.set_verts(forearm_polys)

    # watch plate on wrist
    plate_world = rotate_points(R_w, plate_local) + ELBOW_WORLD
    plate_poly.set_verts([plate_world])

    # wrist joint position in world
    wrist_world = rotate_points(R_w, np.array([[L_FOREARM, 0.0, 0.0]]))[0] + ELBOW_WORLD

    # ---- Racket shaft (second rigid body) ----
    racket_polys = []
    for (i0, j0, i1, j1, i2, j2, i3, j3) in faces_racket:
        pts_local = np.vstack([
            verts_racket[i0, j0],
            verts_racket[i1, j1],
            verts_racket[i2, j2],
            verts_racket[i3, j3],
        ])
        # anchored at wrist_world, rotated by racket IMU orientation
        pts_world = rotate_points(R_r, pts_local) + wrist_world
        racket_polys.append(pts_world)
    racket_poly.set_verts(racket_polys)

    # racket head plate
    head_world = rotate_points(R_r, racket_head_local) + wrist_world
    racket_head_poly.set_verts([head_world])

    # ---- Axes: forearm at elbow ----
    for line, e in zip(axis_lines_forearm, np.eye(3)):
        end = R_w @ (0.15 * e)
        line.set_data(
            [ELBOW_WORLD[0], ELBOW_WORLD[0] + end[0]],
            [ELBOW_WORLD[1], ELBOW_WORLD[1] + end[1]],
        )
        line.set_3d_properties(
            [ELBOW_WORLD[2], ELBOW_WORLD[2] + end[2]]
        )

    # ---- Axes: racket at wrist (shows second link frame) ----
    for line, e in zip(axis_lines_racket, np.eye(3)):
        end = R_r @ (0.12 * e)
        line.set_data(
            [wrist_world[0], wrist_world[0] + end[0]],
            [wrist_world[1], wrist_world[1] + end[1]],
        )
        line.set_3d_properties(
            [wrist_world[2], wrist_world[2] + end[2]]
        )

    return [
        cyl_poly,
        plate_poly,
        racket_poly,
        racket_head_poly,
        elbow_scatter,
        *axis_lines_forearm,
        *axis_lines_racket,
    ]

ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=10,
    blit=False
)

ani.save(OUT_GIF, writer=PillowWriter(fps=24))
plt.close(fig)

print("Saved:", OUT_GIF)
