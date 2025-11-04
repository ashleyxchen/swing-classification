# forearm animation with a fixed elbow at the origin.
# Uses quaternions from the uploaded CSV (falls back to gyro integration if needed).
# The forearm is modeled as a cylinder aligned with its local +X axis, with the elbow at x=0 and the wrist at x=L.
# The watch face plate is drawn near the wrist. The elbow remains fixed in world coordinates.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from caas_jupyter_tools import display_dataframe_to_user

csv_path = "/mnt/data/backhand_raw.csv"
df = pd.read_csv(csv_path)

# columns
TIME_COL = "seconds_elapsed" if "seconds_elapsed" in df.columns else None
quat_cols = ["quaternionW","quaternionX","quaternionY","quaternionZ"]
use_quat = all(c in df.columns for c in quat_cols)

if TIME_COL is not None:
    t = df[TIME_COL].to_numpy(float)
    order = np.argsort(t)
    df = df.iloc[order].reset_index(drop=True)
    t = df[TIME_COL].to_numpy(float)
else:
    t = np.arange(len(df)) / 100.0  # fallback

def normalize_rows(M):
    n = (M**2).sum(axis=1, keepdims=True)**0.5
    n[n==0] = 1.0
    return M / n

def quat_mul(q, r):
    # q and r: (...,4) [w,x,y,z]
    w1,x1,y1,z1 = q.T
    w2,x2,y2,z2 = r.T
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    w,x,y,z = q.T
    return np.column_stack([w,-x,-y,-z])

def quat_to_R(q):
    # q=[w,x,y,z] -> 3x3 rotation matrix
    w,x,y,z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

def integrate_gyro(t, omega):
    qs = [np.array([1.0,0.0,0.0,0.0])]
    for k in range(len(t)-1):
        dt = max(t[k+1]-t[k], 0.0)
        theta = np.linalg.norm(omega[k]) * dt
        if theta < 1e-12:
            dq = np.array([1.0,0.0,0.0,0.0])
        else:
            axis = omega[k] / (theta / dt)
            half = 0.5*theta
            dq = np.array([np.cos(half), *(axis*np.sin(half))])
        qs.append(normalize_rows(quat_mul(qs[-1][None,:], dq[None,:]))[0])
    return np.vstack(qs)

# Orientation sequence
if use_quat:
    quats = df[quat_cols].to_numpy(float)
    quats = normalize_rows(quats)
    source = "quaternion"
else:
    omega = df[["rotationRateX","rotationRateY","rotationRateZ"]].to_numpy(float)
    quats = integrate_gyro(t, omega)
    source = "gyro"

# Rebase to start at identity to "start at an arbitrary position"
q0_inv = quat_conj(quats[0:1, :])
quats = normalize_rows(quat_mul(q0_inv.repeat(len(quats), axis=0), quats))

# Subsample for speed
max_frames = 700
N = len(quats)
if N > max_frames:
    idx = np.linspace(0, N-1, max_frames).astype(int)
    quats_sub = quats[idx]
else:
    quats_sub = quats

# ---------------- Forearm geometry ----------------
# Cylinder along local +X from x=0 (elbow) to x=L (wrist)
L = 0.27  # 27 cm approximate forearm length
R_elbow = 0.045
R_wrist = 0.03
n_theta = 24
n_len = 2

theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
x = np.linspace(0, L, n_len)
# Tapered radius along x
def radius_at(xval):
    return R_elbow + (R_wrist - R_elbow) * (xval / L)

verts = []
faces = []

# Build ring vertices
for xi in x:
    r = radius_at(xi)
    ring = np.column_stack([
        np.full_like(theta, xi),
        r*np.cos(theta),
        r*np.sin(theta)
    ])  # (n_theta, 3) in local forearm coords
    verts.append(ring)

verts = np.stack(verts, axis=0) # (n_len, n_theta, 3)

# Build quad faces between rings
for i in range(n_len-1):
    for j in range(n_theta):
        jn = (j+1) % n_theta
        # each quad as a list of four vertex indices (we will pass actual points later)
        faces.append((i, j, i, jn, i+1, jn, i+1, j))

# Create elbow sphere (small mesh) for visualization
def make_sphere(center, radius, n_u=16, n_v=10):
    u = np.linspace(0, 2*np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v)
    puntos = []
    for vv in v:
        for uu in u:
            x = center[0] + radius*np.sin(vv)*np.cos(uu)
            y = center[1] + radius*np.sin(vv)*np.sin(uu)
            z = center[2] + radius*np.cos(vv)
            puntos.append([x,y,z])
    return np.array(puntos)

elbow_center_world = np.array([0.0, 0.0, 0.0])  # fixed

# watch face plate near the wrist (a thin square in local YZ plane at x=L-plate_offset)
plate_offset = 0.01
plate_size = 0.04
plate_thickness = 0.005
plate_local = np.array([
    [L-plate_offset, -plate_size/2, -plate_size/2],
    [L-plate_offset,  plate_size/2, -plate_size/2],
    [L-plate_offset,  plate_size/2,  plate_size/2],
    [L-plate_offset, -plate_size/2,  plate_size/2]
])

# ---------------- Animation ----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Poly collections for cylinder quads
cyl_poly = Poly3DCollection([])
ax.add_collection3d(cyl_poly)

# Plate
plate_poly = Poly3DCollection([])
ax.add_collection3d(plate_poly)

# Elbow (use a scatter cloud for simplicity)
elbow_pts = make_sphere(elbow_center_world, 0.02)
elbow_scatter = ax.scatter(elbow_pts[:,0], elbow_pts[:,1], elbow_pts[:,2], s=1)

# Axes
axis_lines = [ax.plot([], [], [], linewidth=2)[0] for _ in range(3)]

# World bounds
lim = 0.4
ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Forearm model with fixed elbow")

def rotate_points(R, P):
    return (R @ P.T).T

def update(i):
    q = quats_sub[i]
    R = quat_to_R(q)

    # Rotate cylinder vertices and translate so elbow stays at origin
    polys = []
    for (i0, j0, i1, j1, i2, j2, i3, j3) in faces:
        pts_local = np.vstack([
            verts[i0, j0],
            verts[i1, j1],
            verts[i2, j2],
            verts[i3, j3]
        ])  # (4,3)
        pts_world = rotate_points(R, pts_local) + elbow_center_world
        polys.append(pts_world)

    cyl_poly.set_verts(polys)

    # Rotate the watch plate
    plate_world = rotate_points(R, plate_local) + elbow_center_world
    plate_poly.set_verts([plate_world])

    # Body-fixed axes (forearm local axes)
    for line, e in zip(axis_lines, np.eye(3)):
        end = R @ (0.12*e)
        line.set_data([0, end[0]], [0, end[1]])
        line.set_3d_properties([0, end[2]])

    return [cyl_poly, plate_poly, elbow_scatter, *axis_lines]

ani = FuncAnimation(fig, update, frames=len(quats_sub), interval=10, blit=False)

out_path = "/mnt/data/forearm_backhand.gif"
ani.save(out_path, writer=PillowWriter(fps=24))
plt.close(fig)

print("Saved:", out_path)
