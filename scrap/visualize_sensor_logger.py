import argparse
import json
import math
import re
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import json_normalize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union

DEFAULT_INPUT = str(Path(__file__).resolve().parent.parent / "backhand_x_3-2025-10-21_18-49-27.json")

def read_json_to_df(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    records = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            records = obj["data"]
        elif isinstance(obj, list):
            records = obj
        else:
            records = [obj]
    except json.JSONDecodeError:
        # NDJSON fallback
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    df = json_normalize(records, sep=".")
    return df

def detect_time_index(df: pd.DataFrame) -> Union[pd.DatetimeIndex, pd.Index]:
    cols = list(df.columns)
    # Prioritized time-ish columns
    priority = [
        "timestamp", "time", "startdate", "start_time", "start",
        "systemtime", "epoch", "date", "createdat"
    ]
    cand = None
    for p in priority:
        for c in cols:
            if p in c.lower():
                cand = c
                break
        if cand:
            break
    if cand is None:
        # Try nested deviceMotion.timestamp style
        cand_candidates = [c for c in cols if re.search(r"(?i)\b(time|timestamp|date)\b", c)]
        cand = cand_candidates[0] if cand_candidates else None

    if cand is None:
        # No time column; use sample index
        return pd.RangeIndex(start=0, stop=len(df), step=1, name="sample")

    s = df[cand]
    # Try numeric epoch
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() > 0:
        med = float(s_num.dropna().iloc[len(s_num.dropna()) // 2])
        # Infer unit
        if med > 1e17:
            unit = "ns"
        elif med > 1e11:
            unit = "ms"
        elif med > 1e9:
            # could be ms too, but if >1e9 and <1e11 assume seconds (2025 ~= 1.7e9)
            unit = "s"
        else:
            unit = "s"
        try:
            dt = pd.to_datetime(s_num, unit=unit, utc=True)
            return pd.DatetimeIndex(dt, name=cand)
        except Exception:
            pass
    # Try string datetime
    try:
        dt = pd.to_datetime(s, utc=True, errors="coerce", infer_datetime_format=True)
        if dt.notna().any():
            return pd.DatetimeIndex(dt, name=cand)
    except Exception:
        pass
    # Fallback to sample index
    return pd.RangeIndex(start=0, stop=len(df), step=1, name="sample")

def find_axes(df: pd.DataFrame, base: str):
    cols = list(df.columns)
    # Common shapes: base.x / baseX
    patterns = [
        (f"{base}.x", f"{base}.y", f"{base}.z"),
        (f"{base}X", f"{base}Y", f"{base}Z"),
    ]
    for triplet in patterns:
        if all(c in cols for c in triplet):
            return triplet
    # Regex search
    axis_map = {"x": None, "y": None, "z": None}
    pat = re.compile(rf"(?i)\b{re.escape(base)}(?:[._ ]|)(x|y|z)\b$")
    for c in cols:
        m = pat.search(c)
        if m:
            axis_map[m.group(1).lower()] = c
    if all(axis_map.values()):
        return (axis_map["x"], axis_map["y"], axis_map["z"])
    return None

def first_axes_match(df: pd.DataFrame, bases: list[str]):
    for b in bases:
        axes = find_axes(df, b)
        if axes:
            return axes
    return None

def add_xyz_panel(fig, row, name, df, axes, show_mag=True, axis_labels=("X","Y","Z")):
    colors = {"X":"#1f77b4","Y":"#2ca02c","Z":"#d62728","mag":"#9467bd"}
    for col, lbl in zip(axes, axis_labels):
        if col in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=f"{name} {lbl}", line=dict(color=colors.get(lbl, None))), row=row, col=1)
    if show_mag and all(a in df for a in axes):
        mag = np.sqrt(np.nansum(np.column_stack([df[axes[0]].values, df[axes[1]].values, df[axes[2]].values])**2, axis=1))
        fig.add_trace(go.Scatter(x=df.index, y=mag, name=f"{name} |v|", line=dict(color=colors["mag"], dash="dash")), row=row, col=1)
    fig.update_yaxes(title_text=name, row=row, col=1)

def build_figure(df: pd.DataFrame) -> go.Figure:
    panels = []

    accel_axes = first_axes_match(df, ["userAcceleration", "accelerometerAcceleration", "acceleration", "accel", "acc"])
    if accel_axes: panels.append(("Acceleration (m/s²)", accel_axes, True, ("X","Y","Z")))

    gyro_axes = first_axes_match(df, ["rotationRate", "gyro", "gyroscope"])
    if gyro_axes: panels.append(("Gyroscope (rad/s)", gyro_axes, True, ("X","Y","Z")))

    gravity_axes = first_axes_match(df, ["gravity"])
    if gravity_axes: panels.append(("Gravity (g)", gravity_axes, True, ("X","Y","Z")))

    mag_axes = first_axes_match(df, ["magneticField", "magnetometer", "mag"])
    if mag_axes: panels.append(("Magnetic Field (µT)", mag_axes, True, ("X","Y","Z")))

    attitude_axes = first_axes_match(df, ["attitude"])
    if attitude_axes:
        # Treat as Roll/Pitch/Yaw if we can map suffixes
        axis_labels = []
        for col in attitude_axes:
            if col.lower().endswith(("roll",)): axis_labels.append("Roll")
            elif col.lower().endswith(("pitch",)): axis_labels.append("Pitch")
            elif col.lower().endswith(("yaw",)): axis_labels.append("Yaw")
            else: axis_labels.append(col.split(".")[-1].upper())
        panels.append(("Attitude (rad)", attitude_axes, False, tuple(axis_labels)))

    if not panels:
        # Fallback: plot all numeric columns (up to 8) in separate small panels
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num_cols = num_cols[:8] or df.columns[:4]
        panels = [(c, (c, None, None), False, (c, "", "")) for c in num_cols]

    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, (title, axes, show_mag, labels) in enumerate(panels, start=1):
        if axes[1] is None:
            # single column fallback
            col = axes[0]
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=title), row=i, col=1)
            fig.update_yaxes(title_text=title, row=i, col=1)
        else:
            add_xyz_panel(fig, i, title, df, axes, show_mag=show_mag, axis_labels=labels)

    fig.update_layout(height=max(500, 280 * len(panels)), title="Sensor Logger Visualization", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Time")
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize Sensor Logger JSON (timeseries).")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="Path to Sensor Logger JSON (array or NDJSON).")
    parser.add_argument("-o", "--output", default="sensor_viz.html", help="Output HTML file.")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = read_json_to_df(in_path)
    if df.empty:
        raise SystemExit("No data parsed from JSON.")

    idx = detect_time_index(df)
    df.index = idx
    # Keep only numeric columns for plotting
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[num_cols].copy()
    if df.empty:
        raise SystemExit("No numeric sensor columns found to plot.")

    fig = build_figure(df)
    out = Path(args.output).resolve()
    fig.write_html(out, include_plotlyjs="cdn", auto_open=False)
    print(f"Wrote {out}")
    try:
        webbrowser.open(out.as_uri())
    except Exception:
        pass

if __name__ == "__main__":
    main()