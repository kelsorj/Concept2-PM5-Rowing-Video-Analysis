#!/usr/bin/env python3
"""
Create Advanced Synchronized Overlay Video
Combines corrected force data processing with sophisticated animated overlays
"""

import cv2
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
import re
import glob

def load_and_combine_force_data(raw_csv_path):
    """Load raw force data and combine Drive + Dwelling measurements into complete strokes"""
    print(f"📊 Loading and combining force data: {raw_csv_path}")
    
    all_data = []
    with open(raw_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                raw_json = json.loads(row['raw_json'])
                all_data.append({
                    'timestamp_ns': int(row['ts_ns']),
                    'timestamp_iso': row['ts_iso'],
                    'timestamp_dt': datetime.fromisoformat(row['ts_iso']),
                    'elapsed_s': raw_json.get('time', 0.0),
                    'distance_m': raw_json.get('distance', 0.0),
                    'spm': raw_json.get('spm', 0),
                    'power': raw_json.get('power', 0),
                    'pace': raw_json.get('pace', 0.0),
                    'forceplot': raw_json.get('forceplot', []),
                    'strokestate': raw_json.get('strokestate', ''),
                    'status': raw_json.get('status', '')
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                continue
    
    print(f"   Loaded {len(all_data)} total data points")
    
    # Combine Drive + Dwelling measurements into complete strokes
    combined_strokes = []
    current_stroke = None
    
    for i, data_point in enumerate(all_data):
        if data_point['strokestate'] in ['Drive', 'Dwelling'] and data_point['forceplot']:
            if current_stroke is None:
                # Start a new stroke
                current_stroke = {
                    'start_timestamp_dt': data_point['timestamp_dt'],
                    'start_elapsed_s': data_point['elapsed_s'],
                    'combined_forceplot': data_point['forceplot'].copy(),
                    'measurements': [data_point],
                    'final_power': data_point['power'],
                    'final_spm': data_point['spm'],
                    'final_distance': data_point['distance_m'],
                    'stroke_phases': [data_point['strokestate']]
                }
            else:
                # Continue the current stroke - append forceplot data
                current_stroke['combined_forceplot'].extend(data_point['forceplot'])
                current_stroke['measurements'].append(data_point)
                current_stroke['final_power'] = data_point['power']
                current_stroke['final_spm'] = data_point['spm']
                current_stroke['final_distance'] = data_point['distance_m']
                current_stroke['stroke_phases'].append(data_point['strokestate'])
        
        elif current_stroke is not None and data_point['strokestate'] == 'Recovery':
            # End of current stroke - save it
            current_stroke['end_timestamp_dt'] = data_point['timestamp_dt']
            current_stroke['end_elapsed_s'] = data_point['elapsed_s']
            current_stroke['stroke_duration'] = current_stroke['end_elapsed_s'] - current_stroke['start_elapsed_s']
            combined_strokes.append(current_stroke)
            current_stroke = None
    
    # Don't forget the last stroke if it doesn't end with Recovery
    if current_stroke is not None:
        current_stroke['end_timestamp_dt'] = all_data[-1]['timestamp_dt']
        current_stroke['end_elapsed_s'] = all_data[-1]['elapsed_s']
        current_stroke['stroke_duration'] = current_stroke['end_elapsed_s'] - current_stroke['start_elapsed_s']
        combined_strokes.append(current_stroke)
    
    print(f"   Combined into {len(combined_strokes)} complete strokes")
    
    # Print stroke summary
    for i, stroke in enumerate(combined_strokes):
        phases = " -> ".join(stroke['stroke_phases'])
        print(f"   Stroke {i+1}: {len(stroke['combined_forceplot'])} force points, "
              f"{stroke['stroke_duration']:.2f}s duration, Peak: {max(stroke['combined_forceplot'])}, "
              f"Phases: {phases}")
    
    return combined_strokes

def load_pose_data(pose_json_path):
    """Load pose analysis data from JSON file"""
    print(f"📊 Loading pose data: {pose_json_path}")
    with open(pose_json_path, 'r') as f:
        pose_data = json.load(f)
    print(f"   Loaded {len(pose_data)} pose frames")
    return pose_data

def create_animated_force_curve_plot(force_curve, power, spm, current_idx=None, stroke_num=None, frame_abs_dt=None):
    """Create an animated force curve plot with current position indicator"""
    if not force_curve:
        return None

    # Use smaller size for video overlay
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    x = np.arange(len(force_curve))
    ax.plot(x, force_curve, 'lime', linewidth=3, label='Force')

    # Animated dot showing current position
    if current_idx is None:
        current_idx = int(np.argmax(force_curve))
    current_idx = int(np.clip(current_idx, 0, len(force_curve) - 1))
    current_force = float(force_curve[current_idx])
    
    # Draw animated dot with pulsing effect
    ax.plot(current_idx, current_force, 'ro', markersize=10, label='Current')
    ax.plot(current_idx, current_force, 'ro', markersize=6, alpha=0.7)
    ax.plot(current_idx, current_force, 'ro', markersize=4, alpha=0.4)

    ax.set_xlabel('Stroke Position', color='white', fontsize=10)
    ax.set_ylabel('Force', color='white', fontsize=10)
    
    title = f'Force Curve - {int(power)}W, {int(spm)}spm'
    if stroke_num:
        title = f'Stroke #{stroke_num} - {int(power)}W, {int(spm)}spm'
    ax.set_title(title, color='white', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white', labelsize=8)

    peak_force_val = int(np.max(force_curve))
    stats = f'Peak: {peak_force_val}\nAvg: {np.mean(force_curve):.1f}\nCurrent: {current_force:.0f}'
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            verticalalignment='top', color='white', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    plt.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def create_enhanced_joint_angles_display(pose_frame, frame_num, elapsed_s):
    """Create enhanced joint angles display with more information"""
    img = np.zeros((350, 400, 3), dtype=np.uint8)
    colors = {
        'white': (255, 255, 255), 'lime': (0, 255, 0), 'cyan': (255, 255, 0),
        'yellow': (0, 255, 255), 'orange': (0, 165, 255), 'red': (0, 0, 255)
    }
    cv2.putText(img, "ROWING ANALYSIS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['white'], 2)
    cv2.putText(img, f"Frame: {frame_num}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
    cv2.putText(img, f"Time: {elapsed_s:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
    y_pos = 95; line_height = 25
    
    cv2.putText(img, "ARM ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1); y_pos += line_height
    if pose_frame.get('left_arm_angle') is not None:
        cv2.putText(img, f"  L Arm: {pose_frame['left_arm_angle']:.1f}°", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1); y_pos += line_height
    if pose_frame.get('right_arm_angle') is not None:
        cv2.putText(img, f"  R Arm: {pose_frame['right_arm_angle']:.1f}°", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1); y_pos += line_height
    
    cv2.putText(img, "LEG ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1); y_pos += line_height
    if pose_frame.get('left_leg_angle') is not None:
        cv2.putText(img, f"  L Leg: {pose_frame['left_leg_angle']:.1f}°", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1); y_pos += line_height
    if pose_frame.get('right_leg_angle') is not None:
        cv2.putText(img, f"  R Leg: {pose_frame['right_leg_angle']:.1f}°", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1); y_pos += line_height
    
    cv2.putText(img, "TORSO:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1); y_pos += line_height
    if pose_frame.get('torso_lean_angle') is not None:
        cv2.putText(img, f"  Lean: {pose_frame['torso_lean_angle']:.1f}°", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
    
    return img

def create_angle_trend_chart(current_frame_idx, pose_data, window_size=30):
    """Create a mini chart showing angle trends"""
    if current_frame_idx < 10 or not pose_data:
        return None
    
    # Get recent angle data
    start_idx = max(0, current_frame_idx - window_size)
    recent_frames = pose_data[start_idx:current_frame_idx + 1]
    
    # Extract angles
    left_arm_angles = [f.get('left_arm_angle') for f in recent_frames if f.get('left_arm_angle') is not None]
    right_arm_angles = [f.get('right_arm_angle') for f in recent_frames if f.get('right_arm_angle') is not None]
    
    if not left_arm_angles or not right_arm_angles:
        return None
    
    # Ensure both arrays have the same length
    min_length = min(len(left_arm_angles), len(right_arm_angles))
    left_arm_angles = left_arm_angles[:min_length]
    right_arm_angles = right_arm_angles[:min_length]
    
    # Create mini chart
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    x = np.arange(len(left_arm_angles))
    ax.plot(x, left_arm_angles, 'lime', linewidth=2, label='L Arm')
    ax.plot(x, right_arm_angles, 'cyan', linewidth=2, label='R Arm')
    
    ax.set_title('Arm Angles Trend', color='white', fontsize=10)
    ax.set_ylabel('Degrees', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=6)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    plt.close(fig)
    
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def draw_joint_angles_on_frame(frame, pose_frame):
    """Draw angles at the elbow, knee, hip (and torso lean) directly on the video frame.
    This makes it clear what each angle represents by rendering the numeric value at the
    joint vertex and drawing simple limb segments.
    """
    if not pose_frame:
        return
    
    # Helper to fetch a keypoint triplet
    def get_kpt(name):
        x_key, y_key, c_key = f"{name}_x", f"{name}_y", f"{name}_confidence"
        if x_key in pose_frame and y_key in pose_frame and c_key in pose_frame:
            conf = pose_frame.get(c_key, 0)
            if conf is None:
                conf = 0
            if conf > 0.5:
                return (int(pose_frame[x_key]), int(pose_frame[y_key]))
        return None

    def calc_angle(p1, p2, p3):
        if p1 is None or p2 is None or p3 is None:
            return None
        a = np.array(p1, dtype=np.float32)
        b = np.array(p2, dtype=np.float32)
        c = np.array(p3, dtype=np.float32)
        v1 = a - b
        v2 = c - b
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return None
        cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    def draw_segment(p, q, color, thickness=3):
        if p is not None and q is not None:
            cv2.line(frame, p, q, color, thickness)

    def draw_kpt(p, color):
        if p is not None:
            cv2.circle(frame, p, 6, color, -1)

    def draw_badge(p, text, bg_color, text_color=(0, 0, 0), draw_deg=False):
        if p is None:
            return
        x, y = p
        pad = 6
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x1, y1 = x - tw // 2 - pad, y - th - baseline - pad
        x2, y2 = x + tw // 2 + pad, y + pad
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2)
        y2 = min(frame.shape[0]-1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50,50,50), 1)
        cv2.putText(frame, text, (x1 + pad, y2 - pad - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        # Manually render a degree symbol as a small circle to avoid unicode issues
        if draw_deg:
            cx = x1 + pad + tw + 6
            cy = y1 + th // 3  # slightly above midline
            cx = min(cx, frame.shape[1]-3)
            cy = max(3, min(cy, frame.shape[0]-3))
            cv2.circle(frame, (cx, cy), 4, text_color, 2)

    # Collect keypoints we care about
    ls = get_kpt('left_shoulder')
    rs = get_kpt('right_shoulder')
    le = get_kpt('left_elbow')
    re = get_kpt('right_elbow')
    lw = get_kpt('left_wrist')
    rw = get_kpt('right_wrist')
    lh = get_kpt('left_hip')
    rh = get_kpt('right_hip')
    lk = get_kpt('left_knee')
    rk = get_kpt('right_knee')
    la = get_kpt('left_ankle')
    ra = get_kpt('right_ankle')

    # Colors (BGR)
    color_arm = (0, 255, 255)   # yellow
    color_leg = (255, 153, 0)   # blue-ish? actually BGR; but we'll keep distinct
    color_left = (0, 255, 0)    # green for left joint dot
    color_right = (255, 0, 0)   # red for right joint dot

    # Draw simple skeleton segments for context
    draw_segment(ls, le, color_arm)
    draw_segment(le, lw, color_arm)
    draw_segment(rs, re, color_arm)
    draw_segment(re, rw, color_arm)
    draw_segment(lh, lk, color_leg)
    draw_segment(lk, la, color_leg)
    draw_segment(rh, rk, color_leg)
    draw_segment(rk, ra, color_leg)
    draw_segment(ls, lh, (200,200,200))
    draw_segment(rs, rh, (200,200,200))
    draw_segment(lh, rh, (200,200,200))

    # Draw keypoints
    for p, col in [(le, color_left), (re, color_right), (lk, color_left), (rk, color_right), (lh, (255,255,255))]:
        draw_kpt(p, col)

    # Compute angles and draw badges at the vertex
    ang_left_elbow = calc_angle(ls, le, lw)
    if ang_left_elbow is not None:
        draw_badge(le, f"{ang_left_elbow:.0f}", (0, 255, 255), draw_deg=True)

    ang_right_elbow = calc_angle(rs, re, rw)
    if ang_right_elbow is not None:
        draw_badge(re, f"{ang_right_elbow:.0f}", (0, 255, 255), draw_deg=True)

    ang_left_knee = calc_angle(lh, lk, la)
    if ang_left_knee is not None:
        draw_badge(lk, f"{ang_left_knee:.0f}", (255, 200, 0), draw_deg=True)

    ang_right_knee = calc_angle(rh, rk, ra)
    if ang_right_knee is not None:
        draw_badge(rk, f"{ang_right_knee:.0f}", (255, 200, 0), draw_deg=True)

    # Hip angle (shoulder-hip-knee) — draw at hip
    ang_left_hip = calc_angle(ls, lh, lk)
    if ang_left_hip is not None:
        draw_badge(lh, f"{ang_left_hip:.0f}", (200, 200, 200), draw_deg=True)

    ang_right_hip = calc_angle(rs, rh, rk)
    if ang_right_hip is not None:
        draw_badge(rh, f"{ang_right_hip:.0f}", (200, 200, 200), draw_deg=True)

    # Ankle dorsiflexion/plantarflexion approximation:
    # angle between shank (knee->ankle) and a vertical reference line through the ankle
    def draw_vertical_dotted(p_start, length=110, color=(180, 180, 180)):
        if p_start is None:
            return None
        x, y = p_start
        y2 = max(0, y - length)
        # dotted vertical line
        for t in range(y2, y, 8):
            cv2.line(frame, (x, t), (x, min(y, t+4)), color, 2)
        return (x, y2)

    # Left ankle angle (signed relative to vertical): positive when knee is forward (to the right)
    if la is not None and lk is not None:
        top = draw_vertical_dotted(la)
        if top is not None:
            # Signed angle using reference vector up (vertical) and shank vector
            v_ref = np.array([0.0, -1.0], dtype=np.float32)  # up
            v_shank = np.array([lk[0] - la[0], lk[1] - la[1]], dtype=np.float32)
            n = np.linalg.norm(v_shank)
            if n > 0:
                v_shank /= n
                cross_z = v_ref[0]*v_shank[1] - v_ref[1]*v_shank[0]
                dot = v_ref[0]*v_shank[0] + v_ref[1]*v_shank[1]
                signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                draw_badge(la, f"{signed_deg:.0f}", (255, 255, 0), draw_deg=True)

    # Right ankle angle (signed relative to vertical): positive when knee is forward (to the right)
    if ra is not None and rk is not None:
        top = draw_vertical_dotted(ra)
        if top is not None:
            v_ref = np.array([0.0, -1.0], dtype=np.float32)
            v_shank = np.array([rk[0] - ra[0], rk[1] - ra[1]], dtype=np.float32)
            n = np.linalg.norm(v_shank)
            if n > 0:
                v_shank /= n
                cross_z = v_ref[0]*v_shank[1] - v_ref[1]*v_shank[0]
                dot = v_ref[0]*v_shank[0] + v_ref[1]*v_shank[1]
                signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                draw_badge(ra, f"{signed_deg:.0f}", (255, 255, 0), draw_deg=True)

    # Thigh angle relative to vertical (hip->knee vs vertical), analogous to ankle logic
    # Left thigh
    if lh is not None and lk is not None:
        top = draw_vertical_dotted(lh)
        if top is not None:
            v_ref = np.array([0.0, -1.0], dtype=np.float32)  # up
            v_thigh = np.array([lk[0] - lh[0], lk[1] - lh[1]], dtype=np.float32)
            n = np.linalg.norm(v_thigh)
            if n > 0:
                v_thigh /= n
                cross_z = v_ref[0]*v_thigh[1] - v_ref[1]*v_thigh[0]
                dot = v_ref[0]*v_thigh[0] + v_ref[1]*v_thigh[1]
                signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                # Use orange-ish badge for thigh
                draw_badge(lh, f"{signed_deg:.0f}", (0, 165, 255), draw_deg=True)

    # Right thigh
    if rh is not None and rk is not None:
        top = draw_vertical_dotted(rh)
        if top is not None:
            v_ref = np.array([0.0, -1.0], dtype=np.float32)
            v_thigh = np.array([rk[0] - rh[0], rk[1] - rh[1]], dtype=np.float32)
            n = np.linalg.norm(v_thigh)
            if n > 0:
                v_thigh /= n
                cross_z = v_ref[0]*v_thigh[1] - v_ref[1]*v_thigh[0]
                dot = v_ref[0]*v_thigh[0] + v_ref[1]*v_thigh[1]
                signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                draw_badge(rh, f"{signed_deg:.0f}", (0, 165, 255), draw_deg=True)

    # Torso lean badge near midpoint between shoulders
    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_center = (int((ls[0] + rs[0]) / 2), int((ls[1] + rs[1]) / 2))
        hip_center = (int((lh[0] + rh[0]) / 2), int((lh[1] + rh[1]) / 2))
        # vertical ref from shoulder center straight down
        vertical_ref = (shoulder_center[0], shoulder_center[1] + 100)
        torso_angle = calc_angle(vertical_ref, shoulder_center, hip_center)
        if torso_angle is not None:
            draw_badge(shoulder_center, f"{torso_angle:.0f}", (0, 255, 128), draw_deg=True)

def overlay_display(frame, display, x_offset, y_offset):
    """Overlays a display image onto the video frame"""
    if display is None:
        return
    dh, dw = display.shape[:2]
    fh, fw = frame.shape[:2]
    if (x_offset + dw <= fw and y_offset + dh <= fh):
        frame[y_offset:y_offset+dh, x_offset:x_offset+dw] = display

def find_closest_stroke_for_time(target_time, combined_strokes):
    """Find the closest stroke for a given timestamp and calculate animation position"""
    if not combined_strokes:
        return None, None, None
    
    min_time_diff = float('inf')
    closest_stroke = None
    closest_idx = None
    stroke_num = None
    
    for i, stroke in enumerate(combined_strokes):
        # Check if target time is within stroke duration
        if stroke['start_timestamp_dt'] <= target_time <= stroke['end_timestamp_dt']:
            # Time is within this stroke - calculate position within stroke
            stroke_duration = (stroke['end_timestamp_dt'] - stroke['start_timestamp_dt']).total_seconds()
            time_within_stroke = (target_time - stroke['start_timestamp_dt']).total_seconds()
            
            if stroke_duration > 0:
                progress = time_within_stroke / stroke_duration
                force_idx = int(progress * (len(stroke['combined_forceplot']) - 1))
                force_idx = max(0, min(force_idx, len(stroke['combined_forceplot']) - 1))
                return stroke, force_idx, i + 1
        
        # If not within stroke, find closest
        time_diff = min(
            abs((target_time - stroke['start_timestamp_dt']).total_seconds()),
            abs((target_time - stroke['end_timestamp_dt']).total_seconds())
        )
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_stroke = stroke
            closest_idx = 0  # Default to start of stroke
            stroke_num = i + 1
    
    return closest_stroke, closest_idx, stroke_num

def create_advanced_synchronized_overlay_video(video_path, raw_csv_path, pose_json_path, output_dir="advanced_synchronized_overlay"):
    """Create advanced synchronized overlay video with animated force curves and joint angles"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    combined_strokes = load_and_combine_force_data(raw_csv_path)
    pose_data = load_pose_data(pose_json_path)
    
    if not combined_strokes or not pose_data:
        print("❌ Missing force or pose data. Cannot create overlay video.")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Determine video start time from filename
    video_basename = os.path.basename(video_path)
    match = re.search(r'py3rowcap_(\d{8}_\d{6})', video_basename)
    video_start_dt = None
    if match:
        video_start_dt = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        print(f"📹 Inferred video start time: {video_start_dt.isoformat()}")
    else:
        print("⚠️ Could not infer video start time from filename")
        return
    
    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"advanced_synchronized_overlay_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames with advanced animated overlays...")
    
    frame_count = 0
    overlays_added = 0
    force_overlays = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate current frame's absolute timestamp
        current_frame_elapsed_s = frame_count / fps
        current_frame_abs_time = video_start_dt + timedelta(seconds=current_frame_elapsed_s)
        
        # Get pose data for this frame
        pose_frame = None
        if frame_count <= len(pose_data):
            pose_frame = pose_data[frame_count - 1]
            elapsed_s = float(pose_frame.get('timestamp', current_frame_elapsed_s))
            
            # Create pose analysis display
            angles_display = create_enhanced_joint_angles_display(pose_frame, frame_count, elapsed_s)
            if angles_display is not None:
                overlay_display(frame, angles_display, 10, 10)
                overlays_added += 1
            
            # Create angle trend chart
            angle_chart = create_angle_trend_chart(frame_count - 1, pose_data)
            if angle_chart is not None:
                ch, cw = angle_chart.shape[:2]
                cx = max(0, width - cw - 10)
                cy = max(0, height - ch - 10)
                overlay_display(frame, angle_chart, cx, cy)
            
            # Draw joint angles directly on frame
            draw_joint_angles_on_frame(frame, pose_frame)
        
        # Find closest stroke for current time
        closest_stroke, force_idx, stroke_num = find_closest_stroke_for_time(current_frame_abs_time, combined_strokes)
        
        if closest_stroke and closest_stroke['combined_forceplot']:
            # Create animated force curve plot
            force_plot = create_animated_force_curve_plot(
                closest_stroke['combined_forceplot'],
                closest_stroke['final_power'],
                closest_stroke['final_spm'],
                force_idx,
                stroke_num,
                current_frame_abs_time
            )
            
            if force_plot is not None:
                ph, pw = force_plot.shape[:2]
                px = max(0, width - pw - 10)
                py = 10
                if px + pw <= width and py + ph <= height:
                    overlay_display(frame, force_plot, px, py)
                    force_overlays += 1
        
        # Add frame info
        info_text = f"Frame {frame_count}/{total_frames} | Time: {current_frame_abs_time.strftime('%H:%M:%S.%f')[:-3]}"
        if closest_stroke and stroke_num:
            info_text += f" | Stroke #{stroke_num}"
        cv2.putText(frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n🎉 Advanced synchronized overlay video complete!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   📈 Overlays added: {overlays_added}")
    print(f"   📊 Overlay rate: {(overlays_added/frame_count)*100:.1f}%")
    print(f"   🔋 Force plot overlays: {force_overlays}")
    print(f"   📊 Force overlay rate: {(force_overlays/frame_count)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Create advanced synchronized overlay video with animated force curves")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--raw-csv", required=True, help="Path to raw CSV file")
    parser.add_argument("--pose-json", required=True, help="Path to pose analysis JSON file")
    parser.add_argument("--output-dir", default="advanced_synchronized_overlay", help="Output directory")
    
    args = parser.parse_args()
    
    print("🎬 Creating Advanced Synchronized Overlay Video with Animated Force Curves")
    print("=" * 80)
    
    create_advanced_synchronized_overlay_video(args.video, args.raw_csv, args.pose_json, args.output_dir)

if __name__ == "__main__":
    main()
