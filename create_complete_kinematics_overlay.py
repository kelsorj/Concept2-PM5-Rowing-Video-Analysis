#!/usr/bin/env python3
"""
Create Complete Kinematics Overlay
Runs kinematics analysis on the video first, then creates synchronized overlay with force data
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
from ultralytics import YOLO  # type: ignore
import tempfile
import shutil
from collections import deque
from scipy import ndimage

class PoseSmoother:
    """Advanced pose smoothing to reduce jitter and bouncing in skeleton overlay"""
    
    def __init__(self, window_size=5, confidence_threshold=0.5, outlier_threshold=2.0):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.outlier_threshold = outlier_threshold
        
        # Store recent keypoints for temporal smoothing
        self.keypoint_history = deque(maxlen=window_size)
        self.smoothed_keypoints = None
        
        # Keypoint names for YOLO pose
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def smooth_keypoints(self, keypoints):
        """Apply comprehensive smoothing to keypoints"""
        if keypoints is None or len(keypoints) == 0:
            return None
        
        # Convert to numpy array for easier processing
        kpts_array = np.array(keypoints)
        
        # Filter by confidence
        filtered_kpts = self._filter_by_confidence(kpts_array)
        
        # Add to history
        self.keypoint_history.append(filtered_kpts)
        
        # Apply temporal smoothing
        smoothed_kpts = self._temporal_smoothing()
        
        # Apply outlier rejection
        smoothed_kpts = self._reject_outliers(smoothed_kpts)
        
        # Apply gaussian smoothing
        smoothed_kpts = self._gaussian_smooth(smoothed_kpts)
        
        self.smoothed_keypoints = smoothed_kpts
        return smoothed_kpts
    
    def _filter_by_confidence(self, keypoints):
        """Filter keypoints by confidence threshold"""
        filtered = keypoints.copy()
        for i in range(len(filtered)):
            if filtered[i, 2] < self.confidence_threshold:
                # Set low confidence keypoints to NaN
                filtered[i, 0] = np.nan
                filtered[i, 1] = np.nan
        return filtered
    
    def _temporal_smoothing(self):
        """Apply temporal smoothing using moving average"""
        if len(self.keypoint_history) < 2:
            return self.keypoint_history[-1] if self.keypoint_history else None
        
        # Convert history to numpy array
        history_array = np.array(list(self.keypoint_history))
        
        # Calculate weighted average (more recent frames have higher weight)
        weights = np.linspace(0.5, 1.0, len(self.keypoint_history))
        weights = weights / np.sum(weights)
        
        smoothed = np.zeros_like(history_array[-1])
        
        for i in range(len(self.keypoint_names)):
            # Get x, y coordinates for this keypoint across all frames
            x_coords = history_array[:, i, 0]
            y_coords = history_array[:, i, 1]
            confidences = history_array[:, i, 2]
            
            # Only smooth if we have valid data
            valid_x = ~np.isnan(x_coords)
            valid_y = ~np.isnan(y_coords)
            valid_conf = confidences > self.confidence_threshold
            
            if np.any(valid_x & valid_y & valid_conf):
                # Weighted average of valid coordinates
                valid_weights = weights[valid_x & valid_y & valid_conf]
                valid_weights = valid_weights / np.sum(valid_weights)
                
                smoothed[i, 0] = np.average(x_coords[valid_x & valid_y & valid_conf], weights=valid_weights)
                smoothed[i, 1] = np.average(y_coords[valid_x & valid_y & valid_conf], weights=valid_weights)
                smoothed[i, 2] = np.average(confidences[valid_conf], weights=valid_weights)
            else:
                # Use the most recent valid value
                smoothed[i] = history_array[-1, i]
        
        return smoothed
    
    def _reject_outliers(self, keypoints):
        """Reject outlier keypoints that are too far from expected positions"""
        if keypoints is None or len(self.keypoint_history) < 3:
            return keypoints
        
        # Calculate expected position based on recent history
        recent_history = np.array(list(self.keypoint_history)[-3:])
        expected_positions = np.mean(recent_history, axis=0)
        
        filtered = keypoints.copy()
        
        for i in range(len(keypoints)):
            if not np.isnan(keypoints[i, 0]) and not np.isnan(keypoints[i, 1]):
                # Calculate distance from expected position
                distance = np.sqrt(
                    (keypoints[i, 0] - expected_positions[i, 0])**2 + 
                    (keypoints[i, 1] - expected_positions[i, 1])**2
                )
                
                # If too far from expected, use expected position instead
                if distance > self.outlier_threshold * 50:  # 50 pixels threshold
                    filtered[i, 0] = expected_positions[i, 0]
                    filtered[i, 1] = expected_positions[i, 1]
        
        return filtered
    
    def _gaussian_smooth(self, keypoints):
        """Apply gaussian smoothing to reduce high-frequency noise"""
        if keypoints is None:
            return None
        
        smoothed = keypoints.copy()
        
        # Apply gaussian filter to x and y coordinates
        for i in range(len(keypoints)):
            if not np.isnan(keypoints[i, 0]) and not np.isnan(keypoints[i, 1]):
                # Get recent history for this keypoint
                if len(self.keypoint_history) >= 3:
                    recent_x = [h[i, 0] for h in self.keypoint_history if not np.isnan(h[i, 0])]
                    recent_y = [h[i, 1] for h in self.keypoint_history if not np.isnan(h[i, 1])]
                    
                    if len(recent_x) >= 3:
                        # Apply gaussian filter
                        smoothed[i, 0] = ndimage.gaussian_filter1d(recent_x, sigma=0.5)[-1]
                        smoothed[i, 1] = ndimage.gaussian_filter1d(recent_y, sigma=0.5)[-1]
        
        return smoothed
    
    def get_smoothed_keypoints(self):
        """Get the most recent smoothed keypoints"""
        return self.smoothed_keypoints

def analyze_rowing_pose(keypoints):
    """Analyze rowing pose and calculate body angles"""
    # YOLO pose keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, 
    # left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
    
    analysis = {}
    
    # Extract keypoints with confidence
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Store all keypoint coordinates and confidence
    for i, name in enumerate(keypoint_names):
        if i < len(keypoints):
            kpt = keypoints[i]
            analysis[f'{name}_x'] = float(kpt[0])
            analysis[f'{name}_y'] = float(kpt[1])
            analysis[f'{name}_confidence'] = float(kpt[2])
    
    # Calculate angles
    def calc_angle(p1, p2, p3):
        """Calculate angle between three points"""
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
    
    def get_kpt(name):
        """Get keypoint coordinates if confidence > 0.5"""
        x_key, y_key, c_key = f"{name}_x", f"{name}_y", f"{name}_confidence"
        if x_key in analysis and y_key in analysis and c_key in analysis:
            conf = analysis.get(c_key, 0)
            if conf is None:
                conf = 0
            if conf > 0.5:
                return (analysis[x_key], analysis[y_key])
        return None
    
    # Calculate arm angles (shoulder-elbow-wrist)
    left_shoulder = get_kpt('left_shoulder')
    right_shoulder = get_kpt('right_shoulder')
    left_elbow = get_kpt('left_elbow')
    right_elbow = get_kpt('right_elbow')
    left_wrist = get_kpt('left_wrist')
    right_wrist = get_kpt('right_wrist')
    
    if left_shoulder and left_elbow and left_wrist:
        analysis['left_arm_angle'] = calc_angle(left_shoulder, left_elbow, left_wrist)
    else:
        analysis['left_arm_angle'] = None
    
    if right_shoulder and right_elbow and right_wrist:
        analysis['right_arm_angle'] = calc_angle(right_shoulder, right_elbow, right_wrist)
    else:
        analysis['right_arm_angle'] = None
    
    # Calculate leg angles (hip-knee-ankle)
    left_hip = get_kpt('left_hip')
    right_hip = get_kpt('right_hip')
    left_knee = get_kpt('left_knee')
    right_knee = get_kpt('right_knee')
    left_ankle = get_kpt('left_ankle')
    right_ankle = get_kpt('right_ankle')
    
    if left_hip and left_knee and left_ankle:
        analysis['left_leg_angle'] = calc_angle(left_hip, left_knee, left_ankle)
    else:
        analysis['left_leg_angle'] = None
    
    if right_hip and right_knee and right_ankle:
        analysis['right_leg_angle'] = calc_angle(right_hip, right_knee, right_ankle)
    else:
        analysis['right_leg_angle'] = None
    
    # Calculate torso lean (shoulder-hip angle relative to vertical)
    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        vertical_ref = (shoulder_center[0], shoulder_center[1] + 100)
        analysis['torso_lean_angle'] = calc_angle(vertical_ref, shoulder_center, hip_center)
    else:
        analysis['torso_lean_angle'] = None
    
    # Calculate ankle angles relative to vertical (dorsiflexion/plantarflexion)
    def calc_angle_to_vertical(p1, p2):
        """Calculate angle between line p1-p2 and vertical reference"""
        if p1 is None or p2 is None:
            return None
        # Vector from p1 to p2
        v_line = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        # Vertical reference vector (pointing up)
        v_vertical = np.array([0.0, -1.0], dtype=np.float32)
        
        # Normalize vectors
        n_line = np.linalg.norm(v_line)
        n_vertical = np.linalg.norm(v_vertical)
        
        if n_line == 0 or n_vertical == 0:
            return None
        
        v_line_norm = v_line / n_line
        v_vertical_norm = v_vertical / n_vertical
        
        # Calculate signed angle using cross product
        cross_z = v_vertical_norm[0] * v_line_norm[1] - v_vertical_norm[1] * v_line_norm[0]
        dot = v_vertical_norm[0] * v_line_norm[0] + v_vertical_norm[1] * v_line_norm[1]
        signed_angle = float(np.degrees(np.arctan2(cross_z, dot)))
        
        return signed_angle
    
    # Left ankle angle relative to vertical (shank vector)
    if left_ankle and left_knee:
        analysis['left_ankle_vertical_angle'] = calc_angle_to_vertical(left_ankle, left_knee)
    else:
        analysis['left_ankle_vertical_angle'] = None
    
    # Right ankle angle relative to vertical (shank vector)
    if right_ankle and right_knee:
        analysis['right_ankle_vertical_angle'] = calc_angle_to_vertical(right_ankle, right_knee)
    else:
        analysis['right_ankle_vertical_angle'] = None
    
    # Back angle relative to vertical (torso vector)
    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        analysis['back_vertical_angle'] = calc_angle_to_vertical(hip_center, shoulder_center)
    else:
        analysis['back_vertical_angle'] = None
    
    return analysis

def run_kinematics_analysis(video_path, output_dir="kinematics_analysis"):
    """Run kinematics analysis on the video and return pose data"""
    print(f"🤖 Running kinematics analysis on: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if video exists
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return None
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    
    # Initialize YOLO pose model
    model = YOLO('yolo11n-pose.pt')
    print("🤖 Loaded YOLO11 pose model")
    
    # Initialize pose smoother
    smoother = PoseSmoother(window_size=5, confidence_threshold=0.5, outlier_threshold=2.0)
    print("🎯 Initialized pose smoother for stable skeleton overlay")
    
    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(output_dir, f"pose_data_{timestamp}.json")
    
    # JSON data storage
    json_data = []
    
    frame_count = 0
    print("📊 Processing video frames for kinematics...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Run pose detection
        results = model(frame, verbose=False)
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            # Get keypoints for the first person detected
            raw_keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # Apply smoothing to reduce jitter
            smoothed_keypoints = smoother.smooth_keypoints(raw_keypoints)
            
            # Use smoothed keypoints for analysis
            keypoints_to_use = smoothed_keypoints if smoothed_keypoints is not None else raw_keypoints
            
            # Analyze pose
            pose_analysis = analyze_rowing_pose(keypoints_to_use)
            
            # Add frame info
            pose_analysis['frame_number'] = frame_count
            pose_analysis['timestamp'] = frame_count / fps
            pose_analysis['frame_time'] = datetime.now().isoformat()
            
            # Store for JSON (convert numpy types to Python types)
            json_pose_analysis = {}
            for key, value in pose_analysis.items():
                if hasattr(value, 'item'):  # numpy scalar
                    json_pose_analysis[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    json_pose_analysis[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                else:
                    json_pose_analysis[key] = value
            json_data.append(json_pose_analysis)
        else:
            # No person detected, add empty frame
            empty_frame = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'frame_time': datetime.now().isoformat(),
                'left_arm_angle': None,
                'right_arm_angle': None,
                'left_leg_angle': None,
                'right_leg_angle': None,
                'torso_lean_angle': None
            }
            json_data.append(empty_frame)
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    
    # Save JSON data
    with open(output_json, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Kinematics analysis complete!")
    print(f"   📊 Processed {frame_count} frames")
    print(f"   🎯 Applied pose smoothing for stable skeleton overlay")
    print(f"   💾 Pose data saved: {output_json}")
    
    return output_json

def load_frame_timestamps(frames_csv_path):
    """Load frame timestamps from the CSV file"""
    print(f"📊 Loading frame timestamps: {frames_csv_path}")
    
    frame_timestamps = []
    with open(frames_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_timestamps.append({
                'frame_idx': int(row['frame_idx']),
                'ts_ns': int(row['ts_ns']),
                'ts_iso': row['ts_iso'],
                'timestamp_dt': datetime.fromisoformat(row['ts_iso'])
            })
    
    print(f"   Loaded {len(frame_timestamps)} frame timestamps")
    print(f"   First frame: {frame_timestamps[0]['timestamp_dt'].isoformat()}")
    print(f"   Last frame: {frame_timestamps[-1]['timestamp_dt'].isoformat()}")
    
    return frame_timestamps

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
    
    # Print stroke summary with timing info
    for i, stroke in enumerate(combined_strokes):
        phases = " -> ".join(stroke['stroke_phases'])
        start_time = stroke['start_timestamp_dt'].strftime('%H:%M:%S.%f')[:-3]
        end_time = stroke['end_timestamp_dt'].strftime('%H:%M:%S.%f')[:-3]
        print(f"   Stroke {i+1}: {len(stroke['combined_forceplot'])} force points, "
              f"{stroke['stroke_duration']:.2f}s duration, Peak: {max(stroke['combined_forceplot'])}, "
              f"Time: {start_time} - {end_time}, Phases: {phases}")
    
    return combined_strokes

def create_animated_force_curve_plot(force_curve, power, spm, current_idx=None, stroke_num=None, frame_abs_dt=None):
    """Create an animated force curve plot with current position indicator"""
    if not force_curve:
        return None

    # Use smaller size for video overlay (25% smaller than before)
    fig, ax = plt.subplots(figsize=(3, 2.25), dpi=100)
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
        cv2.putText(img, f"  L Arm: {pose_frame['left_arm_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1); y_pos += line_height
    if pose_frame.get('right_arm_angle') is not None:
        cv2.putText(img, f"  R Arm: {pose_frame['right_arm_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1); y_pos += line_height
    
    cv2.putText(img, "LEG ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1); y_pos += line_height
    if pose_frame.get('left_leg_angle') is not None:
        cv2.putText(img, f"  L Leg: {pose_frame['left_leg_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1); y_pos += line_height
    if pose_frame.get('right_leg_angle') is not None:
        cv2.putText(img, f"  R Leg: {pose_frame['right_leg_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1); y_pos += line_height
    
    cv2.putText(img, "TORSO:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1); y_pos += line_height
    if pose_frame.get('back_vertical_angle') is not None:
        cv2.putText(img, f"  Lean: {pose_frame['back_vertical_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1); y_pos += line_height
    
    cv2.putText(img, "VERTICAL ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1); y_pos += line_height
    if pose_frame.get('left_ankle_vertical_angle') is not None:
        cv2.putText(img, f"  L Ankle: {pose_frame['left_ankle_vertical_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1); y_pos += line_height
    if pose_frame.get('right_ankle_vertical_angle') is not None:
        cv2.putText(img, f"  R Ankle: {pose_frame['right_ankle_vertical_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1); y_pos += line_height
    if pose_frame.get('back_vertical_angle') is not None:
        cv2.putText(img, f"  Back: {pose_frame['back_vertical_angle']:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
    
    return img

def draw_smoothed_skeleton_and_angles(frame, keypoints):
    """Draw smoothed skeleton and joint angles directly on the video frame"""
    if keypoints is None:
        return
    
    def draw_skeleton(keypoints, color, thickness=3):
        """Draw skeleton from keypoints"""
        if keypoints is None:
            return
        
        # Define skeleton connections (COCO format)
        skeleton = [
            (5, 6),   # shoulders
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12), # torso
            (11, 12), # hips
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
        ]
        
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx, 2] > 0.5 and keypoints[end_idx, 2] > 0.5):
                
                start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                
                cv2.line(frame, start_point, end_point, color, thickness)
    
    def draw_keypoints(keypoints, color, radius=6):
        """Draw keypoints as circles"""
        if keypoints is None:
            return
        
        for i, kpt in enumerate(keypoints):
            if kpt[2] > 0.5:  # confidence > 0.5
                center = (int(kpt[0]), int(kpt[1]))
                cv2.circle(frame, center, radius, color, -1)
    
    def calc_angle(p1, p2, p3):
        """Calculate angle between three points"""
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
    
    def draw_badge(p, text, bg_color, text_color=(0, 0, 0)):
        """Draw angle badge at keypoint"""
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
    
    def calc_angle_to_vertical(p1, p2):
        """Calculate angle between line p1-p2 and vertical reference"""
        if p1 is None or p2 is None:
            return None
        # Vector from p1 to p2
        v_line = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        # Vertical reference vector (pointing up)
        v_vertical = np.array([0.0, -1.0], dtype=np.float32)
        
        # Normalize vectors
        n_line = np.linalg.norm(v_line)
        n_vertical = np.linalg.norm(v_vertical)
        
        if n_line == 0 or n_vertical == 0:
            return None
        
        v_line_norm = v_line / n_line
        v_vertical_norm = v_vertical / n_vertical
        
        # Calculate signed angle using cross product
        cross_z = v_vertical_norm[0] * v_line_norm[1] - v_vertical_norm[1] * v_line_norm[0]
        dot = v_vertical_norm[0] * v_line_norm[0] + v_vertical_norm[1] * v_line_norm[1]
        signed_angle = float(np.degrees(np.arctan2(cross_z, dot)))
        
        return signed_angle
    
    # Draw smoothed skeleton in green
    draw_skeleton(keypoints, (0, 255, 0), 3)  # Green skeleton
    
    # Draw key joint points
    draw_keypoints(keypoints, (255, 255, 255), 6)  # White keypoints
    
    # Calculate and draw joint angles
    if len(keypoints) >= 17:  # Ensure we have all keypoints
        # Get key joint positions (ensure integer coordinates)
        ls = (int(keypoints[5, 0]), int(keypoints[5, 1])) if keypoints[5, 2] > 0.5 else None
        rs = (int(keypoints[6, 0]), int(keypoints[6, 1])) if keypoints[6, 2] > 0.5 else None
        le = (int(keypoints[7, 0]), int(keypoints[7, 1])) if keypoints[7, 2] > 0.5 else None
        re = (int(keypoints[8, 0]), int(keypoints[8, 1])) if keypoints[8, 2] > 0.5 else None
        lw = (int(keypoints[9, 0]), int(keypoints[9, 1])) if keypoints[9, 2] > 0.5 else None
        rw = (int(keypoints[10, 0]), int(keypoints[10, 1])) if keypoints[10, 2] > 0.5 else None
        lh = (int(keypoints[11, 0]), int(keypoints[11, 1])) if keypoints[11, 2] > 0.5 else None
        rh = (int(keypoints[12, 0]), int(keypoints[12, 1])) if keypoints[12, 2] > 0.5 else None
        lk = (int(keypoints[13, 0]), int(keypoints[13, 1])) if keypoints[13, 2] > 0.5 else None
        rk = (int(keypoints[14, 0]), int(keypoints[14, 1])) if keypoints[14, 2] > 0.5 else None
        la = (int(keypoints[15, 0]), int(keypoints[15, 1])) if keypoints[15, 2] > 0.5 else None
        ra = (int(keypoints[16, 0]), int(keypoints[16, 1])) if keypoints[16, 2] > 0.5 else None
        
        # Draw elbow angles (yellow badges)
        ang_left_elbow = calc_angle(ls, le, lw)
        if ang_left_elbow is not None:
            draw_badge(le, f"{ang_left_elbow:.0f}", (0, 255, 255))  # Yellow
        
        ang_right_elbow = calc_angle(rs, re, rw)
        if ang_right_elbow is not None:
            draw_badge(re, f"{ang_right_elbow:.0f}", (0, 255, 255))  # Yellow
        
        # Draw knee angles (orange badges)
        ang_left_knee = calc_angle(lh, lk, la)
        if ang_left_knee is not None:
            draw_badge(lk, f"{ang_left_knee:.0f}", (255, 200, 0))  # Orange
        
        ang_right_knee = calc_angle(rh, rk, ra)
        if ang_right_knee is not None:
            draw_badge(rk, f"{ang_right_knee:.0f}", (255, 200, 0))  # Orange
        
        # Draw hip angles (gray badges)
        ang_left_hip = calc_angle(ls, lh, lk)
        if ang_left_hip is not None:
            draw_badge(lh, f"{ang_left_hip:.0f}", (200, 200, 200))  # Gray
        
        ang_right_hip = calc_angle(rs, rh, rk)
        if ang_right_hip is not None:
            draw_badge(rh, f"{ang_right_hip:.0f}", (200, 200, 200))  # Gray
        
        # Draw ankle vertical angles (yellow badges)
        if la is not None and lk is not None:
            left_ankle_vertical = calc_angle_to_vertical(la, lk)
            if left_ankle_vertical is not None:
                draw_badge(la, f"{left_ankle_vertical:.0f}", (255, 255, 0))  # Yellow
        
        if ra is not None and rk is not None:
            right_ankle_vertical = calc_angle_to_vertical(ra, rk)
            if right_ankle_vertical is not None:
                draw_badge(ra, f"{right_ankle_vertical:.0f}", (255, 255, 0))  # Yellow
        
        # Draw back vertical angle (green badge)
        if ls is not None and rs is not None and lh is not None and rh is not None:
            shoulder_center = (int((ls[0] + rs[0]) / 2), int((ls[1] + rs[1]) / 2))
            hip_center = (int((lh[0] + rh[0]) / 2), int((lh[1] + rh[1]) / 2))
            back_vertical = calc_angle_to_vertical(hip_center, shoulder_center)
            if back_vertical is not None:
                draw_badge(shoulder_center, f"{back_vertical:.0f}", (0, 255, 128))  # Green

def draw_joint_angles_on_frame(frame, pose_frame):
    """Legacy function - now redirects to smoothed skeleton drawing"""
    # This function is kept for compatibility but now uses the smoothed approach
    # Convert pose_frame back to keypoints format if needed
    if not pose_frame:
        return
    
    # Extract keypoints from pose_frame
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    keypoints = np.zeros((17, 3))
    for i, name in enumerate(keypoint_names):
        x_key, y_key, c_key = f"{name}_x", f"{name}_y", f"{name}_confidence"
        if x_key in pose_frame and y_key in pose_frame and c_key in pose_frame:
            keypoints[i, 0] = pose_frame[x_key]
            keypoints[i, 1] = pose_frame[y_key]
            keypoints[i, 2] = pose_frame[c_key]
    
    # Use the new smoothed skeleton drawing
    draw_smoothed_skeleton_and_angles(frame, keypoints)

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

def generate_comprehensive_report(pose_data, combined_strokes, frame_timestamps, output_dir):
    """Generate a comprehensive report of body angles relative to forces"""
    print("\n📊 Generating Comprehensive Analysis Report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"rowing_analysis_report_{timestamp}.txt")
    csv_path = os.path.join(output_dir, f"rowing_analysis_data_{timestamp}.csv")
    
    # Create comprehensive report
    with open(report_path, 'w') as f:
        f.write("ROWING BIOMECHANICAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Frames Analyzed: {len(pose_data)}\n")
        f.write(f"Total Strokes: {len(combined_strokes)}\n")
        f.write(f"Analysis Duration: {frame_timestamps[-1]['timestamp_dt'] - frame_timestamps[0]['timestamp_dt']}\n\n")
        
        # Stroke summary
        f.write("STROKE SUMMARY\n")
        f.write("-" * 20 + "\n")
        for i, stroke in enumerate(combined_strokes):
            f.write(f"Stroke {i+1}:\n")
            f.write(f"  Duration: {stroke['stroke_duration']:.2f}s\n")
            f.write(f"  Peak Force: {max(stroke['combined_forceplot'])}\n")
            f.write(f"  Average Power: {stroke['final_power']}W\n")
            f.write(f"  Stroke Rate: {stroke['final_spm']} spm\n")
            f.write(f"  Time: {stroke['start_timestamp_dt'].strftime('%H:%M:%S.%f')[:-3]} - {stroke['end_timestamp_dt'].strftime('%H:%M:%S.%f')[:-3]}\n")
            f.write(f"  Phases: {' -> '.join(stroke['stroke_phases'])}\n\n")
        
        # Body angle statistics
        f.write("BODY ANGLE STATISTICS\n")
        f.write("-" * 25 + "\n")
        
        # Calculate statistics for each angle
        angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle', 
                 'back_vertical_angle', 'left_ankle_vertical_angle', 'right_ankle_vertical_angle']
        
        for angle in angles:
            values = [frame.get(angle) for frame in pose_data if frame.get(angle) is not None]
            if values:
                f.write(f"{angle.replace('_', ' ').title()}:\n")
                f.write(f"  Count: {len(values)}\n")
                f.write(f"  Mean: {np.mean(values):.1f}\n")
                f.write(f"  Std Dev: {np.std(values):.1f}\n")
                f.write(f"  Min: {np.min(values):.1f}\n")
                f.write(f"  Max: {np.max(values):.1f}\n")
                f.write(f"  Range: {np.max(values) - np.min(values):.1f}\n\n")
        
        # Force-angle correlations
        f.write("FORCE-ANGLE CORRELATIONS\n")
        f.write("-" * 25 + "\n")
        
        # For each stroke, find the corresponding pose data and analyze correlations
        for i, stroke in enumerate(combined_strokes):
            f.write(f"Stroke {i+1} Analysis:\n")
            
            # Find pose frames during this stroke
            stroke_start = stroke['start_timestamp_dt']
            stroke_end = stroke['end_timestamp_dt']
            
            stroke_frames = []
            for j, frame_ts in enumerate(frame_timestamps):
                if stroke_start <= frame_ts['timestamp_dt'] <= stroke_end and j < len(pose_data):
                    stroke_frames.append((j, pose_data[j]))
            
            if stroke_frames:
                f.write(f"  Frames during stroke: {len(stroke_frames)}\n")
                
                # Calculate average angles during stroke
                for angle in angles:
                    values = [frame.get(angle) for _, frame in stroke_frames if frame.get(angle) is not None]
                    if values:
                        f.write(f"  {angle.replace('_', ' ').title()}: {np.mean(values):.1f} (avg)\n")
                
                # Force curve analysis
                force_curve = stroke['combined_forceplot']
                if force_curve:
                    f.write(f"  Force Curve Points: {len(force_curve)}\n")
                    f.write(f"  Peak Force: {max(force_curve)}\n")
                    f.write(f"  Average Force: {np.mean(force_curve):.1f}\n")
                    f.write(f"  Force Range: {max(force_curve) - min(force_curve)}\n")
            f.write("\n")
    
    # Create detailed CSV data
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['frame_number', 'timestamp', 'frame_time'] + angles + ['force_peak', 'force_avg', 'stroke_number', 'stroke_phase']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (frame_ts, pose_frame) in enumerate(zip(frame_timestamps, pose_data)):
            row = {
                'frame_number': i,
                'timestamp': frame_ts['timestamp_dt'].isoformat(),
                'frame_time': pose_frame.get('frame_time', ''),
            }
            
            # Add angle data
            for angle in angles:
                row[angle] = pose_frame.get(angle)
            
            # Find which stroke this frame belongs to
            current_time = frame_ts['timestamp_dt']
            stroke_num = None
            stroke_phase = None
            force_peak = None
            force_avg = None
            
            for j, stroke in enumerate(combined_strokes):
                if stroke['start_timestamp_dt'] <= current_time <= stroke['end_timestamp_dt']:
                    stroke_num = j + 1
                    force_curve = stroke['combined_forceplot']
                    if force_curve:
                        force_peak = max(force_curve)
                        force_avg = np.mean(force_curve)
                    
                    # Determine phase within stroke
                    stroke_duration = (stroke['end_timestamp_dt'] - stroke['start_timestamp_dt']).total_seconds()
                    time_in_stroke = (current_time - stroke['start_timestamp_dt']).total_seconds()
                    progress = time_in_stroke / stroke_duration if stroke_duration > 0 else 0
                    
                    if progress < 0.3:
                        stroke_phase = "Drive"
                    elif progress < 0.7:
                        stroke_phase = "Dwelling"
                    else:
                        stroke_phase = "Recovery"
                    break
            
            row['force_peak'] = force_peak
            row['force_avg'] = force_avg
            row['stroke_number'] = stroke_num
            row['stroke_phase'] = stroke_phase
            
            writer.writerow(row)
    
    print(f"   📋 Comprehensive report: {report_path}")
    print(f"   📊 Detailed CSV data: {csv_path}")
    
    return report_path, csv_path

def create_complete_kinematics_overlay(video_path, frames_csv_path, raw_csv_path, output_dir="complete_kinematics_overlay"):
    """Create complete overlay video with kinematics analysis and force data"""
    print("🎬 Creating Complete Kinematics Overlay")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Run kinematics analysis on the video
    print("\n📊 Step 1: Running Kinematics Analysis")
    pose_json_path = run_kinematics_analysis(video_path, output_dir)
    if not pose_json_path:
        print("❌ Kinematics analysis failed")
        return
    
    # Step 2: Load all data
    print("\n📊 Step 2: Loading Data")
    frame_timestamps = load_frame_timestamps(frames_csv_path)
    combined_strokes = load_and_combine_force_data(raw_csv_path)
    
    with open(pose_json_path, 'r') as f:
        pose_data = json.load(f)
    print(f"📊 Loaded {len(pose_data)} pose frames from kinematics analysis")
    
    if not frame_timestamps or not combined_strokes or not pose_data:
        print("❌ Missing required data. Cannot create overlay video.")
        return
    
    # Step 3: Create overlay video with real-time smoothing
    print("\n🎬 Step 3: Creating Overlay Video with Real-time Smoothing")
    
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
    
    # Initialize YOLO model and smoother for real-time overlay
    model = YOLO('yolo11n-pose.pt')
    smoother = PoseSmoother(window_size=5, confidence_threshold=0.5, outlier_threshold=2.0)
    print("🎯 Initialized real-time pose smoother for stable skeleton overlay")
    
    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"complete_kinematics_overlay_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames with smoothed skeleton overlay and force data...")
    
    frame_count = 0
    overlays_added = 0
    force_overlays = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the actual timestamp for this frame from the CSV
        if frame_count < len(frame_timestamps):
            frame_timestamp = frame_timestamps[frame_count]['timestamp_dt']
        else:
            # Fallback if we run out of timestamps
            frame_timestamp = frame_timestamps[-1]['timestamp_dt']
        
        # Run real-time pose detection and smoothing for overlay
        results = model(frame, verbose=False)
        if results and len(results) > 0 and results[0].keypoints is not None:
            # Get keypoints for the first person detected
            raw_keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # Apply smoothing to reduce jitter in overlay
            smoothed_keypoints = smoother.smooth_keypoints(raw_keypoints)
            
            # Use smoothed keypoints for overlay
            keypoints_to_use = smoothed_keypoints if smoothed_keypoints is not None else raw_keypoints
            
            # Create pose analysis from smoothed keypoints
            pose_analysis = analyze_rowing_pose(keypoints_to_use)
            
            # Create pose analysis display
            elapsed_s = frame_count / fps
            angles_display = create_enhanced_joint_angles_display(pose_analysis, frame_count, elapsed_s)
            if angles_display is not None:
                overlay_display(frame, angles_display, 10, 10)
                overlays_added += 1
            
            # Draw smoothed skeleton and angles directly on the video frame
            draw_smoothed_skeleton_and_angles(frame, keypoints_to_use)
        
        # Find closest stroke for current frame timestamp
        closest_stroke, force_idx, stroke_num = find_closest_stroke_for_time(frame_timestamp, combined_strokes)
        
        if closest_stroke and closest_stroke['combined_forceplot']:
            # Create animated force curve plot
            force_plot = create_animated_force_curve_plot(
                closest_stroke['combined_forceplot'],
                closest_stroke['final_power'],
                closest_stroke['final_spm'],
                force_idx,
                stroke_num,
                frame_timestamp
            )
            
            if force_plot is not None:
                ph, pw = force_plot.shape[:2]
                px = max(0, width - pw - 10)
                py = 10
                if px + pw <= width and py + ph <= height:
                    overlay_display(frame, force_plot, px, py)
                    force_overlays += 1
        
        # Add frame info with actual timestamp
        info_text = f"Frame {frame_count}/{total_frames} | Time: {frame_timestamp.strftime('%H:%M:%S.%f')[:-3]}"
        if closest_stroke and stroke_num:
            info_text += f" | Stroke #{stroke_num}"
        cv2.putText(frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add smoothing legend
        cv2.putText(frame, "Smoothed Skeleton", (width - 200, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Generate comprehensive report
    report_path, csv_path = generate_comprehensive_report(pose_data, combined_strokes, frame_timestamps, output_dir)
    
    print(f"\n🎉 Complete kinematics overlay video created!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   📈 Overlays added: {overlays_added}")
    print(f"   📊 Overlay rate: {(overlays_added/frame_count)*100:.1f}%")
    print(f"   🔋 Force plot overlays: {force_overlays}")
    print(f"   📊 Force overlay rate: {(force_overlays/frame_count)*100:.1f}%")
    print(f"   🎯 Applied real-time pose smoothing for stable skeleton")
    print(f"   🤖 Pose data from: {pose_json_path}")
    print(f"   📋 Analysis report: {report_path}")
    print(f"   📊 Detailed data: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Create complete kinematics overlay with force data")
    parser.add_argument("--session-dir", required=True, help="Path to session directory containing video and CSV files")
    parser.add_argument("--output-dir", help="Output directory (defaults to session directory)")
    
    args = parser.parse_args()
    
    # Validate session directory
    if not os.path.exists(args.session_dir):
        print(f"❌ Session directory not found: {args.session_dir}")
        return
    
    # Find files in session directory
    session_dir = args.session_dir
    video_files = glob.glob(os.path.join(session_dir, "*.mp4"))
    frames_csv_files = glob.glob(os.path.join(session_dir, "*_frames.csv"))
    raw_csv_files = glob.glob(os.path.join(session_dir, "*_raw.csv"))
    
    if not video_files:
        print(f"❌ No video file found in {session_dir}")
        return
    if not frames_csv_files:
        print(f"❌ No frames CSV file found in {session_dir}")
        return
    if not raw_csv_files:
        print(f"❌ No raw CSV file found in {session_dir}")
        return
    
    video_path = video_files[0]
    frames_csv_path = frames_csv_files[0]
    raw_csv_path = raw_csv_files[0]
    
    # Use session directory as output if not specified, but create analysis in current directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create analysis directory in current working directory with session name
        session_name = os.path.basename(session_dir.rstrip('/'))
        output_dir = f"analysis_{session_name}"
    
    print("🚣‍♂️ Complete Kinematics Overlay with Force Data")
    print("=" * 60)
    print(f"📁 Session directory: {session_dir}")
    print(f"📹 Video file: {os.path.basename(video_path)}")
    print(f"📊 Frames CSV: {os.path.basename(frames_csv_path)}")
    print(f"📊 Raw CSV: {os.path.basename(raw_csv_path)}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    create_complete_kinematics_overlay(video_path, frames_csv_path, raw_csv_path, output_dir)

if __name__ == "__main__":
    main()
