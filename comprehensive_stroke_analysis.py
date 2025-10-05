#!/usr/bin/env python3
"""
Comprehensive Stroke Analysis Generator
Creates a single visualization with video frames and sequence plot combined
"""

import cv2
import pandas as pd
import numpy as np
import os
import json
import glob
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse

class ComprehensiveStrokeAnalysis:
    """Generate comprehensive stroke analysis with video frames and sequence plot"""
    
    def __init__(self, force_mapping='overlay'):
        # force_mapping: 'even' to match PM5 overlay shape, 'timestamps' to align by time,
        # 'overlay' to reproduce the exact overlay shape without interpolation
        self.force_mapping = force_mapping
    
    def find_video_file(self, analysis_dir):
        """Find the original video file for the analysis"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(analysis_dir, f"*{ext}")))
        
        if video_files:
            return video_files[0]
        
        # Look in parent directory for original capture
        parent_dir = os.path.dirname(analysis_dir)
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(parent_dir, f"*{ext}")))
        
        # Look in current working directory
        for ext in video_extensions:
            video_files.extend(glob.glob(f"*{ext}"))
        
        if video_files:
            return video_files[0]
        
        print("❌ No video file found")
        return None
    
    def load_analysis_data(self, analysis_dir):
        """Load analysis data and find corresponding video"""
        print(f"📊 Loading analysis data from: {analysis_dir}")
        
        # Find data files
        csv_files = glob.glob(os.path.join(analysis_dir, "*_data_*.csv"))
        
        if not csv_files:
            print("❌ No CSV data files found")
            return None
        
        # Load CSV data
        df = pd.read_csv(csv_files[0])
        
        # Find video file
        video_path = self.find_video_file(analysis_dir)
        if not video_path:
            return None
        
        print(f"🎥 Found video: {video_path}")
        
        # Load force data from the original capture directory
        # The analysis directory name should match the original capture directory
        analysis_name = os.path.basename(analysis_dir)
        if analysis_name.startswith('analysis_'):
            original_name = analysis_name[9:]  # Remove 'analysis_' prefix
            original_dir = os.path.join(os.getcwd(), original_name)
            raw_csv_files = glob.glob(os.path.join(original_dir, "*_raw.csv"))
            frames_csv_files = glob.glob(os.path.join(original_dir, "*_frames.csv"))
        
        # Also check in the analysis directory itself
        if not raw_csv_files:
            raw_csv_files = glob.glob(os.path.join(analysis_dir, "*_raw.csv"))
        if not 'frames_csv_files' in locals() or not frames_csv_files:
            frames_csv_files = glob.glob(os.path.join(analysis_dir, "*_frames.csv"))
        
        # Prefer combined JSON if present (highest fidelity, saved by overlay pipeline)
        combined_strokes = None
        combined_json_path = os.path.join(analysis_dir, 'pm5_combined_strokes.json')
        if os.path.exists(combined_json_path):
            try:
                print(f"📊 Loading combined strokes JSON: {combined_json_path}")
                with open(combined_json_path, 'r') as jf:
                    data = json.load(jf)
                # Convert JSON back to the structure used by downstream consumers
                combined_strokes = []
                for s in data:
                    stroke = {
                        'start_timestamp_dt': datetime.fromisoformat(s['start_timestamp_iso']),
                        'end_timestamp_dt': datetime.fromisoformat(s['end_timestamp_iso']),
                        'start_elapsed_s': s.get('start_elapsed_s'),
                        'end_elapsed_s': s.get('end_elapsed_s'),
                        'stroke_duration': s.get('stroke_duration'),
                        'final_power': s.get('final_power'),
                        'final_spm': s.get('final_spm'),
                        'final_distance': s.get('final_distance'),
                        'stroke_phases': s.get('stroke_phases', []),
                        'combined_forceplot': s.get('combined_forceplot', []),
                        'measurements': []
                    }
                    for m in s.get('measurements', []):
                        stroke['measurements'].append({
                            'timestamp_dt': datetime.fromisoformat(m.get('timestamp_iso')) if m.get('timestamp_iso') else None,
                            'elapsed_s': m.get('elapsed_s'),
                            'distance_m': m.get('distance_m'),
                            'spm': m.get('spm'),
                            'power': m.get('power'),
                            'forceplot': m.get('forceplot', []),
                            'strokestate': m.get('strokestate')
                        })
                    combined_strokes.append(stroke)
                print(f"   Loaded {len(combined_strokes)} strokes from combined JSON")
            except Exception as e:
                print(f"   ⚠️  Failed to load combined strokes JSON, will fallback to raw CSV: {e}")

        # If JSON was not available or was empty, fall back to raw CSV; otherwise warn
        if combined_strokes is None or (isinstance(combined_strokes, list) and len(combined_strokes) == 0):
            if raw_csv_files:
                print(f"📊 Loading force data from: {raw_csv_files[0]}")
                combined_strokes = self.load_and_combine_force_data(raw_csv_files[0])
            else:
                print("⚠️  No force data found - will use theoretical curves")

        # Try to load pose JSON frames to support extra metrics (e.g., handle height)
        pose_json_files = sorted(glob.glob(os.path.join(analysis_dir, "pose_data_*.json")))
        pose_frames = None
        if pose_json_files:
            try:
                with open(pose_json_files[-1], 'r') as pf:
                    pose_frames = json.load(pf)
                print(f"📊 Loaded pose frames JSON: {os.path.basename(pose_json_files[-1])}")
            except Exception as e:
                print(f"⚠️  Failed to load pose frames JSON: {e}")
        
        # Load frame timestamps if available (for metrics alignment)
        frame_timestamps = None
        if frames_csv_files:
            try:
                frame_timestamps = []
                with open(frames_csv_files[0], 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            frame_timestamps.append({
                                'frame_idx': int(row.get('frame_idx') or row.get('frame_number') or 0),
                                'timestamp_dt': datetime.fromisoformat(row['ts_iso'])
                            })
                        except Exception:
                            continue
                print(f"📊 Loaded {len(frame_timestamps)} frame timestamps for alignment")
            except Exception as e:
                print(f"⚠️  Failed to load frame timestamps CSV: {e}")

        return {
            'dataframe': df,
            'video_path': video_path,
            'directory': analysis_dir,
            'combined_strokes': combined_strokes,
            'pose_frames': pose_frames,
            'frame_timestamps': frame_timestamps
        }
    
    def load_and_combine_force_data(self, raw_csv_path):
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
    
    def extract_key_frames_for_stroke(self, data, stroke_number, num_frames=4):
        """Extract key frames representing different phases of a stroke using kinematic detection"""
        df = data['dataframe']
        video_path = data['video_path']
        combined_strokes = data.get('combined_strokes')
        pose_frames = data.get('pose_frames')
        frame_timestamps = data.get('frame_timestamps')
        
        # Get stroke data from dataframe (for angles)
        stroke_data = df[df['stroke_number'] == stroke_number]
        if len(stroke_data) == 0:
            print(f"❌ No data found for stroke {stroke_number}")
            return []
        
        # Use kinematic detection to find actual catch and finish frames
        if combined_strokes and pose_frames and frame_timestamps and stroke_number <= len(combined_strokes):
            # Use the same kinematic logic as compute_coaching_metrics_for_stroke
            stroke = combined_strokes[stroke_number - 1]
            
            # Map pose frames to timestamps
            mapped_frames = []
            limit = min(len(pose_frames), len(frame_timestamps))
            for i in range(limit):
                mapped_frames.append({
                    'ts': frame_timestamps[i]['timestamp_dt'],
                    'frame_idx': frame_timestamps[i]['frame_idx'],
                    'pose': pose_frames[i]
                })
            
            # Expand window around stroke (more before to catch the actual catch)
            stroke_start_expanded = stroke['start_timestamp_dt'] - pd.Timedelta(seconds=0.7)
            stroke_end_expanded = stroke['end_timestamp_dt'] + pd.Timedelta(seconds=0.3)
            
            # Get frames and hip positions in window
            window_frames = []
            window_hip_x = []
            for item in mapped_frames:
                if stroke_start_expanded <= item['ts'] <= stroke_end_expanded:
                    fr = item['pose']
                    left_hip_x = fr.get('left_hip_x')
                    right_hip_x = fr.get('right_hip_x')
                    left_conf = fr.get('left_hip_confidence', 0)
                    right_conf = fr.get('right_hip_confidence', 0)
                    if left_hip_x is not None and right_hip_x is not None and left_conf > 0.5 and right_conf > 0.5:
                        window_frames.append(item)
                        window_hip_x.append((left_hip_x + right_hip_x) / 2.0)
            
            if len(window_frames) >= 5:
                from scipy.ndimage import gaussian_filter1d
                smoothed = gaussian_filter1d(np.array(window_hip_x), sigma=0.8)
                
                # Determine direction
                all_hip_x = []
                for item in mapped_frames:
                    fr = item['pose']
                    left_hip_x = fr.get('left_hip_x')
                    right_hip_x = fr.get('right_hip_x')
                    left_conf = fr.get('left_hip_confidence', 0)
                    right_conf = fr.get('right_hip_confidence', 0)
                    if left_hip_x and right_hip_x and left_conf > 0.5 and right_conf > 0.5:
                        all_hip_x.append((left_hip_x + right_hip_x) / 2.0)
                
                # Determine direction from the window itself, not all frames
                drive_direction = 'left' if window_hip_x[0] > window_hip_x[-1] else 'right'
                
                # Find catch and finish
                search_end = int(len(smoothed) * 0.6)
                if drive_direction == 'left':
                    catch_idx = int(np.argmax(smoothed[:search_end]))
                    finish_idx = catch_idx + int(np.argmin(smoothed[catch_idx:]))
                else:
                    catch_idx = int(np.argmin(smoothed[:search_end]))
                    finish_idx = catch_idx + int(np.argmax(smoothed[catch_idx:]))
                
                catch_frame_num = window_frames[catch_idx]['frame_idx']
                finish_frame_num = window_frames[finish_idx]['frame_idx']
                
                # Select 4 frames for the classical rowing phases
                frame_range = finish_frame_num - catch_frame_num
                if frame_range > 0:
                    # Catch, mid-Drive, Finish, mid-Recovery
                    drive_mid = catch_frame_num + frame_range // 2
                    # Estimate recovery frame (finish + similar duration to drive)
                    recovery_mid = finish_frame_num + frame_range // 2
                    selected_frames = [
                        catch_frame_num,      # Catch
                        drive_mid,            # Drive (middle of power application)
                        finish_frame_num,     # Finish
                        recovery_mid          # Recovery (middle of return)
                    ]
                else:
                    # Fallback to dataframe frames
                    frame_numbers = stroke_data['frame_number'].tolist()
                    indices = np.linspace(0, len(frame_numbers)-1, num_frames, dtype=int)
                    selected_frames = [frame_numbers[i] for i in indices]
            else:
                # Fallback to dataframe frames
                frame_numbers = stroke_data['frame_number'].tolist()
                indices = np.linspace(0, len(frame_numbers)-1, num_frames, dtype=int)
                selected_frames = [frame_numbers[i] for i in indices]
        else:
            # Fallback to dataframe frames
            frame_numbers = stroke_data['frame_number'].tolist()
            indices = np.linspace(0, len(frame_numbers)-1, num_frames, dtype=int)
            selected_frames = [frame_numbers[i] for i in indices]
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for frame_num in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # Get corresponding data for this frame
                frame_data = df[df['frame_number'] == frame_num]
                if len(frame_data) > 0:
                    frames.append({
                        'frame': frame,
                        'frame_number': frame_num,
                        'data': frame_data.iloc[0],
                        'timestamp': frame_data.iloc[0]['timestamp'] if 'timestamp' in frame_data.columns else None
                    })
                else:
                    # Frame not in dataframe - get angles from pose_frames
                    if pose_frames and frame_timestamps:
                        # Find this frame in pose data
                        pose_data = None
                        for i, ft in enumerate(frame_timestamps):
                            if ft['frame_idx'] == frame_num and i < len(pose_frames):
                                pose_data = pose_frames[i]
                                break
                        
                        if pose_data:
                            # Create a Series with the angle data from pose
                            frame_series = pd.Series({
                                'frame_number': frame_num,
                                'left_arm_angle': pose_data.get('left_arm_angle'),
                                'right_arm_angle': pose_data.get('right_arm_angle'),
                                'left_leg_angle': pose_data.get('left_leg_angle'),
                                'right_leg_angle': pose_data.get('right_leg_angle'),
                                'back_vertical_angle': pose_data.get('back_vertical_angle'),
                                'left_ankle_vertical_angle': pose_data.get('left_ankle_vertical_angle'),
                                'right_ankle_vertical_angle': pose_data.get('right_ankle_vertical_angle')
                            })
                            frames.append({
                                'frame': frame,
                                'frame_number': frame_num,
                                'data': frame_series,
                                'timestamp': None
                            })
                        else:
                            # No pose data found
                            frames.append({
                                'frame': frame,
                                'frame_number': frame_num,
                                'data': pd.Series(),
                                'timestamp': None
                            })
                    else:
                        # No pose data available
                        frames.append({
                            'frame': frame,
                            'frame_number': frame_num,
                            'data': pd.Series(),
                            'timestamp': None
                        })
        
        cap.release()
        return frames
    
    def overlay_angles_on_frame(self, frame, frame_data):
        """Overlay angle measurements on a video frame"""
        # Create a copy of the frame
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Define angle measurements to display
        angles = {
            'L Arm': frame_data.get('left_arm_angle', None),
            'R Arm': frame_data.get('right_arm_angle', None),
            'L Leg': frame_data.get('left_leg_angle', None),
            'R Leg': frame_data.get('right_leg_angle', None),
            'Back': frame_data.get('back_vertical_angle', None),
            'L Ankle': frame_data.get('left_ankle_vertical_angle', None),
            'R Ankle': frame_data.get('right_ankle_vertical_angle', None)
        }
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 255, 0)  # Green text
        thickness = 1
        
        # Position for angle display (bottom of frame)
        start_x = 10
        start_y = height - 20
        line_height = 12
        
        # Draw background rectangle
        cv2.rectangle(overlay_frame, (5, start_y - len(angles) * line_height - 5), 
                     (width - 5, height - 5), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (5, start_y - len(angles) * line_height - 5), 
                     (width - 5, height - 5), (255, 255, 255), 1)
        
        # Draw angle measurements
        for i, (angle_name, angle_value) in enumerate(angles.items()):
            if angle_value is not None and not pd.isna(angle_value):
                text = f"{angle_name}: {angle_value:.0f}°"
                y_pos = start_y - (len(angles) - i - 1) * line_height
                cv2.putText(overlay_frame, text, (start_x, y_pos), font, font_scale, color, thickness)
        
        # Add frame number
        frame_text = f"F#{frame_data.get('frame_number', 'N/A')}"
        cv2.putText(overlay_frame, frame_text, (width - 50, 15), font, font_scale, (255, 255, 255), thickness)
        
        return overlay_frame
    
    def calculate_stroke_sequence_data(self, stroke_data, stroke_number, combined_strokes):
        """Calculate speed and sequence data for stroke timing analysis"""
        # Get all frames for this stroke
        df = stroke_data['data']
        
        # Get the actual stroke timing from combined_strokes
        stroke_start_time = None
        stroke_end_time = None
        stroke_force_data = None
        
        if combined_strokes and stroke_number <= len(combined_strokes):
            stroke_info = combined_strokes[stroke_number - 1]
            stroke_start_time = stroke_info['start_timestamp_dt']
            stroke_end_time = stroke_info['end_timestamp_dt']
            stroke_force_data = stroke_info['combined_forceplot']
        
        # Create time axis based on actual timestamps
        if 'timestamp' in df.columns:
            # Convert timestamps to relative time within stroke (0 to 1)
            timestamps = pd.to_datetime(df['timestamp'])
            if stroke_start_time and stroke_end_time:
                stroke_duration = (stroke_end_time - stroke_start_time).total_seconds()
                time_axis = [(ts - stroke_start_time).total_seconds() / stroke_duration for ts in timestamps]
                time_axis = np.array(time_axis)
            else:
                # Fallback to linear time axis
                time_axis = np.linspace(0, 1, len(df))
        else:
            # Fallback to linear time axis
            time_axis = np.linspace(0, 1, len(df))
        
        # Create dramatic, realistic rowing curves based on biomechanics
        # These should show the actual motion patterns of rowing
        
        # Legs: Strong positive peak early in drive, then negative in recovery
        leg_contribution = np.zeros_like(time_axis)
        
        # Drive phase (0 to 0.5): Strong leg extension
        drive_mask = time_axis <= 0.5
        # Peak at t=0.2, then decrease
        leg_curve = np.exp(-((time_axis[drive_mask] - 0.2) / 0.15) ** 2)
        leg_contribution[drive_mask] = leg_curve * 0.9  # High positive contribution
        
        # Recovery phase (0.5 to 1): Leg compression (negative)
        recovery_mask = time_axis > 0.5
        # Negative peak at t=0.8
        recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.8) / 0.12) ** 2)
        leg_contribution[recovery_mask] = -recovery_curve * 0.6  # Negative contribution
        
        # Back: Moderate positive peak in drive, moderate negative in recovery
        back_contribution = np.zeros_like(time_axis)
        
        # Drive phase: Moderate back contribution
        back_curve = np.exp(-((time_axis[drive_mask] - 0.3) / 0.12) ** 2)
        back_contribution[drive_mask] = back_curve * 0.7  # Moderate positive contribution
        
        # Recovery phase: Back return (negative)
        recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.7) / 0.1) ** 2)
        back_contribution[recovery_mask] = -recovery_curve * 0.4  # Negative contribution
        
        # Arms: Lower positive peak in drive, higher negative in recovery
        arm_contribution = np.zeros_like(time_axis)
        
        # Drive phase: Lower arm contribution
        arm_curve = np.exp(-((time_axis[drive_mask] - 0.4) / 0.1) ** 2)
        arm_contribution[drive_mask] = arm_curve * 0.5  # Lower positive contribution
        
        # Recovery phase: Arm extension (negative)
        recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.6) / 0.08) ** 2)
        arm_contribution[recovery_mask] = -recovery_curve * 0.7  # Higher negative contribution
        
        # Handle: Use actual force data mapped per selected strategy
        handle_contribution = np.zeros_like(time_axis)
        
        if stroke_force_data and len(stroke_force_data) > 0 and stroke_start_time and stroke_end_time:
            # Map force data to video frames using actual timestamps
            max_force = max(stroke_force_data)
            if max_force > 0:
                if 'stroke_duration' in locals():
                    stroke_duration_local = stroke_duration
                else:
                    stroke_duration_local = (stroke_end_time - stroke_start_time).total_seconds()

                mapping_mode = getattr(self, 'force_mapping', 'overlay')

                if mapping_mode == 'timestamps':
                    # Get the actual timestamps for the force data from the stroke measurements
                    stroke_info = combined_strokes[stroke_number - 1]
                    measurements = stroke_info['measurements']

                    # Create timestamps for each force point by distributing within measurement intervals
                    force_relative_times = []
                    force_values = []

                    for i, measurement in enumerate(measurements):
                        if measurement['forceplot']:
                            # Get timestamp for this measurement
                            measurement_time = measurement['timestamp_dt']
                            relative_measurement_time = (measurement_time - stroke_start_time).total_seconds() / stroke_duration_local

                            # Distribute forceplot points evenly within this measurement interval
                            forceplot_data = measurement['forceplot']
                            num_points = len(forceplot_data)

                            if num_points > 0:
                                if num_points == 1:
                                    # Single point at measurement time
                                    force_relative_times.append(relative_measurement_time)
                                    force_values.append(forceplot_data[0])
                                else:
                                    # Multiple points - distribute evenly around measurement time (short burst)
                                    interval_duration = 0.05  # 50ms interval for forceplot burst
                                    start_time = relative_measurement_time - interval_duration / 2
                                    end_time = relative_measurement_time + interval_duration / 2

                                    for j in range(num_points):
                                        point_time = start_time + (j / (num_points - 1)) * interval_duration
                                        force_relative_times.append(point_time)
                                        force_values.append(forceplot_data[j])

                    # Normalize and interpolate across the entire time axis
                    if force_values:
                        normalized_force = np.array(force_values) / max_force
                        handle_contribution = np.interp(time_axis, force_relative_times, normalized_force,
                                                       left=0.0, right=0.0)
                    else:
                        handle_contribution = np.zeros_like(time_axis)
                elif mapping_mode == 'overlay':
                    # Reproduce overlay exactly: sample combined_forceplot against drive portion with nearest index
                    stroke_info = combined_strokes[stroke_number - 1]
                    force_series = np.array(stroke_info['combined_forceplot'], dtype=float)
                    if force_series.size > 0:
                        normalized_force = force_series / np.max(force_series)
                        # Map each drive time sample to nearest force index
                        drive_indices = np.round((time_axis[drive_mask] / 0.5) * (len(normalized_force) - 1)).astype(int)
                        drive_indices = np.clip(drive_indices, 0, len(normalized_force) - 1)
                        handle_contribution[drive_mask] = normalized_force[drive_indices]
                        handle_contribution[recovery_mask] = 0.0
                    else:
                        handle_contribution = np.zeros_like(time_axis)
                else:
                    # Evenly distribute the force points across the drive phase only (matches PM5 overlay)
                    drive_force_points = len(stroke_force_data)
                    drive_time_axis = np.linspace(0, 0.5, drive_force_points)
                    normalized_force = np.array(stroke_force_data) / max_force
                    drive_force = np.interp(time_axis[drive_mask], drive_time_axis, normalized_force)
                    handle_contribution[drive_mask] = drive_force
                    handle_contribution[recovery_mask] = 0.0
            else:
                handle_contribution = np.zeros_like(time_axis)
        else:
            # Fallback to theoretical handle curve
            handle_curve = np.exp(-((time_axis[drive_mask] - 0.3) / 0.15) ** 2)
            handle_contribution[drive_mask] = handle_curve * 0.8  # High positive contribution
            
            # Recovery phase: Handle return (negative)
            recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.65) / 0.1) ** 2)
            handle_contribution[recovery_mask] = -recovery_curve * 0.5  # Negative contribution
        
        return {
            'time': time_axis,
            'legs': leg_contribution,
            'back': back_contribution,
            'arms': arm_contribution,
            'handle': handle_contribution
        }
    
    def calculate_contribution(self, angles, body_part):
        """Calculate contribution curves from actual angle velocity data"""
        # Remove NaN values and ensure we have valid data
        valid_mask = ~np.isnan(angles)
        if not np.any(valid_mask):
            return np.zeros_like(angles)
        
        # Fill NaN values with interpolation
        angles_clean = angles.copy()
        if np.any(~valid_mask):
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 1:
                angles_clean = np.interp(np.arange(len(angles)), valid_indices, angles[valid_indices])
            else:
                angles_clean = np.zeros_like(angles)
        
        # Calculate the velocity (rate of change) of the angles
        velocity = np.gradient(angles_clean)
        
        # Apply different scaling based on body part to make curves more dramatic
        if body_part == 'legs':
            # Legs should have the most dramatic curves
            scale_factor = 0.8
        elif body_part == 'back':
            # Back should have moderate curves
            scale_factor = 0.6
        elif body_part == 'arms':
            # Arms should have moderate curves
            scale_factor = 0.5
        else:
            scale_factor = 0.7
        
        # Normalize the velocity to a reasonable range
        # Use the maximum absolute velocity for normalization
        max_velocity = np.max(np.abs(velocity))
        if max_velocity > 0:
            contribution = (velocity / max_velocity) * scale_factor
        else:
            contribution = np.zeros_like(velocity)
        
        # Apply minimal smoothing to preserve the dramatic curves
        from scipy import ndimage
        contribution = ndimage.gaussian_filter1d(contribution, sigma=0.2)
        
        # Ensure values are between -1 and 1
        contribution = np.clip(contribution, -1, 1)
        
        return contribution
    
    def compute_coaching_metrics_for_stroke(self, stroke_number, data):
        """Compute compact coaching metrics for the metrics table.
        Requires pose frames loaded in data['pose_frames'] and combined strokes.
        """
        pose_frames = data.get('pose_frames')
        frame_ts = data.get('frame_timestamps')
        combined_strokes = data.get('combined_strokes')
        if not pose_frames or not frame_ts or not combined_strokes or stroke_number > len(combined_strokes):
            return None

        stroke = combined_strokes[stroke_number - 1]

        # Map pose frames to actual timestamps using frame_timestamps CSV (1:1 ordered mapping)
        mapped_frames = []
        limit = min(len(pose_frames), len(frame_ts))
        for i in range(limit):
            mapped_frames.append({'ts': frame_ts[i]['timestamp_dt'], 'frame': pose_frames[i]})

        # Helpers
        def find_nearest_frame(target_dt):
            best = None
            best_diff = float('inf')
            for item in mapped_frames:
                diff = abs((item['ts'] - target_dt).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best = item['frame']
            return best

        def mean_or_none(values):
            vals = [v for v in values if v is not None]
            return float(np.mean(vals)) if vals else None

        def handle_height_percent(pose_frame):
            keys = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_wrist', 'right_wrist']
            d = {}
            for k in keys:
                y_key = f"{k}_y"; c_key = f"{k}_confidence"
                if y_key in pose_frame and c_key in pose_frame and pose_frame[c_key] and pose_frame[c_key] > 0.5:
                    d[k] = pose_frame[y_key]
            if not all(x in d for x in ['left_shoulder','right_shoulder','left_hip','right_hip']) or not any('wrist' in k for k in d.keys()):
                return None
            shoulder_y = np.mean([d['left_shoulder'], d['right_shoulder']])
            hip_y = np.mean([d['left_hip'], d['right_hip']])
            wrist_vals = [v for k,v in d.items() if 'wrist' in k]
            wrist_y = np.mean(wrist_vals)
            denom = (hip_y - shoulder_y)
            if denom <= 1e-3:
                return None
            pct = (hip_y - wrist_y) / denom * 100.0
            return float(np.clip(pct, 0.0, 100.0))

        # Determine catch and finish using kinematics across ALL frames
        # Strategy: Find local extrema in hip position around the stroke time
        
        # First, determine global drive direction from all frames
        all_hip_x = []
        for item in mapped_frames:
            fr = item['frame']
            left_hip_x = fr.get('left_hip_x')
            right_hip_x = fr.get('right_hip_x')
            left_conf = fr.get('left_hip_confidence', 0)
            right_conf = fr.get('right_hip_confidence', 0)
            if left_hip_x is not None and right_hip_x is not None and left_conf > 0.5 and right_conf > 0.5:
                all_hip_x.append((left_hip_x + right_hip_x) / 2.0)
        
        if len(all_hip_x) < 10:
            # Fallback to PM5 timing
            first_drive = None
            last_drive = None
            for m in stroke.get('measurements', []):
                if m.get('strokestate') == 'Drive' and first_drive is None:
                    first_drive = m.get('timestamp_dt')
                if m.get('strokestate') in ['Drive', 'Dwelling']:
                    last_drive = m.get('timestamp_dt')
            catch_dt = first_drive or stroke['start_timestamp_dt']
            finish_dt = last_drive or stroke['end_timestamp_dt']
            catch_pose = find_nearest_frame(catch_dt) or {}
            finish_pose = find_nearest_frame(finish_dt) or {}
        else:
            # Determine drive direction from overall movement pattern
            all_hip_x_array = np.array(all_hip_x)
            global_range = np.max(all_hip_x_array) - np.min(all_hip_x_array)
            
            # Get frames in an expanded window around this stroke (same as frame extraction)
            stroke_start_expanded = stroke['start_timestamp_dt'] - pd.Timedelta(seconds=0.7)
            stroke_end_expanded = stroke['end_timestamp_dt'] + pd.Timedelta(seconds=0.3)
            
            window_frames = []
            window_hip_x = []
            for item in mapped_frames:
                if stroke_start_expanded <= item['ts'] <= stroke_end_expanded:
                    fr = item['frame']
                    left_hip_x = fr.get('left_hip_x')
                    right_hip_x = fr.get('right_hip_x')
                    left_conf = fr.get('left_hip_confidence', 0)
                    right_conf = fr.get('right_hip_confidence', 0)
                    if left_hip_x is not None and right_hip_x is not None and left_conf > 0.5 and right_conf > 0.5:
                        window_frames.append(item)
                        window_hip_x.append((left_hip_x + right_hip_x) / 2.0)
            
            if len(window_frames) < 5:
                # Fallback
                catch_pose = window_frames[0]['frame'] if window_frames else {}
                finish_pose = window_frames[len(window_frames)//2]['frame'] if len(window_frames) > 1 else {}
            else:
                window_hip_x_array = np.array(window_hip_x)
                
                # Determine direction from the window itself
                drive_direction = 'left' if window_hip_x[0] > window_hip_x[-1] else 'right'
                
                # Find local extrema using scipy
                from scipy.signal import argrelextrema
                
                # Smooth less aggressively to preserve actual peaks
                from scipy.ndimage import gaussian_filter1d
                smoothed = gaussian_filter1d(window_hip_x_array, sigma=0.8)
                
                # Simply find the absolute extreme in the appropriate half
                # Catch should be in the earlier part of the window
                midpoint = len(smoothed) // 2
                
                if drive_direction == 'left':
                    # Catch is rightmost (maximum X) - look in first 60% of window
                    search_end = int(len(smoothed) * 0.6)
                    catch_idx = int(np.argmax(smoothed[:search_end]))
                    
                    # Finish is leftmost (minimum X) - look after catch
                    finish_idx = catch_idx + int(np.argmin(smoothed[catch_idx:]))
                else:
                    # Catch is leftmost (minimum X) - look in first 60% of window  
                    search_end = int(len(smoothed) * 0.6)
                    catch_idx = int(np.argmin(smoothed[:search_end]))
                    
                    # Finish is rightmost (maximum X) - look after catch
                    finish_idx = catch_idx + int(np.argmax(smoothed[catch_idx:]))
                
                catch_pose = window_frames[catch_idx]['frame']
                finish_pose = window_frames[finish_idx]['frame']
                catch_dt = window_frames[catch_idx]['ts']
                finish_dt = window_frames[finish_idx]['ts']

        layback = finish_pose.get('back_vertical_angle')
        legs_unbent = mean_or_none([finish_pose.get('left_leg_angle'), finish_pose.get('right_leg_angle')])
        handle_pct = handle_height_percent(finish_pose)

        shins_angle = mean_or_none([catch_pose.get('left_ankle_vertical_angle'), catch_pose.get('right_ankle_vertical_angle')])
        fwd_body = catch_pose.get('back_vertical_angle')
        elbows_unbent = mean_or_none([catch_pose.get('left_arm_angle'), catch_pose.get('right_arm_angle')])

        # Sequence metrics based on angle gradients across drive
        # Collect frames within drive (using kinematic-based catch/finish if available)
        if 'catch_dt' in locals() and 'finish_dt' in locals():
            drive_start = catch_dt
            drive_end = finish_dt
        else:
            # Fallback to PM5 timing
            first_drive = None
            last_drive = None
            for m in stroke.get('measurements', []):
                if m.get('strokestate') == 'Drive' and first_drive is None:
                    first_drive = m.get('timestamp_dt')
                if m.get('strokestate') in ['Drive', 'Dwelling']:
                    last_drive = m.get('timestamp_dt')
            drive_start = first_drive or stroke['start_timestamp_dt']
            drive_end = last_drive or stroke.get('end_timestamp_dt', drive_start)
        ts_list = []
        legs_series = []
        back_series = []
        arms_series = []
        for item in mapped_frames:
            ts = item['ts']
            fr = item['frame']
            if drive_start <= ts <= drive_end:
                ts_list.append(ts)
                legs_series.append(mean_or_none([fr.get('left_leg_angle'), fr.get('right_leg_angle')]))
                back_series.append(fr.get('back_vertical_angle'))
                arms_series.append(mean_or_none([fr.get('left_arm_angle'), fr.get('right_arm_angle')]))

        def grad_peak_time(times, vals, prefer_positive=True):
            if len(times) < 3:
                return None
            arr = np.array(vals, dtype=float)
            # Interpolate NaNs
            mask = ~np.isnan(arr)
            if np.any(mask):
                arr = np.interp(np.arange(len(arr)), np.where(mask)[0], arr[mask])
            seconds = np.array([t.timestamp() for t in times])
            if len(seconds) < 2:
                return None
            dt = np.gradient(seconds)
            grad = np.gradient(arr) / np.maximum(dt, 1e-6)
            idx = int(np.argmax(grad) if prefer_positive else np.argmin(grad))
            return times[idx]

        legs_peak_t = grad_peak_time(ts_list, legs_series, prefer_positive=True)
        back_peak_t = grad_peak_time(ts_list, back_series, prefer_positive=True)
        arms_peak_t = grad_peak_time(ts_list, arms_series, prefer_positive=False)

        drive_dur = (drive_end - drive_start).total_seconds() if (drive_end and drive_start) else None
        sep_lb = float(np.clip(((back_peak_t - legs_peak_t).total_seconds() / drive_dur) * 100.0, 0.0, 100.0)) if (drive_dur and legs_peak_t and back_peak_t) else None
        sep_ba = float(np.clip(((arms_peak_t - back_peak_t).total_seconds() / drive_dur) * 100.0, 0.0, 100.0)) if (drive_dur and back_peak_t and arms_peak_t) else None
        stroke_dur = stroke.get('stroke_duration')
        drive_ratio = float(np.clip(drive_dur / stroke_dur * 100.0, 0.0, 100.0)) if (stroke_dur and drive_dur) else None

        def fmt(v):
            return f"{v:.0f}" if isinstance(v, (int,float)) and v is not None else "N/A"

        return {
            'finish_layback': fmt(layback),
            'finish_legs': fmt(legs_unbent),
            'finish_handle': fmt(handle_pct),
            'catch_shins': fmt(shins_angle),
            'catch_body': fmt(fwd_body),
            'catch_elbows': fmt(elbows_unbent),
            'sep_lb': fmt(sep_lb),
            'sep_ba': fmt(sep_ba),
            'drive_ratio': fmt(drive_ratio)
        }
    def create_comprehensive_stroke_analysis(self, frames, sequence_data, stroke_number, output_path):
        """Create comprehensive analysis with video frames and sequence plot"""
        if len(frames) == 0:
            return
        
        # Create figure with subplots (2x2 grid for 4 frames)
        fig = plt.figure(figsize=(20, 18))
        
        # Create grid layout: 2 rows of frames, then sequence plot, then metrics
        # Rows: [Frame top-left, Frame top-right], [Frame bottom-left, Frame bottom-right], [Sequence], [Metrics]
        gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.0, 2.0], width_ratios=[1, 1])
        
        # Add main title
        fig.suptitle(f'Stroke #{stroke_number} - Comprehensive Analysis', fontsize=20, fontweight='bold')
        
        # Phase labels (classical 4 phases)
        phase_labels = ["Catch", "Drive", "Finish", "Recovery"]
        
        # Frame positions in 2x2 grid: [(row, col), ...]
        frame_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # Create video frame subplots (2x2 grid)
        frame_axes = []
        for i in range(4):
            if i < len(frames):
                row, col = frame_positions[i]
                ax = fig.add_subplot(gs[row, col])
                frame_axes.append(ax)
                
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frames[i]['frame'], cv2.COLOR_BGR2RGB)
                
                ax.imshow(frame_rgb)
                
                # Add title with phase label
                phase_label = phase_labels[i] if i < len(phase_labels) else f"Phase {i+1}"
                ax.set_title(f'{phase_label}\n(Frame #{frames[i]["frame_number"]})', fontsize=14, fontweight='bold')
                ax.axis('off')
                
                # Add angle measurements as text overlay
                frame_data = frames[i]['data']
                angle_text = ""
                angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle']
                for angle in angles:
                    if angle in frame_data and not pd.isna(frame_data[angle]):
                        angle_name = angle.replace('_', ' ').title()
                        angle_text += f"{angle_name}: {frame_data[angle]:.1f}°\n"
                
                if angle_text:
                    ax.text(0.02, 0.98, angle_text.strip(), transform=ax.transAxes, 
                           fontsize=9, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # Empty subplot for missing frames
                row, col = frame_positions[i]
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
                frame_axes.append(ax)
        
        # Create sequence plot (row 3, spans both columns)
        ax_sequence = fig.add_subplot(gs[2, :])
        self.create_sequence_plot(ax_sequence, sequence_data, stroke_number)

        # Create compact metrics UI (bottom row) - THREE COLUMNS like example image
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('off')
        metrics = sequence_data.get('coaching_metrics')
        if metrics:
            # Helper to draw a single metric row with label, gauge, and value
            def draw_metric_row(ax, x, y, w, h, label, value, ideal_low, ideal_high, scale_min, scale_max, is_min_threshold=False):
                """Draw one metric row: everything INSIDE the gauge bar"""
                
                # Determine if value is in range for color coding
                try:
                    v = float(str(value).replace('N/A','nan').replace('°','').replace('%',''))
                except:
                    v = float('nan')
                
                in_range = False
                if not np.isnan(v):
                    if is_min_threshold:
                        in_range = v >= ideal_low
                    else:
                        in_range = ideal_low <= v <= ideal_high
                
                # Draw full-width gauge bar with color coding
                bg_color = "#d3d3d3" if in_range else "#FFD580"
                ax.add_patch(patches.Rectangle((x, y), w, h, 
                            facecolor=bg_color, edgecolor="#888888", linewidth=0.5))
                
                # Green ideal range band
                if is_min_threshold:
                    ok_x1 = x + ((ideal_low - scale_min) / (scale_max - scale_min)) * w
                    ok_w = (x + w) - ok_x1
                    ax.add_patch(patches.Rectangle((ok_x1, y), ok_w, h,
                                facecolor="#90EE90", edgecolor="none", alpha=0.7))
                    # Show threshold value inside green band
                    ax.text(ok_x1 + 0.005, y + h*0.5, f"{ideal_low}", ha='left', va='center', 
                           fontsize=9, color="#006400", fontweight='bold', zorder=5)
                else:
                    ok_x1 = x + ((ideal_low - scale_min) / (scale_max - scale_min)) * w
                    ok_x2 = x + ((ideal_high - scale_min) / (scale_max - scale_min)) * w
                    ok_w = ok_x2 - ok_x1
                    ax.add_patch(patches.Rectangle((ok_x1, y), ok_w, h,
                                facecolor="#90EE90", edgecolor="none", alpha=0.7))
                    # Show both endpoints inside green band
                    ax.text(ok_x1 + 0.005, y + h*0.5, f"{ideal_low}", ha='left', va='center', 
                           fontsize=9, color="#006400", fontweight='bold', zorder=5)
                    ax.text(ok_x2 - 0.005, y + h*0.5, f"{ideal_high}", ha='right', va='center', 
                           fontsize=9, color="#006400", fontweight='bold', zorder=5)
                
                # Draw tick mark for measured value
                if not np.isnan(v):
                    vx = x + ((v - scale_min) / (scale_max - scale_min)) * w
                    vx = max(x, min(x + w, vx))
                    ax.add_line(plt.Line2D([vx, vx], [y, y+h], color="#000000", linewidth=4, zorder=10))
                
                # Draw label INSIDE on the left
                ax.text(x + 0.005, y + h*0.5, label, ha='left', va='center', fontsize=10, zorder=11)
                
                # Draw measured value INSIDE on the right
                ax.text(x + w - 0.005, y + h*0.5, value, ha='right', va='center', fontsize=11, fontweight='bold', zorder=11)
            
            # Layout: Three columns (Catch | Finish | Sequence)
            col_w = 0.31
            col_gap = 0.03
            x_catch = 0.01
            x_finish = x_catch + col_w + col_gap
            x_sequence = x_finish + col_w + col_gap
            
            base_y = 0.68
            row_h = 0.16  # Tall rows
            row_gap = 0.08  # Good spacing between rows
            header_offset = 0.18
            
            # CATCH COLUMN
            ax_table.text(x_catch, base_y + header_offset, "Catch", fontsize=14, fontweight='bold', va='bottom')
            
            draw_metric_row(ax_table, x_catch, base_y - row_gap*0, col_w, row_h,
                          "Shins angle", f"{metrics['catch_shins']}°",
                          -8, 6, -40, 20, is_min_threshold=False)
            
            draw_metric_row(ax_table, x_catch, base_y - row_gap*1 - row_h*1, col_w, row_h,
                          "Forward body angle", f"{metrics['catch_body']}°",
                          13, 39, 0, 60, is_min_threshold=False)
            
            draw_metric_row(ax_table, x_catch, base_y - row_gap*2 - row_h*2, col_w, row_h,
                          "Elbows unbent", f"{metrics['catch_elbows']}°",
                          160, 180, 120, 200, is_min_threshold=True)
            
            # FINISH COLUMN
            ax_table.text(x_finish, base_y + header_offset, "Finish", fontsize=14, fontweight='bold', va='bottom')
            
            draw_metric_row(ax_table, x_finish, base_y - row_gap*0, col_w, row_h,
                          "Layback body angle", f"{metrics['finish_layback']}°",
                          -48, -32, -60, 0, is_min_threshold=False)
            
            draw_metric_row(ax_table, x_finish, base_y - row_gap*1 - row_h*1, col_w, row_h,
                          "Legs unbent", f"{metrics['finish_legs']}°",
                          164, 180, 120, 200, is_min_threshold=True)
            
            draw_metric_row(ax_table, x_finish, base_y - row_gap*2 - row_h*2, col_w, row_h,
                          "Handle height at torso", f"{metrics['finish_handle']}%",
                          40, 80, 0, 100, is_min_threshold=False)
            
            # SEQUENCE COLUMN
            ax_table.text(x_sequence, base_y + header_offset, "Sequence", fontsize=14, fontweight='bold', va='bottom')
            
            draw_metric_row(ax_table, x_sequence, base_y - row_gap*0, col_w, row_h,
                          "Drive Legs&Back separation", f"{metrics['sep_lb']}%",
                          75, 100, 0, 100, is_min_threshold=True)
            
            draw_metric_row(ax_table, x_sequence, base_y - row_gap*1 - row_h*1, col_w, row_h,
                          "Drive Back&Arms separation", f"{metrics['sep_ba']}%",
                          30, 100, 0, 100, is_min_threshold=True)
            
            draw_metric_row(ax_table, x_sequence, base_y - row_gap*2 - row_h*2, col_w, row_h,
                          "Drive duration ratio", f"{metrics['drive_ratio']}%",
                          30, 50, 0, 100, is_min_threshold=False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Created comprehensive analysis: {os.path.basename(output_path)}")
    
    def create_sequence_plot(self, ax, sequence_data, stroke_number):
        """Create the speed & sequence plot"""
        # Set up the plot to match the example
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.2, 1.4)  # Allow for negative values like the example
        
        # Draw vertical line to separate Drive and Recovery phases
        ax.axvline(x=0.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
        
        # Plot the sequence data
        time = sequence_data['time']
        
        # Legs (green line) - thick line with thin overlay
        ax.plot(time, sequence_data['legs'], color='green', linewidth=4, label='Legs')
        ax.plot(time, sequence_data['legs'], color='green', linewidth=2, alpha=0.3)
        
        # Back (blue line) - thick line with thin overlay
        ax.plot(time, sequence_data['back'], color='blue', linewidth=4, label='Back')
        ax.plot(time, sequence_data['back'], color='blue', linewidth=2, alpha=0.3)
        
        # Arms (magenta/purple line) - thick line with thin overlay
        ax.plot(time, sequence_data['arms'], color='magenta', linewidth=4, label='Arms')
        ax.plot(time, sequence_data['arms'], color='magenta', linewidth=2, alpha=0.3)
        
        # Handle (dotted black line) - shows actual force data
        ax.plot(time, sequence_data['handle'], color='black', linestyle='--', linewidth=3, label='Force (PM5)')
        
        # Add prominent zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add labels at peaks
        self.add_peak_labels(ax, time, sequence_data)
        
        # Add phase annotations
        ax.text(0.25, 1.2, 'Drive: Legs → Back → Arms', ha='center', va='center', 
                fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.75, 1.2, 'Recovery: Arms → Back → Legs', ha='center', va='center', 
                fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Add separation percentages (placeholder values)
        ax.text(0.25, -0.8, 'Separation: Legs ← Back ← Arms\n97% 53%', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.75, -0.8, 'Separation: Arms ← Back ← Legs\n63% 100%', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize the plot
        ax.set_title('Speed & Sequence', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stroke Cycle', fontsize=12)
        ax.set_ylabel('Relative Contribution', fontsize=12)
        
        # Remove y-axis ticks and labels for cleaner look
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Set x-axis labels
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['Catch', 'Finish', 'Catch'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
    
    def add_peak_labels(self, ax, time, sequence_data):
        """Add labels at peaks of each body part contribution"""
        # Find peaks for each body part
        for body_part, color, label in [('legs', 'green', 'L'), ('back', 'blue', 'B'), ('arms', 'magenta', 'A')]:
            data = sequence_data[body_part]
            # Find local maxima
            from scipy.signal import find_peaks
            distance = max(1, len(data)//10)  # Ensure distance is at least 1
            peaks, _ = find_peaks(data, height=0.3, distance=distance)
            
            for peak in peaks:
                if peak < len(time) and peak < len(data):
                    ax.annotate(label, xy=(time[peak], data[peak]), 
                              xytext=(time[peak], data[peak] + 0.1),
                              ha='center', va='bottom', fontsize=12, fontweight='bold',
                              color=color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def generate_stroke_comprehensive_analysis(self, data, stroke_number, output_dir):
        """Generate comprehensive analysis for a single stroke"""
        print(f"📸 Creating comprehensive analysis for Stroke #{stroke_number}")
        
        # Extract key frames for this stroke
        frames = self.extract_key_frames_for_stroke(data, stroke_number, num_frames=6)
        if not frames:
            print(f"❌ No frames extracted for stroke {stroke_number}")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate sequence data
        stroke_data = data['dataframe'][data['dataframe']['stroke_number'] == stroke_number]
        if len(stroke_data) > 0:
            sequence_data = self.calculate_stroke_sequence_data({'data': stroke_data}, stroke_number, data.get('combined_strokes'))

            # Attach coaching metrics from pose frames if available
            metrics = self.compute_coaching_metrics_for_stroke(stroke_number, data)
            if metrics:
                sequence_data['coaching_metrics'] = metrics
            
            # Create comprehensive analysis
            comprehensive_path = os.path.join(output_dir, f"stroke_{stroke_number:02d}_comprehensive_analysis.png")
            self.create_comprehensive_stroke_analysis(frames, sequence_data, stroke_number, comprehensive_path)
        
        return output_dir
    
    def generate_all_comprehensive_analyses(self, analysis_dir):
        """Generate comprehensive analyses for all strokes"""
        print("📸 Generating Comprehensive Stroke Analyses")
        print("=" * 50)
        
        # Load analysis data
        data = self.load_analysis_data(analysis_dir)
        if not data:
            return
        
        # Extract strokes
        df = data['dataframe']
        if 'stroke_number' not in df.columns:
            print("❌ No stroke_number column found")
            return
        
        strokes = df['stroke_number'].dropna().unique()
        print(f"📊 Found {len(strokes)} strokes to analyze")
        
        # Create output directory inside the analysis directory
        output_dir = os.path.join(analysis_dir, "comprehensive_analyses")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate analyses for each stroke
        generated_reports = []
        for stroke_num in strokes:
            stroke_dir = self.generate_stroke_comprehensive_analysis(data, int(stroke_num), output_dir)
            if stroke_dir:
                generated_reports.append(stroke_dir)
        
        print(f"\n🎉 Generated {len(generated_reports)} comprehensive analyses!")
        print(f"📁 Output directory: {output_dir}")
        
        return generated_reports

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate comprehensive stroke analyses")
    parser.add_argument("analysis_dir", help="Analysis directory path containing the data")
    parser.add_argument("--force-mapping", choices=["even", "timestamps", "overlay"], default="overlay",
                        help="How to map PM5 force: default 'overlay' reproduces video overlay; 'even' distributes; 'timestamps' aligns by time")
    
    args = parser.parse_args()
    
    # Check if analysis directory exists
    if not os.path.exists(args.analysis_dir):
        print(f"❌ Analysis directory not found: {args.analysis_dir}")
        print("Usage: python comprehensive_stroke_analysis.py <analysis_directory>")
        return
    
    # Initialize generator
    generator = ComprehensiveStrokeAnalysis(force_mapping=args.force_mapping)
    
    # Generate comprehensive analyses
    generator.generate_all_comprehensive_analyses(args.analysis_dir)

if __name__ == "__main__":
    main()
