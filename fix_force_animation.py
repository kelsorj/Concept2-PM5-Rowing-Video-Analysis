#!/usr/bin/env python3
"""
Fix Force Animation
Debug and fix the force curve animation issue
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

def find_closest_stroke_for_time_with_debug(target_time, combined_strokes, frame_num):
    """Find the closest stroke for a given timestamp and calculate animation position with debug info"""
    if not combined_strokes:
        return None, None, None, "No strokes available"
    
    target_time_str = target_time.strftime('%H:%M:%S.%f')[:-3]
    debug_info = f"Frame {frame_num}: Looking for time {target_time_str}\n"
    
    min_time_diff = float('inf')
    closest_stroke = None
    closest_idx = None
    stroke_num = None
    
    for i, stroke in enumerate(combined_strokes):
        start_time = stroke['start_timestamp_dt']
        end_time = stroke['end_timestamp_dt']
        start_str = start_time.strftime('%H:%M:%S.%f')[:-3]
        end_str = end_time.strftime('%H:%M:%S.%f')[:-3]
        
        debug_info += f"  Stroke {i+1}: {start_str} - {end_str}\n"
        
        # Check if target time is within stroke duration
        if start_time <= target_time <= end_time:
            # Time is within this stroke - calculate position within stroke
            stroke_duration = (end_time - start_time).total_seconds()
            time_within_stroke = (target_time - start_time).total_seconds()
            
            if stroke_duration > 0:
                progress = time_within_stroke / stroke_duration
                force_idx = int(progress * (len(stroke['combined_forceplot']) - 1))
                force_idx = max(0, min(force_idx, len(stroke['combined_forceplot']) - 1))
                
                debug_info += f"    -> WITHIN STROKE! Progress: {progress:.3f}, Force idx: {force_idx}\n"
                return stroke, force_idx, i + 1, debug_info
        
        # If not within stroke, find closest
        time_diff = min(
            abs((target_time - start_time).total_seconds()),
            abs((target_time - end_time).total_seconds())
        )
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_stroke = stroke
            closest_idx = 0  # Default to start of stroke
            stroke_num = i + 1
    
    debug_info += f"  -> Not within any stroke. Closest: Stroke {stroke_num}, time diff: {min_time_diff:.3f}s\n"
    return closest_stroke, closest_idx, stroke_num, debug_info

def create_fixed_force_animation_video(video_path, raw_csv_path, output_dir="fixed_force_animation"):
    """Create video with properly animated force curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    combined_strokes = load_and_combine_force_data(raw_csv_path)
    
    if not combined_strokes:
        print("❌ No force data available. Cannot create overlay video.")
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
    output_path = os.path.join(output_dir, f"fixed_force_animation_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames with FIXED force animation...")
    
    frame_count = 0
    force_overlays = 0
    debug_log = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate current frame's absolute timestamp
        current_frame_elapsed_s = frame_count / fps
        current_frame_abs_time = video_start_dt + timedelta(seconds=current_frame_elapsed_s)
        
        # Find closest stroke for current time with debug info
        closest_stroke, force_idx, stroke_num, debug_msg = find_closest_stroke_for_time_with_debug(
            current_frame_abs_time, combined_strokes, frame_count
        )
        
        # Log debug info for first few frames
        if frame_count <= 10:
            debug_log.append(debug_msg)
        
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
                    frame[py:py+ph, px:px+pw] = force_plot
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
    
    # Print debug info
    print(f"\n🔍 Debug info for first 10 frames:")
    for i, debug_msg in enumerate(debug_log):
        print(f"Frame {i+1}:")
        print(debug_msg)
    
    print(f"\n🎉 Fixed force animation video complete!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   🔋 Force plot overlays: {force_overlays}")
    print(f"   📊 Force overlay rate: {(force_overlays/frame_count)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Fix force curve animation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--raw-csv", required=True, help="Path to raw CSV file")
    parser.add_argument("--output-dir", default="fixed_force_animation", help="Output directory")
    
    args = parser.parse_args()
    
    print("🎬 Fixing Force Curve Animation")
    print("=" * 50)
    
    create_fixed_force_animation_video(args.video, args.raw_csv, args.output_dir)

if __name__ == "__main__":
    main()
