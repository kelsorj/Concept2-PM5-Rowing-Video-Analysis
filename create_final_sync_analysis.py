#!/usr/bin/env python3
"""
Create final synchronized analysis using actual session timestamps
"""

import cv2
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import glob

def load_raw_force_data(raw_csv_path):
    """Load and parse raw force data with proper timestamps"""
    print(f"📊 Loading raw force data: {raw_csv_path}")
    
    force_data = []
    with open(raw_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse the raw JSON
                raw_json = json.loads(row['raw_json'])
                
                # Extract forceplot data
                forceplot = raw_json.get('forceplot', [])
                if forceplot:  # Only include rows with actual forceplot data
                    force_data.append({
                        'timestamp_ns': int(row['ts_ns']),
                        'timestamp_iso': row['ts_iso'],
                        'timestamp_dt': datetime.fromisoformat(row['ts_iso']),
                        'elapsed_s': raw_json.get('time', 0.0),
                        'distance_m': raw_json.get('distance', 0.0),
                        'spm': raw_json.get('spm', 0),
                        'power': raw_json.get('power', 0),
                        'pace': raw_json.get('pace', 0.0),
                        'forceplot': forceplot,
                        'strokestate': raw_json.get('strokestate', ''),
                        'status': raw_json.get('status', '')
                    })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                continue
    
    print(f"   Found {len(force_data)} rows with forceplot data")
    return force_data

def load_pose_data():
    """Load the most recent pose data"""
    pose_files = glob.glob("rowing_pose_data_*.json")
    if not pose_files:
        print("❌ No pose data files found")
        return None
    
    latest_pose_file = max(pose_files, key=os.path.getctime)
    print(f"📊 Loading pose data: {latest_pose_file}")
    
    with open(latest_pose_file, 'r') as f:
        pose_data = json.load(f)
    
    print(f"   Found {len(pose_data)} pose frames")
    return pose_data

def create_force_curve_plot(force_curve, power, spm, distance, elapsed_s, current_idx=None):
    """Create a matplotlib plot of the force curve with animated position"""
    if not force_curve:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot force curve
    x = np.arange(len(force_curve))
    ax.plot(x, force_curve, 'lime', linewidth=3, label='Force')
    
    # Highlight current position or peak
    if current_idx is not None and current_idx < len(force_curve):
        current_force = force_curve[current_idx]
        ax.plot(current_idx, current_force, 'ro', markersize=8)
        ax.annotate(f'{current_force}', (current_idx, current_force), 
                    xytext=(5, 5), textcoords='offset points',
                    color='white', fontsize=10, fontweight='bold')
    else:
        # Highlight peak
        peak_idx = np.argmax(force_curve)
        peak_force = force_curve[peak_idx]
        ax.plot(peak_idx, peak_force, 'ro', markersize=8)
        ax.annotate(f'{peak_force}', (peak_idx, peak_force), 
                    xytext=(5, 5), textcoords='offset points',
                    color='white', fontsize=10, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Stroke Position', color='white', fontsize=10)
    ax.set_ylabel('Force', color='white', fontsize=10)
    ax.set_title(f'Force Curve - {power}W, {spm}spm', color='white', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white', labelsize=8)
    
    # Add stats text
    peak_force = max(force_curve) if force_curve else 0
    avg_force = np.mean(force_curve) if force_curve else 0
    stats_text = f'Peak: {peak_force}\nAvg: {avg_force:.1f}\nDist: {distance:.1f}m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', color='white', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def create_angle_display(pose_frame, frame_num, elapsed_s, force_data=None):
    """Create a comprehensive display of body angles and force data"""
    # Create a semi-transparent background
    img = np.zeros((300, 350, 3), dtype=np.uint8)
    
    # Define colors
    colors = {
        'white': (255, 255, 255),
        'lime': (0, 255, 0),
        'cyan': (255, 255, 0),
        'yellow': (0, 255, 255),
        'red': (0, 0, 255),
        'orange': (0, 165, 255)
    }
    
    # Title
    cv2.putText(img, "ROWING ANALYSIS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['white'], 2)
    
    # Frame info
    cv2.putText(img, f"Frame: {frame_num}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
    cv2.putText(img, f"Time: {elapsed_s:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
    
    y_pos = 95
    line_height = 20
    
    # Body angles
    if pose_frame:
        cv2.putText(img, "BODY ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        if pose_frame.get('left_arm_angle') is not None:
            left_arm = pose_frame['left_arm_angle']
            cv2.putText(img, f"  L Arm: {left_arm:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
            y_pos += line_height
        
        if pose_frame.get('right_arm_angle') is not None:
            right_arm = pose_frame['right_arm_angle']
            cv2.putText(img, f"  R Arm: {right_arm:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
            y_pos += line_height
        
        if pose_frame.get('left_leg_angle') is not None:
            left_leg = pose_frame['left_leg_angle']
            cv2.putText(img, f"  L Leg: {left_leg:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
            y_pos += line_height
        
        if pose_frame.get('right_leg_angle') is not None:
            right_leg = pose_frame['right_leg_angle']
            cv2.putText(img, f"  R Leg: {right_leg:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
            y_pos += line_height
    
    # Force data info
    if force_data:
        cv2.putText(img, "FORCE DATA:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        cv2.putText(img, f"  Power: {force_data['power']}W", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
        y_pos += line_height
        
        cv2.putText(img, f"  SPM: {force_data['spm']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
        y_pos += line_height
        
        cv2.putText(img, f"  State: {force_data['strokestate']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['red'], 1)
    
    return img

def create_final_sync_overlay(video_path, raw_csv_path, output_dir):
    """Create final synchronized overlay video"""
    print("🎬 Creating Final Synchronized Overlay Video")
    print("=" * 60)
    
    # Load data
    force_data = load_raw_force_data(raw_csv_path)
    pose_data = load_pose_data()
    
    if not force_data:
        print("❌ No force data found")
        return
    
    if not pose_data:
        print("❌ No pose data found")
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
    
    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"rowing_final_sync_overlay_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames with final synchronization...")
    
    frame_count = 0
    overlays_added = 0
    force_overlays = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate elapsed time from frame number
        elapsed_s = (frame_count - 1) / fps
        
        # Get pose data for this frame
        pose_frame = None
        if frame_count <= len(pose_data):
            pose_frame = pose_data[frame_count - 1]
        
        # Find closest force data using elapsed time
        closest_force = None
        min_time_diff = float('inf')
        
        for force_entry in force_data:
            time_diff = abs(elapsed_s - force_entry['elapsed_s'])
            
            if time_diff < 2.0 and time_diff < min_time_diff:  # Within 2 seconds
                min_time_diff = time_diff
                closest_force = force_entry
        
        # Create angle display
        angle_display = create_angle_display(pose_frame, frame_count, elapsed_s, closest_force)
        
        # Create force curve plot if available
        force_plot = None
        if closest_force and closest_force.get('forceplot'):
            force_curve = closest_force['forceplot']
            if force_curve:
                # Calculate animated position in force curve
                current_idx = None
                if len(force_curve) > 0:
                    # Simple animation: cycle through force curve points
                    animation_cycle = (frame_count % (len(force_curve) * 3)) // 3
                    current_idx = min(animation_cycle, len(force_curve) - 1)
                
                force_plot = create_force_curve_plot(
                    force_curve,
                    closest_force.get('power', 0),
                    closest_force.get('spm', 0),
                    closest_force.get('distance_m', 0),
                    elapsed_s,
                    current_idx
                )
        
        # Overlay angle display (top-left)
        if angle_display is not None:
            angle_height, angle_width = angle_display.shape[:2]
            angle_x_offset = 10
            angle_y_offset = 10
            
            if (angle_x_offset + angle_width <= width and 
                angle_y_offset + angle_height <= height):
                frame[angle_y_offset:angle_y_offset+angle_height, 
                      angle_x_offset:angle_x_offset+angle_width] = angle_display
                overlays_added += 1
        
        # Overlay force plot (top-right)
        if force_plot is not None:
            plot_height, plot_width = force_plot.shape[:2]
            plot_x_offset = width - plot_width - 10
            plot_y_offset = 10
            
            if (plot_x_offset >= 0 and 
                plot_y_offset + plot_height <= height):
                frame[plot_y_offset:plot_y_offset+plot_height, 
                      plot_x_offset:plot_x_offset+plot_width] = force_plot
                force_overlays += 1
        
        # Add frame info at bottom
        force_info = ""
        sync_info = ""
        if closest_force:
            force_info = f" | Power: {closest_force.get('power', 0)}W | SPM: {closest_force.get('spm', 0)}"
            sync_info = f" | Sync: {min_time_diff:.3f}s"
        else:
            sync_info = " | No force data"
        
        info_text = f"Frame {frame_count}/{total_frames} | Time: {elapsed_s:.1f}s{force_info}{sync_info}"
        cv2.putText(frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n🎉 Final overlay video complete!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   📈 Overlays added: {overlays_added}")
    print(f"   📊 Overlay rate: {(overlays_added/frame_count)*100:.1f}%")
    print(f"   🔋 Force plot overlays: {force_overlays}")
    print(f"   📊 Force overlay rate: {(force_overlays/frame_count)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Create final synchronized overlay video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--raw-csv", required=True, help="Path to raw CSV file")
    parser.add_argument("--output-dir", default="final_sync_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.raw_csv):
        print(f"❌ Raw CSV file not found: {args.raw_csv}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_final_sync_overlay(args.video, args.raw_csv, args.output_dir)

if __name__ == "__main__":
    main()
