#!/usr/bin/env python3
"""
Analyze session using raw data with proper timestamp synchronization
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
import subprocess

def extract_video_creation_time(video_path):
    """Extract creation time from video metadata"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        if 'format' in metadata and 'tags' in metadata['format']:
            creation_time_str = metadata['format']['tags'].get('creation_time')
            if creation_time_str:
                return datetime.fromisoformat(creation_time_str.replace('Z', '+00:00'))
        
        return None
    except Exception as e:
        print(f"⚠️  Error extracting video metadata: {e}")
        return None

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

def load_frame_timestamps(frames_csv_path):
    """Load frame timestamps"""
    print(f"📊 Loading frame timestamps: {frames_csv_path}")
    
    frame_timestamps = []
    with open(frames_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_timestamps.append({
                    'frame_idx': int(row['frame_idx']),
                    'timestamp_ns': int(row['ts_ns']),
                    'timestamp_iso': row['ts_iso'],
                    'timestamp_dt': datetime.fromisoformat(row['ts_iso'])
                })
            except (ValueError, KeyError) as e:
                continue
    
    print(f"   Loaded {len(frame_timestamps)} frame timestamps")
    return frame_timestamps

def find_closest_force_data(frame_timestamp_dt, force_data, time_window_seconds=1.0):
    """Find the closest force data to a frame timestamp"""
    closest_force = None
    min_time_diff = float('inf')
    
    for force_entry in force_data:
        time_diff = abs((frame_timestamp_dt - force_entry['timestamp_dt']).total_seconds())
        
        if time_diff < time_window_seconds and time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_force = force_entry
    
    return closest_force, min_time_diff

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

def create_angle_display(frame_num, elapsed_s, force_data=None):
    """Create a display showing frame info and force data"""
    # Create a semi-transparent background
    img = np.zeros((200, 350, 3), dtype=np.uint8)
    
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
        
        cv2.putText(img, f"  Distance: {force_data['distance_m']:.1f}m", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
        y_pos += line_height
        
        cv2.putText(img, f"  State: {force_data['strokestate']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['red'], 1)
    else:
        cv2.putText(img, "No force data", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['red'], 1)
    
    return img

def create_overlay_video_with_raw_data(video_path, raw_csv_path, frames_csv_path, output_dir):
    """Create overlay video using raw data with proper timestamp synchronization"""
    print("🎬 Creating Overlay Video with Raw Data Synchronization")
    print("=" * 60)
    
    # Load data
    force_data = load_raw_force_data(raw_csv_path)
    frame_timestamps = load_frame_timestamps(frames_csv_path)
    
    if not force_data:
        print("❌ No force data found")
        return
    
    if not frame_timestamps:
        print("❌ No frame timestamps found")
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
    output_path = os.path.join(output_dir, f"rowing_raw_data_overlay_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames with raw data overlays...")
    
    frame_count = 0
    overlays_added = 0
    force_overlays = 0
    sync_errors = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get frame timestamp
        if frame_count <= len(frame_timestamps):
            frame_ts = frame_timestamps[frame_count - 1]
            frame_timestamp_dt = frame_ts['timestamp_dt']
            
            # Calculate elapsed time from first frame
            first_frame_dt = frame_timestamps[0]['timestamp_dt']
            elapsed_s = (frame_timestamp_dt - first_frame_dt).total_seconds()
            
            # Find closest force data
            closest_force, time_diff = find_closest_force_data(frame_timestamp_dt, force_data)
            
            # Create angle display
            angle_display = create_angle_display(frame_count, elapsed_s, closest_force)
            
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
                sync_info = f" | Sync: {time_diff:.3f}s"
            else:
                sync_info = " | No force data"
            
            info_text = f"Frame {frame_count}/{total_frames} | Time: {elapsed_s:.1f}s{force_info}{sync_info}"
            cv2.putText(frame, info_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Track sync errors
            if time_diff > 0.5:  # More than 500ms difference
                sync_errors.append(time_diff)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n🎉 Overlay video complete!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   📈 Overlays added: {overlays_added}")
    print(f"   📊 Overlay rate: {(overlays_added/frame_count)*100:.1f}%")
    print(f"   🔋 Force plot overlays: {force_overlays}")
    print(f"   📊 Force overlay rate: {(force_overlays/frame_count)*100:.1f}%")
    
    if sync_errors:
        print(f"   ⚠️  Sync errors: {len(sync_errors)} frames with >500ms difference")
        print(f"   📊 Average sync error: {np.mean(sync_errors):.3f}s")
        print(f"   📊 Max sync error: {np.max(sync_errors):.3f}s")
    else:
        print(f"   ✅ Perfect synchronization achieved!")

def main():
    parser = argparse.ArgumentParser(description="Create overlay video with raw data")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--raw-csv", required=True, help="Path to raw CSV file")
    parser.add_argument("--frames-csv", required=True, help="Path to frames CSV file")
    parser.add_argument("--output-dir", default="raw_data_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.raw_csv):
        print(f"❌ Raw CSV file not found: {args.raw_csv}")
        return
    
    if not os.path.exists(args.frames_csv):
        print(f"❌ Frames CSV file not found: {args.frames_csv}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_overlay_video_with_raw_data(args.video, args.raw_csv, args.frames_csv, args.output_dir)

if __name__ == "__main__":
    main()
