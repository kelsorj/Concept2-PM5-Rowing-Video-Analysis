#!/usr/bin/env python3
"""
Overlay Force Data and Body Angles on Video
Creates a synchronized video showing force curves and body angles in real-time
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import glob
import os
from datetime import datetime

def load_force_data():
    """Load the most recent force data"""
    force_files = glob.glob("pm5_py3row_parsed_*.csv")
    if not force_files:
        print("❌ No force data files found")
        return None
    
    latest_force_file = max(force_files, key=os.path.getctime)
    print(f"📊 Loading force data: {latest_force_file}")
    
    force_data = []
    with open(latest_force_file, 'r') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            if row['forceplot'] and row['forceplot'] != '':
                try:
                    force_curve = json.loads(row['forceplot'])
                    if force_curve:
                        force_data.append({
                            'elapsed_s': float(row['elapsed_s']),
                            'force_curve': force_curve,
                            'power': int(row['avg_power_w']),
                            'spm': int(row['spm']),
                            'distance': float(row['distance_m'])
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
    
    print(f"   Found {len(force_data)} complete force curves")
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

def find_closest_force_data(pose_elapsed_s, force_data, time_window=2.0):
    """Find the closest force data to a pose timestamp"""
    closest_force = None
    min_time_diff = float('inf')
    
    for force_entry in force_data:
        time_diff = abs(pose_elapsed_s - force_entry['elapsed_s'])
        
        if time_diff < time_window and time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_force = force_entry
    
    return closest_force, min_time_diff

def create_force_curve_plot(force_curve, power, spm, distance, elapsed_s):
    """Create a matplotlib plot of the force curve"""
    if not force_curve:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot force curve
    x = np.arange(len(force_curve))
    ax.plot(x, force_curve, 'lime', linewidth=3, label='Force')
    
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
    stats_text = f'Peak: {peak_force}\nAvg: {np.mean(force_curve):.1f}\nDist: {distance:.1f}m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', color='white', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def create_angle_display(left_arm, right_arm, left_leg, right_leg, torso):
    """Create a text display of body angles"""
    # Create a black background
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Define colors
    colors = {
        'white': (255, 255, 255),
        'lime': (0, 255, 0),
        'cyan': (255, 255, 0),
        'yellow': (0, 255, 255),
        'red': (0, 0, 255)
    }
    
    # Title
    cv2.putText(img, "BODY ANGLES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['white'], 2)
    
    y_pos = 50
    line_height = 30
    
    # Arm angles
    if left_arm is not None:
        cv2.putText(img, f"L Arm: {left_arm:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['lime'], 2)
        y_pos += line_height
    
    if right_arm is not None:
        cv2.putText(img, f"R Arm: {right_arm:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['lime'], 2)
        y_pos += line_height
    
    # Leg angles
    if left_leg is not None:
        cv2.putText(img, f"L Leg: {left_leg:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['cyan'], 2)
        y_pos += line_height
    
    if right_leg is not None:
        cv2.putText(img, f"R Leg: {right_leg:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['cyan'], 2)
        y_pos += line_height
    
    # Torso angle
    if torso is not None:
        cv2.putText(img, f"Torso: {torso:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['yellow'], 2)
    
    return img

def overlay_data_on_video():
    """Main function to overlay force and angle data on video"""
    print("🎬 Creating Video with Force and Angle Overlays")
    print("=" * 50)
    
    # Load data
    force_data = load_force_data()
    if force_data is None:
        return
    
    pose_data = load_pose_data()
    if pose_data is None:
        return
    
    # Find the video file
    video_files = glob.glob("runs/pose/predict/*.mp4")
    if not video_files:
        print("❌ No video files found in runs/pose/predict/")
        return
    
    video_path = video_files[0]
    print(f"📹 Processing video: {video_path}")
    
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
    
    print(f"   Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"rowing_with_overlays_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames...")
    
    frame_count = 0
    correlations_found = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get pose data for this frame
        if frame_count <= len(pose_data):
            pose_frame = pose_data[frame_count - 1]
            pose_elapsed_s = pose_frame.get('timestamp', 0)
            
            # Find closest force data
            closest_force, time_diff = find_closest_force_data(pose_elapsed_s, force_data)
            
            if closest_force and time_diff < 2.0:  # Within 2 seconds
                correlations_found += 1
                
                # Create force curve plot
                force_plot = create_force_curve_plot(
                    closest_force['force_curve'],
                    closest_force['power'],
                    closest_force['spm'],
                    closest_force['distance'],
                    closest_force['elapsed_s']
                )
                
                # Create angle display
                angle_display = create_angle_display(
                    pose_frame.get('left_arm_angle'),
                    pose_frame.get('right_arm_angle'),
                    pose_frame.get('left_leg_angle'),
                    pose_frame.get('right_leg_angle'),
                    pose_frame.get('torso_lean_angle')
                )
                
                # Overlay force plot (top-right)
                if force_plot is not None:
                    plot_height, plot_width = force_plot.shape[:2]
                    x_offset = width - plot_width - 10
                    y_offset = 10
                    
                    # Ensure the plot fits within the frame
                    if x_offset >= 0 and y_offset + plot_height <= height:
                        frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = force_plot
                
                # Overlay angle display (top-left)
                angle_height, angle_width = angle_display.shape[:2]
                angle_x_offset = 10
                angle_y_offset = 10
                
                if angle_x_offset + angle_width <= width and angle_y_offset + angle_height <= height:
                    frame[angle_y_offset:angle_y_offset+angle_height, 
                          angle_x_offset:angle_x_offset+angle_width] = angle_display
                
                # Add time and correlation info
                time_text = f"Time: {pose_elapsed_s:.1f}s | Force: {closest_force['elapsed_s']:.1f}s | Diff: {time_diff:.2f}s"
                cv2.putText(frame, time_text, (10, height - 20), 
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
    
    print(f"\n🎉 Video overlay complete!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   🔗 Correlations found: {correlations_found}")
    print(f"   📈 Correlation rate: {(correlations_found/frame_count)*100:.1f}%")

def main():
    """Main function"""
    print("🚣‍♂️ Rowing Video with Force & Angle Overlays")
    print("=" * 50)
    
    overlay_data_on_video()

if __name__ == "__main__":
    main()
