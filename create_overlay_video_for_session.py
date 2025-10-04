#!/usr/bin/env python3
"""
Create overlay video for the specific session data
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from datetime import datetime
import argparse

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
    if current_idx is not None:
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
    peak_force = max(force_curve)
    stats_text = f'Peak: {peak_force}\nAvg: {np.mean(force_curve):.1f}\nDist: {distance:.1f}m'
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

def create_angle_display(pose_frame, frame_num, elapsed_s):
    """Create a comprehensive display of body angles and info"""
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
    line_height = 25
    
    # Arm angles
    cv2.putText(img, "ARM ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
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
    
    # Leg angles
    cv2.putText(img, "LEG ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
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
    
    # Torso angle
    cv2.putText(img, "TORSO:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
    y_pos += line_height
    
    if pose_frame.get('torso_lean_angle') is not None:
        torso = pose_frame['torso_lean_angle']
        cv2.putText(img, f"  Lean: {torso:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
    
    return img

def create_angle_trend_chart(pose_data, current_frame_idx):
    """Create a mini chart showing angle trends"""
    if current_frame_idx < 10:
        return None
    
    # Get recent angle data
    recent_frames = pose_data[max(0, current_frame_idx-30):current_frame_idx+1]
    
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
    buf = buf[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def create_overlay_video(video_path, synchronized_data_path, output_dir):
    """Create overlay video with force plots and body angles"""
    print("🎬 Creating Overlay Video with Force and Angle Data")
    print("=" * 60)
    
    # Load synchronized data
    with open(synchronized_data_path, 'r') as f:
        synchronized_data = json.load(f)
    
    print(f"📊 Loaded {len(synchronized_data)} synchronized data points")
    
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
    output_path = os.path.join(output_dir, f"rowing_overlay_analysis_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames with overlays...")
    
    frame_count = 0
    overlays_added = 0
    force_overlays = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get synchronized data for this frame
        if frame_count <= len(synchronized_data):
            sync_entry = synchronized_data[frame_count - 1]
            pose_frame = sync_entry.get('pose_data', {})
            force_data = sync_entry.get('force_data', {})
            elapsed_s = sync_entry.get('frame_elapsed_s', 0)
            
            # Create angle display
            angle_display = create_angle_display(pose_frame, frame_count, elapsed_s)
            
            # Create force curve plot if available
            force_plot = None
            if force_data and force_data.get('forceplot'):
                force_curve = force_data['forceplot']
                if force_curve:
                    # Calculate animated position in force curve
                    current_idx = None
                    if len(force_curve) > 0:
                        # Simple animation: cycle through force curve points
                        animation_cycle = (frame_count % (len(force_curve) * 2)) // 2
                        current_idx = min(animation_cycle, len(force_curve) - 1)
                    
                    force_plot = create_force_curve_plot(
                        force_curve,
                        force_data.get('power', 0),
                        force_data.get('spm', 0),
                        force_data.get('distance_m', 0),
                        elapsed_s,
                        current_idx
                    )
            
            # Create angle trend chart
            angle_chart = create_angle_trend_chart([entry['pose_data'] for entry in synchronized_data], frame_count - 1)
            
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
            
            # Overlay angle chart (bottom-right)
            if angle_chart is not None:
                chart_height, chart_width = angle_chart.shape[:2]
                chart_x_offset = width - chart_width - 10
                chart_y_offset = height - chart_height - 10
                
                if (chart_x_offset >= 0 and 
                    chart_y_offset >= 0):
                    frame[chart_y_offset:chart_y_offset+chart_height, 
                          chart_x_offset:chart_x_offset+chart_width] = angle_chart
            
            # Add frame info at bottom
            force_info = ""
            if force_data:
                force_info = f" | Power: {force_data.get('power', 0)}W | SPM: {force_data.get('spm', 0)}"
            
            info_text = f"Frame {frame_count}/{total_frames} | Time: {elapsed_s:.1f}s{force_info}"
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
    
    print(f"\n🎉 Overlay video complete!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   📈 Overlays added: {overlays_added}")
    print(f"   📊 Overlay rate: {(overlays_added/frame_count)*100:.1f}%")
    print(f"   🔋 Force plot overlays: {force_overlays}")
    print(f"   📊 Force overlay rate: {(force_overlays/frame_count)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Create overlay video for session data")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--sync-data", required=True, help="Path to synchronized data JSON")
    parser.add_argument("--output-dir", default="synchronized_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.sync_data):
        print(f"❌ Synchronized data file not found: {args.sync_data}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_overlay_video(args.video, args.sync_data, args.output_dir)

if __name__ == "__main__":
    main()
