#!/usr/bin/env python3
"""
Create Overlay Preview
Shows what the force and angle overlays look like on a single frame
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import glob
import os

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
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def create_angle_display(left_arm, right_arm, left_leg, right_leg, torso, frame_num, elapsed_s):
    """Create a comprehensive display of body angles and info"""
    # Create a semi-transparent background
    img = np.zeros((250, 350, 3), dtype=np.uint8)
    
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
    
    if left_arm is not None:
        cv2.putText(img, f"  L Arm: {left_arm:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
        y_pos += line_height
    
    if right_arm is not None:
        cv2.putText(img, f"  R Arm: {right_arm:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
        y_pos += line_height
    
    # Leg angles
    cv2.putText(img, "LEG ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
    y_pos += line_height
    
    if left_leg is not None:
        cv2.putText(img, f"  L Leg: {left_leg:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
        y_pos += line_height
    
    if right_leg is not None:
        cv2.putText(img, f"  R Leg: {right_leg:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
        y_pos += line_height
    
    # Torso angle
    cv2.putText(img, "TORSO:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
    y_pos += line_height
    
    if torso is not None:
        cv2.putText(img, f"  Lean: {torso:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
    
    return img

def main():
    """Create preview images showing the overlays"""
    print("🎬 Creating Overlay Preview Images")
    print("=" * 40)
    
    # Load demo data
    with open("demo_synchronized_data.json", 'r') as f:
        demo_data = json.load(f)
    
    # Load pose data
    pose_files = glob.glob("rowing_pose_data_*.json")
    latest_pose_file = max(pose_files, key=os.path.getctime)
    
    with open(latest_pose_file, 'r') as f:
        pose_data = json.load(f)
    
    # Create a sample frame (1920x1080)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)  # Dark gray background
    
    # Add some text to simulate video content
    cv2.putText(frame, "ROWING VIDEO FRAME", (800, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "This is where the rowing video would be", (700, 600), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    
    # Get sample data
    sample_pose = pose_data[100]  # Middle of the video
    sample_demo = demo_data[2]    # Sample force data
    
    # Create overlays
    angle_display = create_angle_display(
        sample_pose.get('left_arm_angle'),
        sample_pose.get('right_arm_angle'),
        sample_pose.get('left_leg_angle'),
        sample_pose.get('right_leg_angle'),
        sample_pose.get('torso_lean_angle'),
        100,
        sample_pose.get('timestamp', 0)
    )
    
    force_plot = create_force_curve_plot(
        sample_demo['force_curve'],
        sample_demo['power'],
        sample_demo['spm'],
        sample_demo.get('distance', 0),
        sample_demo['elapsed_s']
    )
    
    # Overlay angle display (top-left)
    if angle_display is not None:
        angle_height, angle_width = angle_display.shape[:2]
        frame[10:10+angle_height, 10:10+angle_width] = angle_display
    
    # Overlay force plot (top-right)
    if force_plot is not None:
        plot_height, plot_width = force_plot.shape[:2]
        x_offset = 1920 - plot_width - 10
        frame[10:10+plot_height, x_offset:x_offset+plot_width] = force_plot
    
    # Add info text
    info_text = "OVERLAY PREVIEW - Force curves and body angles synchronized with video"
    cv2.putText(frame, info_text, (10, 1050), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save preview
    cv2.imwrite("overlay_preview.png", frame)
    
    # Also save individual components
    if angle_display is not None:
        cv2.imwrite("angle_display_preview.png", angle_display)
    
    if force_plot is not None:
        cv2.imwrite("force_plot_preview.png", force_plot)
    
    print("✅ Preview images created:")
    print("   📸 overlay_preview.png - Complete frame with overlays")
    print("   📸 angle_display_preview.png - Body angles display")
    print("   📸 force_plot_preview.png - Force curve plot")
    
    print(f"\n📊 Sample Data Used:")
    print(f"   Pose: Frame 100, Time {sample_pose.get('timestamp', 0):.1f}s")
    print(f"   Force: {sample_demo['power']}W, {sample_demo['spm']}spm, Peak {max(sample_demo['force_curve'])}")

if __name__ == "__main__":
    main()
