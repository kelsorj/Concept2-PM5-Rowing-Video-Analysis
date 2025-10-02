#!/usr/bin/env python3
"""
Create Analysis Video Preview
Shows what the advanced analysis video overlays look like
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import glob
import os
from datetime import datetime

def create_joint_angles_display(pose_frame, frame_num, elapsed_s):
    """Create joint angles display overlay"""
    # Create a semi-transparent background
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Define colors
    colors = {
        'white': (255, 255, 255),
        'lime': (0, 255, 0),
        'cyan': (255, 255, 0),
        'yellow': (0, 255, 255),
        'red': (0, 0, 255),
        'orange': (0, 165, 255),
        'magenta': (255, 0, 255)
    }
    
    # Title
    cv2.putText(img, "JOINT ANGLES ANALYSIS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['white'], 2)
    
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
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
        y_pos += line_height
    
    # Leg angles
    cv2.putText(img, "LEG ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
    y_pos += line_height
    
    if pose_frame.get('left_leg_angle') is not None:
        left_leg = pose_frame['left_leg_angle']
        cv2.putText(img, f"  L Leg: {left_leg:.1f}°", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
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

def create_power_curve_display():
    """Create sample power curve display"""
    # Create matplotlib plot
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Sample force curve data
    force_curve = [0, 15, 35, 60, 85, 110, 135, 155, 170, 180, 185, 180, 170, 155, 135, 110, 85, 60, 35, 15, 0]
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
    ax.set_title('Force Curve - 235W', color='white', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white', labelsize=8)
    
    # Add stats text
    stats_text = f'Peak: {peak_force}\nAvg: {np.mean(force_curve):.1f}\nPower: 235W'
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

def create_analysis_summary_display():
    """Create analysis summary display"""
    # Create a semi-transparent background
    img = np.zeros((150, 500, 3), dtype=np.uint8)
    
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
    cv2.putText(img, "ANALYSIS SUMMARY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['white'], 2)
    
    y_pos = 50
    line_height = 20
    
    # Sample analysis results
    cv2.putText(img, "Symmetry Score: 95.0%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
    y_pos += line_height
    
    cv2.putText(img, "Technique Score: 42.2%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
    y_pos += line_height
    
    cv2.putText(img, "Stroke Rate: 20.7 spm", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
    y_pos += line_height
    
    cv2.putText(img, "Current Power: 235W", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
    
    return img

def main():
    """Create preview of analysis video overlays"""
    print("🎬 Creating Analysis Video Preview")
    print("=" * 40)
    
    # Load sample pose data
    pose_files = glob.glob("rowing_pose_data_*.json")
    if not pose_files:
        print("❌ No pose data files found")
        return
    
    latest_pose_file = max(pose_files, key=os.path.getctime)
    with open(latest_pose_file, 'r') as f:
        pose_data = json.load(f)
    
    # Create a sample frame (1920x1080)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)  # Dark gray background
    
    # Add some text to simulate video content
    cv2.putText(frame, "ROWING VIDEO FRAME", (800, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "Advanced Analysis Overlay Preview", (700, 600), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    
    # Get sample data
    sample_pose = pose_data[100]  # Middle of the video
    
    # Create overlays
    angles_display = create_joint_angles_display(sample_pose, 100, sample_pose.get('timestamp', 0))
    power_display = create_power_curve_display()
    summary_display = create_analysis_summary_display()
    
    # Overlay displays on frame
    # Top-left: Joint angles
    frame[10:10+300, 10:10+400] = angles_display
    
    # Top-right: Power curve
    power_height, power_width = power_display.shape[:2]
    x_offset = 1920 - power_width - 10
    frame[10:10+power_height, x_offset:x_offset+power_width] = power_display
    
    # Bottom-left: Analysis summary
    summary_height, summary_width = summary_display.shape[:2]
    frame[1080-summary_height-10:1080-10, 10:10+summary_width] = summary_display
    
    # Add frame info
    info_text = "ADVANCED ANALYSIS OVERLAY PREVIEW - Joint angles, power curves, and analysis summary"
    cv2.putText(frame, info_text, (10, 1050), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save preview
    cv2.imwrite("advanced_analysis_video_preview.png", frame)
    
    # Also save individual components
    cv2.imwrite("joint_angles_display_preview.png", angles_display)
    cv2.imwrite("power_curve_display_preview.png", power_display)
    cv2.imwrite("analysis_summary_display_preview.png", summary_display)
    
    print("✅ Preview images created:")
    print("   📸 advanced_analysis_video_preview.png - Complete frame with all overlays")
    print("   📸 joint_angles_display_preview.png - Joint angles display")
    print("   📸 power_curve_display_preview.png - Power curve display")
    print("   📸 analysis_summary_display_preview.png - Analysis summary display")
    
    print(f"\n📊 Sample Data Used:")
    print(f"   Pose: Frame 100, Time {sample_pose.get('timestamp', 0):.1f}s")
    print(f"   Joint Angles: L Arm {sample_pose.get('left_arm_angle', 0):.1f}°, R Arm {sample_pose.get('right_arm_angle', 0):.1f}°")
    print(f"   Leg Angles: L Leg {sample_pose.get('left_leg_angle', 0):.1f}°, R Leg {sample_pose.get('right_leg_angle', 0):.1f}°")

if __name__ == "__main__":
    main()
