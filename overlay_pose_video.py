#!/usr/bin/env python3
"""
Overlay Body Angles on Video
Creates a video showing body angles and pose keypoints in real-time
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import glob
import os
from datetime import datetime

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

def create_angle_chart(pose_data, current_frame_idx):
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

def overlay_pose_on_video():
    """Main function to overlay pose data on video"""
    print("🎬 Creating Video with Pose Angle Overlays")
    print("=" * 50)
    
    # Load pose data
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
    output_path = f"rowing_pose_overlay_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Processing frames...")
    
    frame_count = 0
    overlays_added = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get pose data for this frame
        if frame_count <= len(pose_data):
            pose_frame = pose_data[frame_count - 1]
            elapsed_s = pose_frame.get('timestamp', 0)
            
            # Create angle display
            angle_display = create_angle_display(
                pose_frame.get('left_arm_angle'),
                pose_frame.get('right_arm_angle'),
                pose_frame.get('left_leg_angle'),
                pose_frame.get('right_leg_angle'),
                pose_frame.get('torso_lean_angle'),
                frame_count,
                elapsed_s
            )
            
            # Create angle trend chart
            angle_chart = create_angle_chart(pose_data, frame_count - 1)
            
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
            
            # Overlay angle chart (top-right)
            if angle_chart is not None:
                chart_height, chart_width = angle_chart.shape[:2]
                chart_x_offset = width - chart_width - 10
                chart_y_offset = 10
                
                if (chart_x_offset >= 0 and 
                    chart_y_offset + chart_height <= height):
                    frame[chart_y_offset:chart_y_offset+chart_height, 
                          chart_x_offset:chart_x_offset+chart_width] = angle_chart
            
            # Add frame info at bottom
            info_text = f"Frame {frame_count}/{total_frames} | Time: {elapsed_s:.1f}s | FPS: {fps}"
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
    
    print(f"\n🎉 Video overlay complete!")
    print(f"   📹 Output video: {output_path}")
    print(f"   📊 Total frames: {frame_count}")
    print(f"   📈 Overlays added: {overlays_added}")
    print(f"   📊 Overlay rate: {(overlays_added/frame_count)*100:.1f}%")

def main():
    """Main function"""
    print("🚣‍♂️ Rowing Video with Pose Angle Overlays")
    print("=" * 50)
    
    overlay_pose_on_video()

if __name__ == "__main__":
    main()
