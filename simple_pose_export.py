#!/usr/bin/env python3
"""
Simple Rowing Pose Data Export
Extracts all keypoint positions and body angles from rowing video
"""

import cv2
import json
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import math

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p2 is the vertex)"""
    if p1 is None or p2 is None or p3 is None:
        return None
    
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def main():
    """Extract pose data from rowing video"""
    print("🚣‍♂️ Simple Rowing Pose Data Export")
    print("=" * 50)
    
    # Input video
    video_path = "/Users/kelsorj/Desktop/rowing/122404-122447.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return
    
    # Video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    
    # Load YOLO model
    model = YOLO('yolo11n-pose.pt')
    print("🤖 Loaded YOLO11 pose model")
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rowing_pose_data_{timestamp}.json"
    
    # Keypoint names (COCO format)
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    all_data = []
    frame_count = 0
    
    print("📊 Processing frames...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Run pose detection
        results = model(frame, verbose=False)
        
        frame_data = {
            'frame_number': frame_count,
            'timestamp': frame_count / fps,
            'frame_time': datetime.now().isoformat()
        }
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            # Get keypoints for first person
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # Store all keypoint positions
            for i, kpt in enumerate(keypoints):
                if i < len(keypoint_names):
                    name = keypoint_names[i]
                    frame_data[f'{name}_x'] = float(kpt[0])
                    frame_data[f'{name}_y'] = float(kpt[1])
                    frame_data[f'{name}_confidence'] = float(kpt[2])
            
            # Calculate key rowing angles
            # Left arm angle (shoulder-elbow-wrist)
            if (frame_data.get('left_shoulder_confidence', 0) > 0.5 and
                frame_data.get('left_elbow_confidence', 0) > 0.5 and
                frame_data.get('left_wrist_confidence', 0) > 0.5):
                
                p1 = (frame_data['left_shoulder_x'], frame_data['left_shoulder_y'])
                p2 = (frame_data['left_elbow_x'], frame_data['left_elbow_y'])
                p3 = (frame_data['left_wrist_x'], frame_data['left_wrist_y'])
                frame_data['left_arm_angle'] = calculate_angle(p1, p2, p3)
            
            # Right arm angle
            if (frame_data.get('right_shoulder_confidence', 0) > 0.5 and
                frame_data.get('right_elbow_confidence', 0) > 0.5 and
                frame_data.get('right_wrist_confidence', 0) > 0.5):
                
                p1 = (frame_data['right_shoulder_x'], frame_data['right_shoulder_y'])
                p2 = (frame_data['right_elbow_x'], frame_data['right_elbow_y'])
                p3 = (frame_data['right_wrist_x'], frame_data['right_wrist_y'])
                frame_data['right_arm_angle'] = calculate_angle(p1, p2, p3)
            
            # Left leg angle (hip-knee-ankle)
            if (frame_data.get('left_hip_confidence', 0) > 0.5 and
                frame_data.get('left_knee_confidence', 0) > 0.5 and
                frame_data.get('left_ankle_confidence', 0) > 0.5):
                
                p1 = (frame_data['left_hip_x'], frame_data['left_hip_y'])
                p2 = (frame_data['left_knee_x'], frame_data['left_knee_y'])
                p3 = (frame_data['left_ankle_x'], frame_data['left_ankle_y'])
                frame_data['left_leg_angle'] = calculate_angle(p1, p2, p3)
            
            # Right leg angle
            if (frame_data.get('right_hip_confidence', 0) > 0.5 and
                frame_data.get('right_knee_confidence', 0) > 0.5 and
                frame_data.get('right_ankle_confidence', 0) > 0.5):
                
                p1 = (frame_data['right_hip_x'], frame_data['right_hip_y'])
                p2 = (frame_data['right_knee_x'], frame_data['right_knee_y'])
                p3 = (frame_data['right_ankle_x'], frame_data['right_ankle_y'])
                frame_data['right_leg_angle'] = calculate_angle(p1, p2, p3)
            
            # Torso lean angle
            if (frame_data.get('left_shoulder_confidence', 0) > 0.5 and
                frame_data.get('right_shoulder_confidence', 0) > 0.5 and
                frame_data.get('left_hip_confidence', 0) > 0.5 and
                frame_data.get('right_hip_confidence', 0) > 0.5):
                
                shoulder_center_x = (frame_data['left_shoulder_x'] + frame_data['right_shoulder_x']) / 2
                shoulder_center_y = (frame_data['left_shoulder_y'] + frame_data['right_shoulder_y']) / 2
                hip_center_x = (frame_data['left_hip_x'] + frame_data['right_hip_x']) / 2
                hip_center_y = (frame_data['left_hip_y'] + frame_data['right_hip_y']) / 2
                
                dx = shoulder_center_x - hip_center_x
                dy = shoulder_center_y - hip_center_y
                frame_data['torso_lean_angle'] = math.degrees(math.atan2(dx, dy))
        
        all_data.append(frame_data)
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    
    # Save data
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n🎉 Export complete!")
    print(f"   📋 Data saved to: {output_file}")
    print(f"   📈 Total frames: {frame_count}")
    
    # Summary statistics
    if all_data:
        print(f"\n📊 Body Angle Summary:")
        angle_keys = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle', 'torso_lean_angle']
        for angle_key in angle_keys:
            values = [d.get(angle_key) for d in all_data if d.get(angle_key) is not None]
            if values:
                print(f"   {angle_key}: {min(values):.1f}° - {max(values):.1f}° (avg: {np.mean(values):.1f}°)")

if __name__ == "__main__":
    main()
