#!/usr/bin/env python3
"""
Rowing Pose Analysis with Ultralytics
Extracts body angles and keypoint positions for rowing technique analysis
"""

import cv2
import json
import csv
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import math
from datetime import datetime

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p2 is the vertex)"""
    if p1 is None or p2 is None or p3 is None:
        return None
    
    # Convert to numpy arrays
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def get_keypoint_name(index):
    """Get human-readable name for keypoint index"""
    keypoint_names = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
    }
    return keypoint_names.get(index, f"keypoint_{index}")

def analyze_rowing_pose(keypoints):
    """Analyze rowing-specific body angles and positions"""
    if keypoints is None or len(keypoints) < 17:
        return {}
    
    # Extract key points (assuming COCO format)
    nose = keypoints[0] if keypoints[0][2] > 0.5 else None
    left_shoulder = keypoints[5] if keypoints[5][2] > 0.5 else None
    right_shoulder = keypoints[6] if keypoints[6][2] > 0.5 else None
    left_elbow = keypoints[7] if keypoints[7][2] > 0.5 else None
    right_elbow = keypoints[8] if keypoints[8][2] > 0.5 else None
    left_wrist = keypoints[9] if keypoints[9][2] > 0.5 else None
    right_wrist = keypoints[10] if keypoints[10][2] > 0.5 else None
    left_hip = keypoints[11] if keypoints[11][2] > 0.5 else None
    right_hip = keypoints[12] if keypoints[12][2] > 0.5 else None
    left_knee = keypoints[13] if keypoints[13][2] > 0.5 else None
    right_knee = keypoints[14] if keypoints[14][2] > 0.5 else None
    left_ankle = keypoints[15] if keypoints[15][2] > 0.5 else None
    right_ankle = keypoints[16] if keypoints[16][2] > 0.5 else None
    
    analysis = {}
    
    # Calculate key rowing angles
    if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
        analysis['left_arm_angle'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
        analysis['right_arm_angle'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    if left_hip is not None and left_knee is not None and left_ankle is not None:
        analysis['left_leg_angle'] = calculate_angle(left_hip, left_knee, left_ankle)
    
    if right_hip is not None and right_knee is not None and right_ankle is not None:
        analysis['right_leg_angle'] = calculate_angle(right_hip, right_knee, right_ankle)
    
    if left_shoulder is not None and left_hip is not None and left_knee is not None:
        analysis['left_hip_angle'] = calculate_angle(left_shoulder, left_hip, left_knee)
    
    if right_shoulder is not None and right_hip is not None and right_knee is not None:
        analysis['right_hip_angle'] = calculate_angle(right_shoulder, right_hip, right_knee)
    
    # Torso angle (shoulder to hip)
    if (left_shoulder is not None and right_shoulder is not None and 
        left_hip is not None and right_hip is not None):
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                          (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                     (left_hip[1] + right_hip[1]) / 2)
        
        # Calculate torso angle relative to vertical
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        analysis['torso_angle'] = math.degrees(math.atan2(dx, dy))
    
    # Store all keypoint positions
    for i, kpt in enumerate(keypoints):
        if kpt[2] > 0.5:  # Confidence threshold
            analysis[f'{get_keypoint_name(i)}_x'] = kpt[0]
            analysis[f'{get_keypoint_name(i)}_y'] = kpt[1]
            analysis[f'{get_keypoint_name(i)}_confidence'] = kpt[2]
    
    return analysis

def main():
    """Main analysis function"""
    print("🚣‍♂️ Rowing Pose Analysis with Body Angles")
    print("=" * 50)
    
    # Input video path
    video_path = "/Users/kelsorj/Desktop/rowing/122404-122447.mp4"
    
    # Check if video exists
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    
    # Initialize YOLO pose model
    model = YOLO('yolo11n-pose.pt')
    print("🤖 Loaded YOLO11 pose model")
    
    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = f"rowing_pose_analysis_{timestamp}.mp4"
    output_csv = f"rowing_pose_data_{timestamp}.csv"
    output_json = f"rowing_pose_data_{timestamp}.json"
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    # CSV writer
    csv_file = open(output_csv, 'w', newline='')
    csv_writer = None
    
    # JSON data storage
    json_data = []
    
    frame_count = 0
    print("📊 Processing video frames...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Run pose detection
        results = model(frame, verbose=False)
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            # Get keypoints for the first person detected
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # Analyze pose
            pose_analysis = analyze_rowing_pose(keypoints)
            
            # Add frame info
            pose_analysis['frame_number'] = frame_count
            pose_analysis['timestamp'] = frame_count / fps
            pose_analysis['frame_time'] = datetime.now().isoformat()
            
            # Initialize CSV writer with headers from first frame
            if csv_writer is None and pose_analysis:
                fieldnames = list(pose_analysis.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
            
            # Write to CSV (only if we have a writer and the data matches fieldnames)
            if csv_writer and pose_analysis:
                # Ensure all required fields are present
                row_data = {}
                for field in csv_writer.fieldnames:
                    row_data[field] = pose_analysis.get(field, None)
                csv_writer.writerow(row_data)
            
            # Store for JSON (convert numpy types to Python types)
            json_pose_analysis = {}
            for key, value in pose_analysis.items():
                if hasattr(value, 'item'):  # numpy scalar
                    json_pose_analysis[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    json_pose_analysis[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                else:
                    json_pose_analysis[key] = value
            json_data.append(json_pose_analysis)
            
            # Draw pose on frame
            annotator = Annotator(frame, line_width=2)
            
            # Draw keypoints
            for i, kpt in enumerate(keypoints):
                if kpt[2] > 0.5:  # Confidence threshold
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, get_keypoint_name(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw skeleton connections
            skeleton = [
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            
            for start_idx, end_idx in skeleton:
                if (keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
            
            # Add angle information to frame
            y_offset = 30
            for angle_name, angle_value in pose_analysis.items():
                if 'angle' in angle_name and angle_value is not None:
                    text = f"{angle_name}: {angle_value:.1f}°"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
        
        # Write frame to output video
        video_writer.write(frame)
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    video_writer.release()
    csv_file.close()
    
    # Save JSON data
    with open(output_json, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n🎉 Analysis complete!")
    print(f"   📹 Output video: {output_video}")
    print(f"   📊 CSV data: {output_csv}")
    print(f"   📋 JSON data: {output_json}")
    print(f"   📈 Total frames processed: {frame_count}")
    
    # Summary statistics
    if json_data:
        print(f"\n📊 Body Angle Summary:")
        angle_keys = [k for k in json_data[0].keys() if 'angle' in k]
        for angle_key in angle_keys:
            values = [d.get(angle_key) for d in json_data if d.get(angle_key) is not None]
            if values:
                print(f"   {angle_key}: {min(values):.1f}° - {max(values):.1f}° (avg: {np.mean(values):.1f}°)")

if __name__ == "__main__":
    main()
