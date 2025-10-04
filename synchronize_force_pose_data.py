#!/usr/bin/env python3
"""
Synchronize force data with pose data using video metadata timestamps
"""

import json
import csv
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os
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
                # Parse ISO format timestamp
                return datetime.fromisoformat(creation_time_str.replace('Z', '+00:00'))
        
        return None
    except Exception as e:
        print(f"⚠️  Error extracting video metadata: {e}")
        return None

def synchronize_data(video_path, force_csv_path, pose_json_path, frames_csv_path, output_dir):
    """Synchronize force and pose data using video timestamps"""
    print("🔄 Synchronizing force and pose data...")
    
    # Extract video creation time
    video_creation_time = extract_video_creation_time(video_path)
    if not video_creation_time:
        print("❌ Could not extract video creation time")
        return False
    
    print(f"📹 Video creation time: {video_creation_time}")
    
    # Load force data
    print("📊 Loading force data...")
    force_data = []
    with open(force_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                force_data.append({
                    'timestamp_ns': int(row['timestamp_ns']),
                    'timestamp_iso': row['timestamp_iso'],
                    'elapsed_s': float(row['elapsed_s']),
                    'distance_m': float(row['distance_m']),
                    'spm': int(row['spm']),
                    'power': int(row['power']),
                    'pace': float(row['pace']),
                    'forceplot': json.loads(row['forceplot']),
                    'strokestate': row['strokestate'],
                    'status': row['status']
                })
            except (ValueError, json.JSONDecodeError) as e:
                print(f"⚠️  Error parsing force data row: {e}")
                continue
    
    print(f"   Loaded {len(force_data)} force data points")
    
    # Load pose data
    print("📊 Loading pose data...")
    with open(pose_json_path, 'r') as f:
        pose_data = json.load(f)
    
    print(f"   Loaded {len(pose_data)} pose data points")
    
    # Load frame timestamps
    print("📊 Loading frame timestamps...")
    frame_timestamps = []
    with open(frames_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_timestamps.append({
                    'frame_idx': int(row['frame_idx']),
                    'ts_ns': int(row['ts_ns']),
                    'ts_iso': row['ts_iso']
                })
            except (ValueError, KeyError) as e:
                print(f"⚠️  Error parsing frame timestamp row: {e}")
                continue
    
    print(f"   Loaded {len(frame_timestamps)} frame timestamps")
    
    # Create synchronized dataset
    print("🔄 Creating synchronized dataset...")
    synchronized_data = []
    
    for i, pose_frame in enumerate(pose_data):
        if i < len(frame_timestamps):
            frame_ts = frame_timestamps[i]
            
            # Calculate frame time relative to video start
            frame_elapsed_s = i / 30.0  # Assuming 30fps from metadata
            
            # Find closest force data point
            closest_force = None
            min_time_diff = float('inf')
            
            for force_point in force_data:
                time_diff = abs(force_point['elapsed_s'] - frame_elapsed_s)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_force = force_point
            
            # Create synchronized entry
            sync_entry = {
                'frame_idx': i,
                'frame_elapsed_s': frame_elapsed_s,
                'frame_timestamp_ns': frame_ts['ts_ns'],
                'frame_timestamp_iso': frame_ts['ts_iso'],
                'pose_data': pose_frame,
                'force_data': closest_force,
                'time_sync_diff_s': min_time_diff if closest_force else None
            }
            
            synchronized_data.append(sync_entry)
    
    print(f"   Created {len(synchronized_data)} synchronized data points")
    
    # Save synchronized data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    sync_json_path = os.path.join(output_dir, 'synchronized_data.json')
    with open(sync_json_path, 'w') as f:
        json.dump(synchronized_data, f, indent=2, default=str)
    
    # Save as CSV (flattened for analysis)
    sync_csv_path = os.path.join(output_dir, 'synchronized_data.csv')
    with open(sync_csv_path, 'w', newline='') as f:
        if synchronized_data:
            # Create flattened structure
            flattened_data = []
            for entry in synchronized_data:
                flat_entry = {
                    'frame_idx': entry['frame_idx'],
                    'frame_elapsed_s': entry['frame_elapsed_s'],
                    'frame_timestamp_ns': entry['frame_timestamp_ns'],
                    'frame_timestamp_iso': entry['frame_timestamp_iso'],
                    'time_sync_diff_s': entry['time_sync_diff_s']
                }
                
                # Add pose data
                if entry['pose_data']:
                    for key, value in entry['pose_data'].items():
                        flat_entry[f'pose_{key}'] = value
                
                # Add force data
                if entry['force_data']:
                    flat_entry['force_elapsed_s'] = entry['force_data']['elapsed_s']
                    flat_entry['force_power'] = entry['force_data']['power']
                    flat_entry['force_spm'] = entry['force_data']['spm']
                    flat_entry['force_distance_m'] = entry['force_data']['distance_m']
                    flat_entry['force_strokestate'] = entry['force_data']['strokestate']
                    flat_entry['force_forceplot'] = json.dumps(entry['force_data']['forceplot'])
                
                flattened_data.append(flat_entry)
            
            if flattened_data:
                # Get all possible fieldnames from all entries
                all_fieldnames = set()
                for entry in flattened_data:
                    all_fieldnames.update(entry.keys())
                fieldnames = sorted(list(all_fieldnames))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
    
    print(f"✅ Synchronized data saved:")
    print(f"   📋 JSON: {sync_json_path}")
    print(f"   📊 CSV: {sync_csv_path}")
    
    # Print synchronization statistics
    if synchronized_data:
        sync_diffs = [entry['time_sync_diff_s'] for entry in synchronized_data if entry['time_sync_diff_s'] is not None]
        if sync_diffs:
            print(f"\n📊 Synchronization Statistics:")
            print(f"   Average time difference: {sum(sync_diffs)/len(sync_diffs):.3f}s")
            print(f"   Max time difference: {max(sync_diffs):.3f}s")
            print(f"   Min time difference: {min(sync_diffs):.3f}s")
            
            # Count frames with force data
            frames_with_force = sum(1 for entry in synchronized_data if entry['force_data'] is not None)
            print(f"   Frames with force data: {frames_with_force}/{len(synchronized_data)} ({frames_with_force/len(synchronized_data)*100:.1f}%)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Synchronize force and pose data")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--force-csv", required=True, help="Path to force data CSV")
    parser.add_argument("--pose-json", required=True, help="Path to pose data JSON")
    parser.add_argument("--frames-csv", required=True, help="Path to frames CSV")
    parser.add_argument("--output-dir", default="synchronized_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    # Check input files
    for file_path in [args.video, args.force_csv, args.pose_json, args.frames_csv]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
    
    synchronize_data(args.video, args.force_csv, args.pose_json, args.frames_csv, args.output_dir)

if __name__ == "__main__":
    main()
