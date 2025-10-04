#!/usr/bin/env python3
"""
Extract frame timestamps and metadata from video files for synchronization
"""

import sys
import os
import json
import subprocess
import csv
from datetime import datetime

def extract_video_metadata(video_path):
    """Extract metadata from video file using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting metadata: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing metadata JSON: {e}")
        return None

def extract_frame_timestamps(video_path, output_csv=None):
    """Extract frame timestamps using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time,pkt_dts_time,pkt_duration_time',
        '-of', 'csv=p=0',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        frame_data = []
        
        for line_num, line in enumerate(result.stdout.strip().split('\n')):
            if line:
                parts = line.split(',')
                if len(parts) >= 3:
                    frame_data.append({
                        'frame_idx': line_num,
                        'pkt_pts_time': float(parts[0]) if parts[0] else None,
                        'pkt_dts_time': float(parts[1]) if parts[1] else None,
                        'pkt_duration_time': float(parts[2]) if parts[2] else None,
                    })
        
        if output_csv:
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['frame_idx', 'pkt_pts_time', 'pkt_dts_time', 'pkt_duration_time'])
                writer.writeheader()
                writer.writerows(frame_data)
            print(f"📊 Frame timestamps saved to: {output_csv}")
        
        return frame_data
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting frame timestamps: {e}")
        return None

def create_synchronization_map(video_path, frames_csv_path, output_path):
    """Create a synchronization map between video frames and PM5 data"""
    
    # Extract video metadata
    metadata = extract_video_metadata(video_path)
    if not metadata:
        return None
    
    # Extract frame timestamps
    frame_timestamps = extract_frame_timestamps(video_path)
    if not frame_timestamps:
        return None
    
    # Read frames CSV
    try:
        with open(frames_csv_path, 'r') as f:
            frames_data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"❌ Frames CSV not found: {frames_csv_path}")
        return None
    
    # Get session start time from video metadata
    session_start = None
    if 'format' in metadata and 'tags' in metadata['format']:
        tags = metadata['format']['tags']
        if 'session_start' in tags:
            session_start = datetime.fromisoformat(tags['session_start'].replace('Z', '+00:00'))
        elif 'creation_time' in tags:
            session_start = datetime.fromisoformat(tags['creation_time'].replace('Z', '+00:00'))
    
    if not session_start:
        print("⚠️  Could not determine session start time from video metadata")
        return None
    
    # Create synchronization map
    sync_map = []
    
    for i, frame_ts in enumerate(frame_timestamps):
        if i < len(frames_data):
            frame_data = frames_data[i]
            
            # Calculate absolute timestamp for this frame
            if frame_ts['pkt_pts_time'] is not None:
                frame_absolute_time = session_start.timestamp() + frame_ts['pkt_pts_time']
                
                sync_entry = {
                    'frame_idx': i,
                    'video_timestamp': frame_ts['pkt_pts_time'],
                    'absolute_timestamp': frame_absolute_time,
                    'iso_timestamp': datetime.fromtimestamp(frame_absolute_time).isoformat(),
                    'frame_csv_ts_ns': frame_data.get('ts_ns'),
                    'frame_csv_ts_iso': frame_data.get('ts_iso'),
                }
                sync_map.append(sync_entry)
    
    # Save synchronization map
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame_idx', 'video_timestamp', 'absolute_timestamp', 'iso_timestamp',
            'frame_csv_ts_ns', 'frame_csv_ts_iso'
        ])
        writer.writeheader()
        writer.writerows(sync_map)
    
    print(f"📊 Synchronization map saved to: {output_path}")
    return sync_map

def correlate_with_pm5_data(sync_map_path, pm5_csv_path, output_path):
    """Correlate video frames with PM5 data"""
    
    # Read synchronization map
    try:
        with open(sync_map_path, 'r') as f:
            sync_data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"❌ Sync map not found: {sync_map_path}")
        return None
    
    # Read PM5 data
    try:
        with open(pm5_csv_path, 'r') as f:
            pm5_data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"❌ PM5 CSV not found: {pm5_csv_path}")
        return None
    
    # Create correlation map
    correlation_map = []
    
    for frame in sync_data:
        frame_ts_ns = int(frame['frame_csv_ts_ns']) if frame['frame_csv_ts_ns'] else None
        
        if frame_ts_ns is not None:
            # Find closest PM5 data point
            closest_pm5 = None
            min_diff = float('inf')
            
            for pm5_point in pm5_data:
                pm5_ts_ns = int(pm5_point['ts_ns']) if pm5_point['ts_ns'] else None
                if pm5_ts_ns is not None:
                    diff = abs(frame_ts_ns - pm5_ts_ns)
                    if diff < min_diff:
                        min_diff = diff
                        closest_pm5 = pm5_point
            
            if closest_pm5:
                correlation_entry = {
                    'frame_idx': frame['frame_idx'],
                    'video_timestamp': frame['video_timestamp'],
                    'frame_ts_ns': frame_ts_ns,
                    'pm5_ts_ns': closest_pm5['ts_ns'],
                    'time_diff_ns': min_diff,
                    'time_diff_ms': min_diff / 1_000_000,
                    'power': closest_pm5.get('power', ''),
                    'spm': closest_pm5.get('spm', ''),
                    'strokestate': closest_pm5.get('strokestate', ''),
                    'forceplot_current_json': closest_pm5.get('forceplot_current_json', ''),
                    'forceplot_complete_json': closest_pm5.get('forceplot_complete_json', ''),
                }
                correlation_map.append(correlation_entry)
    
    # Save correlation map
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame_idx', 'video_timestamp', 'frame_ts_ns', 'pm5_ts_ns',
            'time_diff_ns', 'time_diff_ms', 'power', 'spm', 'strokestate',
            'forceplot_current_json', 'forceplot_complete_json'
        ])
        writer.writeheader()
        writer.writerows(correlation_map)
    
    print(f"📊 Correlation map saved to: {output_path}")
    return correlation_map

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_video_metadata.py <session_directory>")
        print("Example: python extract_video_metadata.py sessions/py3rowcap_20251004_121815/")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    if not os.path.exists(session_dir):
        print(f"❌ Session directory not found: {session_dir}")
        sys.exit(1)
    
    # Find video and CSV files
    video_files = [f for f in os.listdir(session_dir) if f.endswith('.mp4')]
    frames_csv_files = [f for f in os.listdir(session_dir) if f.endswith('_frames.csv')]
    pm5_csv_files = [f for f in os.listdir(session_dir) if f.endswith('_pm5.csv')]
    
    if not video_files:
        print(f"❌ No video file found in {session_dir}")
        sys.exit(1)
    
    video_path = os.path.join(session_dir, video_files[0])
    frames_csv_path = os.path.join(session_dir, frames_csv_files[0]) if frames_csv_files else None
    pm5_csv_path = os.path.join(session_dir, pm5_csv_files[0]) if pm5_csv_files else None
    
    print(f"🎥 Processing video: {video_path}")
    
    # Extract metadata
    metadata = extract_video_metadata(video_path)
    if metadata:
        print("📊 Video metadata:")
        if 'format' in metadata and 'tags' in metadata['format']:
            for key, value in metadata['format']['tags'].items():
                print(f"   {key}: {value}")
    
    # Extract frame timestamps
    frame_timestamps_csv = os.path.join(session_dir, 'frame_timestamps.csv')
    frame_timestamps = extract_frame_timestamps(video_path, frame_timestamps_csv)
    
    if frames_csv_path and frame_timestamps:
        # Create synchronization map
        sync_map_path = os.path.join(session_dir, 'video_frame_sync_map.csv')
        sync_map = create_synchronization_map(video_path, frames_csv_path, sync_map_path)
        
        if pm5_csv_path and sync_map:
            # Create correlation map
            correlation_path = os.path.join(session_dir, 'video_pm5_correlation.csv')
            correlation_map = correlate_with_pm5_data(sync_map_path, pm5_csv_path, correlation_path)
            
            if correlation_map:
                print(f"\n✅ Synchronization complete!")
                print(f"   Video frames: {len(frame_timestamps)}")
                print(f"   PM5 data points: {len(correlation_map)}")
                print(f"   Average time difference: {sum(float(c['time_diff_ms']) for c in correlation_map) / len(correlation_map):.2f}ms")
    
    print(f"\n🎯 Metadata extraction complete!")

if __name__ == "__main__":
    main()
