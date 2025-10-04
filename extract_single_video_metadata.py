#!/usr/bin/env python3
"""
Extract metadata from a single video file
"""

import subprocess
import json
import sys
import os
from datetime import datetime

def extract_video_metadata(video_path):
    """Extract metadata from video file using ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ffprobe: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing ffprobe output: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_single_video_metadata.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"🎥 Extracting metadata from: {video_path}")
    
    # Extract metadata
    metadata = extract_video_metadata(video_path)
    if metadata:
        print("📊 Video metadata:")
        
        # Format tags
        if 'format' in metadata and 'tags' in metadata['format']:
            print("   Format tags:")
            for key, value in metadata['format']['tags'].items():
                print(f"     {key}: {value}")
        
        # Stream info
        if 'streams' in metadata:
            for i, stream in enumerate(metadata['streams']):
                if stream.get('codec_type') == 'video':
                    print(f"   Video stream {i}:")
                    print(f"     Codec: {stream.get('codec_name', 'unknown')}")
                    print(f"     Resolution: {stream.get('width', 'unknown')}x{stream.get('height', 'unknown')}")
                    print(f"     Frame rate: {stream.get('r_frame_rate', 'unknown')}")
                    print(f"     Duration: {stream.get('duration', 'unknown')} seconds")
        
        # Format info
        if 'format' in metadata:
            print("   Format info:")
            print(f"     Duration: {metadata['format'].get('duration', 'unknown')} seconds")
            print(f"     Bit rate: {metadata['format'].get('bit_rate', 'unknown')} bps")
            print(f"     Size: {metadata['format'].get('size', 'unknown')} bytes")
        
        # Check for embedded timestamps
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            if 'creation_time' in tags:
                print(f"   Creation time: {tags['creation_time']}")
            if 'session_start' in tags:
                print(f"   Session start: {tags['session_start']}")
        
        return metadata
    else:
        print("❌ Failed to extract metadata")
        return None

if __name__ == "__main__":
    main()
