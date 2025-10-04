#!/usr/bin/env python3
"""
Test script to verify video metadata extraction works
"""

import sys
import os
import subprocess
import json

def test_ffmpeg_availability():
    """Test if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
        else:
            print("❌ FFmpeg is not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg is not installed")
        print("💡 Install FFmpeg: brew install ffmpeg")
        return False

def test_ffprobe_availability():
    """Test if FFprobe is available"""
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFprobe is available")
            return True
        else:
            print("❌ FFprobe is not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFprobe is not installed")
        print("💡 Install FFmpeg (includes FFprobe): brew install ffmpeg")
        return False

def test_video_metadata_extraction(video_path):
    """Test metadata extraction from a video file"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        print(f"📊 Video metadata for {os.path.basename(video_path)}:")
        if 'format' in metadata:
            format_info = metadata['format']
            print(f"   Duration: {format_info.get('duration', 'Unknown')} seconds")
            print(f"   Bitrate: {format_info.get('bit_rate', 'Unknown')} bps")
            print(f"   Format: {format_info.get('format_name', 'Unknown')}")
            
            if 'tags' in format_info:
                print("   Tags:")
                for key, value in format_info['tags'].items():
                    print(f"     {key}: {value}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting metadata: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing metadata JSON: {e}")
        return False

def main():
    print("🔍 Testing video metadata extraction capabilities")
    print("="*60)
    
    # Test FFmpeg availability
    ffmpeg_ok = test_ffmpeg_availability()
    ffprobe_ok = test_ffprobe_availability()
    
    if not (ffmpeg_ok and ffprobe_ok):
        print("\n❌ FFmpeg/FFprobe not available. Please install:")
        print("   brew install ffmpeg")
        sys.exit(1)
    
    # Test with existing video file if provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"\n🎥 Testing with video: {video_path}")
        test_video_metadata_extraction(video_path)
    else:
        print("\n💡 To test with a specific video file:")
        print("   python test_video_metadata.py <path_to_video.mp4>")
    
    print("\n✅ Video metadata extraction test complete!")

if __name__ == "__main__":
    main()
