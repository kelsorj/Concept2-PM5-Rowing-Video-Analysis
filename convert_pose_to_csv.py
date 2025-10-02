#!/usr/bin/env python3
"""
Convert pose JSON data to CSV format for analysis
"""

import json
import csv
import glob
import os

def convert_json_to_csv(json_file):
    """Convert pose JSON data to CSV format"""
    
    # Read JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("❌ No data found in JSON file")
        return
    
    # Create CSV filename
    csv_file = json_file.replace('.json', '.csv')
    
    # Get all fieldnames from first frame
    fieldnames = list(data[0].keys())
    
    # Write CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for frame_data in data:
            writer.writerow(frame_data)
    
    print(f"✅ Converted {json_file} to {csv_file}")
    print(f"   📊 {len(data)} frames, {len(fieldnames)} columns")
    
    return csv_file

def main():
    """Convert all pose JSON files to CSV"""
    print("🔄 Converting Pose Data to CSV")
    print("=" * 40)
    
    # Find all pose JSON files
    json_files = glob.glob("rowing_pose_data_*.json")
    
    if not json_files:
        print("❌ No pose JSON files found")
        return
    
    for json_file in json_files:
        print(f"\n📋 Processing: {json_file}")
        csv_file = convert_json_to_csv(json_file)
        
        if csv_file:
            # Show file size
            size = os.path.getsize(csv_file)
            print(f"   📁 File size: {size:,} bytes")

if __name__ == "__main__":
    main()
