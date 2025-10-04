#!/usr/bin/env python3
"""
Parse forceplot data from raw CSV and convert to format expected by advanced analysis
"""

import json
import csv
import pandas as pd
from datetime import datetime
import argparse
import os

def parse_raw_force_data(raw_csv_path, output_csv_path):
    """Parse forceplot data from raw CSV and create formatted output"""
    print(f"📊 Parsing force data from: {raw_csv_path}")
    
    force_data = []
    
    with open(raw_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse the raw JSON
                raw_json = json.loads(row['raw_json'])
                
                # Extract forceplot data
                forceplot = raw_json.get('forceplot', [])
                if forceplot:  # Only include rows with actual forceplot data
                    force_data.append({
                        'timestamp_ns': int(row['ts_ns']),
                        'timestamp_iso': row['ts_iso'],
                        'elapsed_s': raw_json.get('time', 0.0),
                        'distance_m': raw_json.get('distance', 0.0),
                        'spm': raw_json.get('spm', 0),
                        'power': raw_json.get('power', 0),
                        'pace': raw_json.get('pace', 0.0),
                        'calhr': raw_json.get('calhr', 0.0),
                        'calories': raw_json.get('calories', 0),
                        'heartrate': raw_json.get('heartrate', 0),
                        'forceplot': json.dumps(forceplot),  # Store as JSON string
                        'strokestate': raw_json.get('strokestate', ''),
                        'status': raw_json.get('status', ''),
                        'userid': raw_json.get('userid', ''),
                        'type': raw_json.get('type', 0),
                        'state': raw_json.get('state', 0),
                        'inttype': raw_json.get('inttype', 0),
                        'intcount': raw_json.get('intcount', 0)
                    })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"⚠️  Error parsing row: {e}")
                continue
    
    print(f"   Found {len(force_data)} rows with forceplot data")
    
    if force_data:
        # Write to CSV
        with open(output_csv_path, 'w', newline='') as f:
            if force_data:
                fieldnames = force_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(force_data)
        
        print(f"✅ Force data saved to: {output_csv_path}")
        
        # Show sample of parsed data
        print("\n📋 Sample of parsed force data:")
        for i, entry in enumerate(force_data[:3]):
            forceplot = json.loads(entry['forceplot'])
            print(f"   {i+1}. Time: {entry['elapsed_s']:.2f}s, Power: {entry['power']}W, "
                  f"SPM: {entry['spm']}, Forceplot: {len(forceplot)} points, "
                  f"Peak: {max(forceplot) if forceplot else 0}")
    else:
        print("❌ No forceplot data found")
    
    return len(force_data)

def main():
    parser = argparse.ArgumentParser(description="Parse forceplot data from raw CSV")
    parser.add_argument("--raw-csv", required=True, help="Path to raw CSV file")
    parser.add_argument("--output-csv", help="Path to output CSV file (default: parsed_force_data.csv)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.raw_csv):
        print(f"❌ Raw CSV file not found: {args.raw_csv}")
        return
    
    if args.output_csv is None:
        # Generate output filename based on input
        base_name = os.path.splitext(os.path.basename(args.raw_csv))[0]
        args.output_csv = f"{base_name}_parsed_force.csv"
    
    parse_raw_force_data(args.raw_csv, args.output_csv)

if __name__ == "__main__":
    main()
