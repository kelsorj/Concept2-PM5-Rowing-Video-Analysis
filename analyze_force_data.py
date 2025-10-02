#!/usr/bin/env python3
"""
Force Data Analysis and Plotting Script
Analyzes Concept2 PM5 force data and creates individual plots for each drive
"""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import glob
from pathlib import Path

def load_raw_data(filename):
    """Load raw CSV data and parse JSON force data"""
    data = []
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse the JSON data
                raw_data = json.loads(row['raw_data'])
                
                # Extract relevant fields
                entry = {
                    'timestamp': row['timestamp_iso'],
                    'time': raw_data.get('time', 0),
                    'distance': raw_data.get('distance', 0),
                    'spm': raw_data.get('spm', 0),
                    'power': raw_data.get('power', 0),
                    'forceplot': raw_data.get('forceplot', []),
                    'strokestate': raw_data.get('strokestate', ''),
                    'status': raw_data.get('status', '')
                }
                data.append(entry)
            except (json.JSONDecodeError, KeyError) as e:
                # Skip non-JSON rows (like device detection messages)
                continue
    
    return data

def extract_drives(data):
    """Extract individual drive sequences from the data"""
    drives = []
    current_drive = []
    in_drive = False
    
    for entry in data:
        strokestate = entry['strokestate']
        forceplot = entry['forceplot']
        
        # Start of a drive
        if strokestate == 'Drive' and forceplot:
            if not in_drive:
                in_drive = True
                current_drive = []
            current_drive.append(entry)
        
        # End of a drive (transition to Recovery or other state)
        elif in_drive and strokestate != 'Drive':
            if current_drive:
                drives.append(current_drive)
                current_drive = []
            in_drive = False
    
    # Handle case where data ends during a drive
    if current_drive:
        drives.append(current_drive)
    
    return drives

def aggregate_force_data(drive_entries):
    """Aggregate force data across all entries in a drive"""
    all_force_data = []
    timestamps = []
    
    for entry in drive_entries:
        if entry['forceplot']:
            all_force_data.extend(entry['forceplot'])
            # Create relative timestamps for the force data points
            base_time = entry['time']
            for i in range(len(entry['forceplot'])):
                timestamps.append(base_time + (i * 0.1))  # Assume ~100ms between points
    
    return all_force_data, timestamps

def create_force_plot(drive_data, drive_number, output_dir):
    """Create a force vs time plot for a single drive"""
    force_data, timestamps = aggregate_force_data(drive_data)
    
    if not force_data:
        print(f"⚠️  Drive {drive_number}: No force data to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot force over time
    plt.plot(timestamps, force_data, 'b-', linewidth=2, label='Force')
    
    # Add markers for key points
    if force_data:
        max_force = max(force_data)
        max_idx = force_data.index(max_force)
        plt.plot(timestamps[max_idx], max_force, 'ro', markersize=8, label=f'Peak Force: {max_force}')
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Force', fontsize=12)
    plt.title(f'Drive #{drive_number} - Force Curve\n'
              f'Duration: {timestamps[-1] - timestamps[0]:.1f}s, '
              f'Peak: {max(force_data)}, '
              f'Points: {len(force_data)}', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics as text
    stats_text = f'Avg Force: {np.mean(force_data):.1f}\n'
    stats_text += f'Min Force: {min(force_data):.1f}\n'
    stats_text += f'Max Force: {max(force_data):.1f}\n'
    stats_text += f'Total Points: {len(force_data)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    filename = f"drive_{drive_number:03d}_force_curve.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Drive {drive_number}: Saved {filename} ({len(force_data)} force points)")

def create_summary_plot(all_drives, output_dir):
    """Create a summary plot showing all drives overlaid"""
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_drives)))
    
    for i, drive_data in enumerate(all_drives):
        force_data, timestamps = aggregate_force_data(drive_data)
        
        if force_data and timestamps:
            # Normalize timestamps to start from 0 for each drive
            normalized_times = [t - timestamps[0] for t in timestamps]
            plt.plot(normalized_times, force_data, color=colors[i], 
                    linewidth=1.5, alpha=0.7, label=f'Drive {i+1}')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Force', fontsize=12)
    plt.title(f'All Drive Force Curves Overlay\n{len(all_drives)} drives total', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the summary plot
    filepath = os.path.join(output_dir, "all_drives_summary.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary: Saved all_drives_summary.png ({len(all_drives)} drives)")

def create_statistics_plot(all_drives, output_dir):
    """Create a plot showing drive statistics"""
    drive_stats = []
    
    for i, drive_data in enumerate(all_drives):
        force_data, timestamps = aggregate_force_data(drive_data)
        
        if force_data:
            stats = {
                'drive': i + 1,
                'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
                'peak_force': max(force_data),
                'avg_force': np.mean(force_data),
                'min_force': min(force_data),
                'force_points': len(force_data)
            }
            drive_stats.append(stats)
    
    if not drive_stats:
        print("⚠️  No drive statistics to plot")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    drives = [s['drive'] for s in drive_stats]
    
    # Peak force
    ax1.bar(drives, [s['peak_force'] for s in drive_stats], color='red', alpha=0.7)
    ax1.set_xlabel('Drive Number')
    ax1.set_ylabel('Peak Force')
    ax1.set_title('Peak Force per Drive')
    ax1.grid(True, alpha=0.3)
    
    # Average force
    ax2.bar(drives, [s['avg_force'] for s in drive_stats], color='blue', alpha=0.7)
    ax2.set_xlabel('Drive Number')
    ax2.set_ylabel('Average Force')
    ax2.set_title('Average Force per Drive')
    ax2.grid(True, alpha=0.3)
    
    # Duration
    ax3.bar(drives, [s['duration'] for s in drive_stats], color='green', alpha=0.7)
    ax3.set_xlabel('Drive Number')
    ax3.set_ylabel('Duration (seconds)')
    ax3.set_title('Drive Duration')
    ax3.grid(True, alpha=0.3)
    
    # Force points
    ax4.bar(drives, [s['force_points'] for s in drive_stats], color='orange', alpha=0.7)
    ax4.set_xlabel('Drive Number')
    ax4.set_ylabel('Number of Force Points')
    ax4.set_title('Force Data Points per Drive')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Drive Statistics Summary', fontsize=16)
    plt.tight_layout()
    
    # Save the statistics plot
    filepath = os.path.join(output_dir, "drive_statistics.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Statistics: Saved drive_statistics.png")

def main():
    """Main analysis function"""
    print("🚣‍♂️ Concept2 PM5 Force Data Analysis")
    print("=" * 50)
    
    # Find the most recent raw data file
    raw_files = glob.glob("pm5_py3row_raw_*.csv")
    if not raw_files:
        print("❌ No raw data files found (pm5_py3row_raw_*.csv)")
        return
    
    # Use the most recent file
    latest_file = max(raw_files, key=os.path.getctime)
    print(f"📊 Analyzing: {latest_file}")
    
    # Create output directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    print("📈 Loading data...")
    data = load_raw_data(latest_file)
    print(f"   Loaded {len(data)} data points")
    
    # Extract drives
    print("🔍 Extracting drives...")
    drives = extract_drives(data)
    print(f"   Found {len(drives)} complete drives")
    
    if not drives:
        print("❌ No drives found in the data")
        return
    
    # Create individual drive plots
    print("📊 Creating individual drive plots...")
    for i, drive_data in enumerate(drives, 1):
        create_force_plot(drive_data, i, output_dir)
    
    # Create summary plots
    print("📊 Creating summary plots...")
    create_summary_plot(drives, output_dir)
    create_statistics_plot(drives, output_dir)
    
    print(f"\n🎉 Analysis complete!")
    print(f"   📁 Plots saved to: {output_dir}/")
    print(f"   📊 Individual drives: {len(drives)} plots")
    print(f"   📈 Summary plots: 2 additional plots")

if __name__ == "__main__":
    main()
