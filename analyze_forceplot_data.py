#!/usr/bin/env python3
"""
Analyze forceplot data from video capture sessions
"""

import sys
import os
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_forceplot_data(csv_path):
    """Load forceplot data from the PM5 CSV file"""
    print(f"📊 Loading forceplot data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter rows that have forceplot data
    has_current_fp = df['forceplot_current_json'].notna() & (df['forceplot_current_json'] != '')
    has_complete_fp = df['forceplot_complete_json'].notna() & (df['forceplot_complete_json'] != '')
    
    print(f"   Total rows: {len(df)}")
    print(f"   Rows with current forceplot: {has_current_fp.sum()}")
    print(f"   Rows with complete forceplot: {has_complete_fp.sum()}")
    
    return df, has_current_fp, has_complete_fp

def parse_forceplot_json(json_str):
    """Parse forceplot JSON string into list of values"""
    if not json_str or json_str == '':
        return []
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []

def analyze_stroke_data(df, has_complete_fp):
    """Analyze complete stroke forceplot data"""
    complete_strokes = df[has_complete_fp].copy()
    
    if len(complete_strokes) == 0:
        print("❌ No complete strokes found!")
        return None
    
    print(f"\n🏁 Found {len(complete_strokes)} complete strokes")
    
    stroke_data = []
    for idx, row in complete_strokes.iterrows():
        fp_data = parse_forceplot_json(row['forceplot_complete_json'])
        if fp_data:
            stroke_info = {
                'timestamp': row['ts_iso'],
                'elapsed_s': row['elapsed_s'],
                'power': row['power'],
                'spm': row['spm'],
                'distance': row['distance_m'],
                'forceplot': fp_data,
                'forceplot_length': len(fp_data),
                'max_force': max(fp_data) if fp_data else 0,
                'avg_force': np.mean(fp_data) if fp_data else 0,
                'stroke_duration': len(fp_data) * 0.05  # Assuming ~20Hz sampling
            }
            stroke_data.append(stroke_info)
    
    return stroke_data

def plot_force_curves(stroke_data, output_dir):
    """Plot individual force curves"""
    if not stroke_data:
        print("❌ No stroke data to plot")
        return
    
    print(f"\n📈 Creating force curve plots...")
    
    # Create subplots for multiple strokes
    n_strokes = min(len(stroke_data), 6)  # Show up to 6 strokes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(n_strokes):
        stroke = stroke_data[i]
        fp_data = stroke['forceplot']
        
        if fp_data:
            time_points = np.arange(len(fp_data)) * 0.05  # 20Hz sampling
            
            axes[i].plot(time_points, fp_data, 'b-', linewidth=2)
            axes[i].set_title(f"Stroke {i+1}\nPower: {stroke['power']}W, SPM: {stroke['spm']}")
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Force')
            axes[i].grid(True, alpha=0.3)
            
            # Add max force annotation
            max_idx = np.argmax(fp_data)
            axes[i].annotate(f'Max: {stroke["max_force"]:.1f}', 
                           xy=(time_points[max_idx], stroke["max_force"]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Hide unused subplots
    for i in range(n_strokes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'force_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path}")

def plot_stroke_summary(stroke_data, output_dir):
    """Plot summary statistics across strokes"""
    if not stroke_data:
        return
    
    print(f"📊 Creating stroke summary plots...")
    
    # Extract summary data
    stroke_nums = list(range(1, len(stroke_data) + 1))
    max_forces = [s['max_force'] for s in stroke_data]
    avg_forces = [s['avg_force'] for s in stroke_data]
    powers = [s['power'] for s in stroke_data]
    spms = [s['spm'] for s in stroke_data]
    durations = [s['stroke_duration'] for s in stroke_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Max force over time
    axes[0,0].plot(stroke_nums, max_forces, 'ro-', linewidth=2, markersize=6)
    axes[0,0].set_title('Max Force per Stroke')
    axes[0,0].set_xlabel('Stroke Number')
    axes[0,0].set_ylabel('Max Force')
    axes[0,0].grid(True, alpha=0.3)
    
    # Power over time
    axes[0,1].plot(stroke_nums, powers, 'go-', linewidth=2, markersize=6)
    axes[0,1].set_title('Power per Stroke')
    axes[0,1].set_xlabel('Stroke Number')
    axes[0,1].set_ylabel('Power (W)')
    axes[0,1].grid(True, alpha=0.3)
    
    # SPM over time
    axes[1,0].plot(stroke_nums, spms, 'bo-', linewidth=2, markersize=6)
    axes[1,0].set_title('Strokes per Minute')
    axes[1,0].set_xlabel('Stroke Number')
    axes[1,0].set_ylabel('SPM')
    axes[1,0].grid(True, alpha=0.3)
    
    # Stroke duration
    axes[1,1].plot(stroke_nums, durations, 'mo-', linewidth=2, markersize=6)
    axes[1,1].set_title('Stroke Duration')
    axes[1,1].set_xlabel('Stroke Number')
    axes[1,1].set_ylabel('Duration (s)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'stroke_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path}")

def create_analysis_report(stroke_data, output_dir):
    """Create a text report of the analysis"""
    if not stroke_data:
        return
    
    report_path = os.path.join(output_dir, 'forceplot_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("FORCEPLOT ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().isoformat()}\n")
        f.write(f"Total Strokes Analyzed: {len(stroke_data)}\n\n")
        
        if stroke_data:
            # Calculate statistics
            max_forces = [s['max_force'] for s in stroke_data]
            avg_forces = [s['avg_force'] for s in stroke_data]
            powers = [s['power'] for s in stroke_data]
            spms = [s['spm'] for s in stroke_data]
            durations = [s['stroke_duration'] for s in stroke_data]
            
            f.write("STROKE STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Max Force - Avg: {np.mean(max_forces):.1f}, Min: {np.min(max_forces):.1f}, Max: {np.max(max_forces):.1f}\n")
            f.write(f"Avg Force - Avg: {np.mean(avg_forces):.1f}, Min: {np.min(avg_forces):.1f}, Max: {np.max(avg_forces):.1f}\n")
            f.write(f"Power - Avg: {np.mean(powers):.1f}W, Min: {np.min(powers):.1f}W, Max: {np.max(powers):.1f}W\n")
            f.write(f"SPM - Avg: {np.mean(spms):.1f}, Min: {np.min(spms):.1f}, Max: {np.max(spms):.1f}\n")
            f.write(f"Stroke Duration - Avg: {np.mean(durations):.2f}s, Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s\n\n")
            
            f.write("INDIVIDUAL STROKE DETAILS:\n")
            f.write("-" * 30 + "\n")
            for i, stroke in enumerate(stroke_data, 1):
                f.write(f"Stroke {i}:\n")
                f.write(f"  Timestamp: {stroke['timestamp']}\n")
                f.write(f"  Power: {stroke['power']}W\n")
                f.write(f"  SPM: {stroke['spm']}\n")
                f.write(f"  Max Force: {stroke['max_force']:.1f}\n")
                f.write(f"  Avg Force: {stroke['avg_force']:.1f}\n")
                f.write(f"  Duration: {stroke['stroke_duration']:.2f}s\n")
                f.write(f"  Data Points: {stroke['forceplot_length']}\n\n")
    
    print(f"   Saved: {report_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_forceplot_data.py <session_directory>")
        print("Example: python analyze_forceplot_data.py sessions/rowcap_2025-10-04_11-20/")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    if not os.path.exists(session_dir):
        print(f"❌ Session directory not found: {session_dir}")
        sys.exit(1)
    
    # Find the PM5 CSV file
    pm5_files = [f for f in os.listdir(session_dir) if f.endswith('_pm5.csv')]
    if not pm5_files:
        print(f"❌ No PM5 CSV file found in {session_dir}")
        sys.exit(1)
    
    pm5_csv_path = os.path.join(session_dir, pm5_files[0])
    
    # Load and analyze data
    df, has_current_fp, has_complete_fp = load_forceplot_data(pm5_csv_path)
    stroke_data = analyze_stroke_data(df, has_complete_fp)
    
    if stroke_data:
        # Create output directory for analysis results
        analysis_dir = os.path.join(session_dir, 'forceplot_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Generate plots and report
        plot_force_curves(stroke_data, analysis_dir)
        plot_stroke_summary(stroke_data, analysis_dir)
        create_analysis_report(stroke_data, analysis_dir)
        
        print(f"\n✅ Analysis complete! Results saved to: {analysis_dir}")
    else:
        print("\n❌ No forceplot data found to analyze.")
        print("💡 Make sure you were actively rowing during the recording session.")

if __name__ == "__main__":
    main()
