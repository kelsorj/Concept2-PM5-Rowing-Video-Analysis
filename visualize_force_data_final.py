#!/usr/bin/env python3
"""
Final Corrected Force Data Visualization
Properly combines Drive + Dwelling measurements into complete stroke curves
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import argparse

def load_and_combine_force_data(raw_csv_path):
    """Load raw force data and combine Drive + Dwelling measurements into complete strokes"""
    print(f"📊 Loading and combining force data: {raw_csv_path}")
    
    all_data = []
    with open(raw_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                raw_json = json.loads(row['raw_json'])
                all_data.append({
                    'timestamp_ns': int(row['ts_ns']),
                    'timestamp_iso': row['ts_iso'],
                    'timestamp_dt': datetime.fromisoformat(row['ts_iso']),
                    'elapsed_s': raw_json.get('time', 0.0),
                    'distance_m': raw_json.get('distance', 0.0),
                    'spm': raw_json.get('spm', 0),
                    'power': raw_json.get('power', 0),
                    'pace': raw_json.get('pace', 0.0),
                    'forceplot': raw_json.get('forceplot', []),
                    'strokestate': raw_json.get('strokestate', ''),
                    'status': raw_json.get('status', '')
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                continue
    
    print(f"   Loaded {len(all_data)} total data points")
    
    # Now combine Drive + Dwelling measurements into complete strokes
    combined_strokes = []
    current_stroke = None
    
    for i, data_point in enumerate(all_data):
        if data_point['strokestate'] in ['Drive', 'Dwelling'] and data_point['forceplot']:
            if current_stroke is None:
                # Start a new stroke
                current_stroke = {
                    'start_timestamp_dt': data_point['timestamp_dt'],
                    'start_elapsed_s': data_point['elapsed_s'],
                    'combined_forceplot': data_point['forceplot'].copy(),
                    'measurements': [data_point],
                    'final_power': data_point['power'],
                    'final_spm': data_point['spm'],
                    'final_distance': data_point['distance_m'],
                    'stroke_phases': [data_point['strokestate']]
                }
            else:
                # Continue the current stroke - append forceplot data
                current_stroke['combined_forceplot'].extend(data_point['forceplot'])
                current_stroke['measurements'].append(data_point)
                current_stroke['final_power'] = data_point['power']
                current_stroke['final_spm'] = data_point['spm']
                current_stroke['final_distance'] = data_point['distance_m']
                current_stroke['stroke_phases'].append(data_point['strokestate'])
        
        elif current_stroke is not None and data_point['strokestate'] == 'Recovery':
            # End of current stroke - save it
            current_stroke['end_timestamp_dt'] = data_point['timestamp_dt']
            current_stroke['end_elapsed_s'] = data_point['elapsed_s']
            current_stroke['stroke_duration'] = current_stroke['end_elapsed_s'] - current_stroke['start_elapsed_s']
            combined_strokes.append(current_stroke)
            current_stroke = None
    
    # Don't forget the last stroke if it doesn't end with Recovery
    if current_stroke is not None:
        current_stroke['end_timestamp_dt'] = all_data[-1]['timestamp_dt']
        current_stroke['end_elapsed_s'] = all_data[-1]['elapsed_s']
        current_stroke['stroke_duration'] = current_stroke['end_elapsed_s'] - current_stroke['start_elapsed_s']
        combined_strokes.append(current_stroke)
    
    print(f"   Combined into {len(combined_strokes)} complete strokes")
    
    # Print stroke summary
    for i, stroke in enumerate(combined_strokes):
        phases = " -> ".join(stroke['stroke_phases'])
        print(f"   Stroke {i+1}: {len(stroke['combined_forceplot'])} force points, "
              f"{len(stroke['measurements'])} measurements, "
              f"{stroke['stroke_duration']:.2f}s duration, "
              f"Peak: {max(stroke['combined_forceplot'])}, "
              f"Phases: {phases}")
    
    return combined_strokes

def create_combined_force_plot(stroke, index, output_dir):
    """Create a force plot for a complete combined stroke"""
    force_curve = stroke['combined_forceplot']
    start_timestamp = stroke['start_timestamp_dt']
    end_timestamp = stroke['end_timestamp_dt']
    duration = stroke['stroke_duration']
    power = stroke['final_power']
    spm = stroke['final_spm']
    distance = stroke['final_distance']
    num_measurements = len(stroke['measurements'])
    phases = stroke['stroke_phases']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot force curve
    x = np.arange(len(force_curve))
    ax.plot(x, force_curve, 'b-', linewidth=2, marker='o', markersize=3, label='Complete Force Curve')
    
    # Highlight peak
    peak_idx = np.argmax(force_curve)
    peak_force = force_curve[peak_idx]
    ax.plot(peak_idx, peak_force, 'ro', markersize=10, label=f'Peak: {peak_force}')
    
    # Mark measurement boundaries and phases
    measurement_boundaries = []
    cumulative_length = 0
    phase_colors = {'Drive': 'green', 'Dwelling': 'orange'}
    
    for i, measurement in enumerate(stroke['measurements']):
        measurement_boundaries.append(cumulative_length)
        cumulative_length += len(measurement['forceplot'])
        
        # Color code by phase
        phase = measurement['strokestate']
        color = phase_colors.get(phase, 'blue')
        
        # Add phase label
        if i == 0:
            ax.text(cumulative_length/2, max(force_curve) * 0.9, phase, 
                   ha='center', fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Add phase boundary lines
    for i, boundary in enumerate(measurement_boundaries[1:], 1):
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(boundary, max(force_curve) * 0.8, f'M{i}', rotation=90, 
                verticalalignment='top', fontsize=8, color='red', fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Force Data Points', fontsize=12)
    ax.set_ylabel('Force', fontsize=12)
    ax.set_title(f'Complete Stroke #{index+1} - {start_timestamp.strftime("%H:%M:%S.%f")[:-3]}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add detailed information
    phases_str = " -> ".join(phases)
    info_text = f"""Start: {start_timestamp.strftime("%H:%M:%S.%f")[:-3]}
End: {end_timestamp.strftime("%H:%M:%S.%f")[:-3]}
Duration: {duration:.2f}s
Power: {power}W
SPM: {spm}
Distance: {distance:.1f}m
Phases: {phases_str}
Measurements: {num_measurements}
Total Points: {len(force_curve)}
Peak Force: {peak_force}
Avg Force: {np.mean(force_curve):.1f}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = f"complete_stroke_{index+1:02d}_{start_timestamp.strftime('%H%M%S_%f')[:-3]}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_stroke_comparison_plot(combined_strokes, output_dir):
    """Create a comparison plot showing all complete strokes"""
    print("📊 Creating stroke comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: All strokes normalized and overlaid
    colors = plt.cm.tab10(np.linspace(0, 1, len(combined_strokes)))
    
    for i, stroke in enumerate(combined_strokes):
        force_curve = stroke['combined_forceplot']
        if force_curve:
            # Normalize to 0-1 for comparison
            normalized_force = np.array(force_curve) / max(force_curve)
            x_positions = np.linspace(0, 1, len(normalized_force))
            
            ax1.plot(x_positions, normalized_force, color=colors[i], linewidth=2, 
                    label=f'Stroke {i+1} (Peak: {max(force_curve)})', alpha=0.8)
    
    ax1.set_xlabel('Normalized Stroke Position', fontsize=12)
    ax1.set_ylabel('Normalized Force', fontsize=12)
    ax1.set_title('All Complete Strokes (Normalized for Comparison)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Stroke statistics
    stroke_numbers = list(range(1, len(combined_strokes) + 1))
    peak_forces = [max(stroke['combined_forceplot']) for stroke in combined_strokes]
    avg_forces = [np.mean(stroke['combined_forceplot']) for stroke in combined_strokes]
    durations = [stroke['stroke_duration'] for stroke in combined_strokes]
    powers = [stroke['final_power'] for stroke in combined_strokes]
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar([x - 0.2 for x in stroke_numbers], peak_forces, 0.4, 
                   label='Peak Force', color='red', alpha=0.7)
    bars2 = ax2.bar([x + 0.2 for x in stroke_numbers], avg_forces, 0.4, 
                   label='Avg Force', color='blue', alpha=0.7)
    line = ax2_twin.plot(stroke_numbers, powers, 'go-', linewidth=2, markersize=8, 
                        label='Power')
    
    ax2.set_xlabel('Stroke Number', fontsize=12)
    ax2.set_ylabel('Force', fontsize=12, color='black')
    ax2_twin.set_ylabel('Power (W)', fontsize=12, color='green')
    ax2.set_title('Stroke Statistics', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(output_dir, "complete_strokes_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return comparison_path

def create_phase_analysis_plot(combined_strokes, output_dir):
    """Create a plot showing the phase breakdown of each stroke"""
    print("📊 Creating phase analysis plot...")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    y_pos = 0
    phase_colors = {'Drive': 'green', 'Dwelling': 'orange'}
    
    for i, stroke in enumerate(combined_strokes):
        stroke_y = y_pos
        
        # Plot each measurement in the stroke with phase coloring
        x_start = 0
        for j, measurement in enumerate(stroke['measurements']):
            force_curve = measurement['forceplot']
            if force_curve:
                x_end = x_start + len(force_curve)
                x_positions = np.arange(x_start, x_end)
                
                phase = measurement['strokestate']
                color = phase_colors.get(phase, 'blue')
                
                ax.plot(x_positions, force_curve, color=color, linewidth=3, 
                       label=f'{phase} Phase' if i == 0 and j == 0 else "")
                
                # Add phase boundary
                if j < len(stroke['measurements']) - 1:
                    ax.axvline(x=x_end, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add stroke label
        max_force = max([max(m['forceplot']) for m in stroke['measurements'] if m['forceplot']])
        ax.text(-5, stroke_y + max_force/2, f'Stroke {i+1}', rotation=90, 
                verticalalignment='center', fontsize=10, fontweight='bold')
        
        y_pos += 200  # Space between strokes
    
    ax.set_xlabel('Combined Force Data Points', fontsize=12)
    ax.set_ylabel('Force', fontsize=12)
    ax.set_title('Complete Strokes with Drive/Dwelling Phase Breakdown', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save phase analysis plot
    phase_path = os.path.join(output_dir, "phase_analysis.png")
    plt.savefig(phase_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return phase_path

def main():
    parser = argparse.ArgumentParser(description="Visualize force data with proper Drive+Dwelling combination")
    parser.add_argument("--raw-csv", required=True, help="Path to raw CSV file")
    parser.add_argument("--output-dir", default="final_force_visualization", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.raw_csv):
        print(f"❌ Raw CSV file not found: {args.raw_csv}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and combine force data
    combined_strokes = load_and_combine_force_data(args.raw_csv)
    
    if not combined_strokes:
        print("❌ No complete strokes found")
        return
    
    print(f"\n🎨 Creating final force visualizations...")
    
    # Create individual stroke plots
    print(f"📊 Creating {len(combined_strokes)} complete stroke plots...")
    plot_files = []
    for i, stroke in enumerate(combined_strokes):
        plot_file = create_combined_force_plot(stroke, i, args.output_dir)
        plot_files.append(plot_file)
        print(f"   Created stroke {i+1}/{len(combined_strokes)}")
    
    # Create comparison plots
    comparison_path = create_stroke_comparison_plot(combined_strokes, args.output_dir)
    phase_path = create_phase_analysis_plot(combined_strokes, args.output_dir)
    
    # Create HTML viewer
    html_path = os.path.join(args.output_dir, "view_final_plots.html")
    with open(html_path, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Final Force Data Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot {{ margin: 20px 0; border: 1px solid #ccc; padding: 10px; }}
        .plot img {{ max-width: 100%; height: auto; }}
        .summary {{ background-color: #e8f4fd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .stroke-info {{ background-color: #f0f8f0; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .phase-info {{ background-color: #fff3cd; padding: 8px; margin: 5px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>Final Force Data Visualization</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Complete Strokes:</strong> {len(combined_strokes)}</p>
        <p><strong>Time Range:</strong> {combined_strokes[0]['start_timestamp_dt'].strftime('%H:%M:%S')} - {combined_strokes[-1]['end_timestamp_dt'].strftime('%H:%M:%S')}</p>
        <p><strong>Total Duration:</strong> {combined_strokes[-1]['end_elapsed_s'] - combined_strokes[0]['start_elapsed_s']:.2f} seconds</p>
        <p><strong>Average Stroke Duration:</strong> {np.mean([s['stroke_duration'] for s in combined_strokes]):.2f} seconds</p>
    </div>
    
    <div class="plot">
        <h2>Complete Strokes Comparison</h2>
        <img src="complete_strokes_comparison.png" alt="Complete Strokes Comparison">
    </div>
    
    <div class="plot">
        <h2>Phase Analysis (Drive + Dwelling)</h2>
        <img src="phase_analysis.png" alt="Phase Analysis">
    </div>
    
    <h2>Individual Complete Strokes</h2>
""")
        
        for i, stroke in enumerate(combined_strokes):
            filename = os.path.basename(plot_files[i])
            phases_str = " → ".join(stroke['stroke_phases'])
            f.write(f"""
    <div class="stroke-info">
        <h3>Complete Stroke #{i+1}</h3>
        <p><strong>Duration:</strong> {stroke['stroke_duration']:.2f}s | 
           <strong>Peak Force:</strong> {max(stroke['combined_forceplot'])} | 
           <strong>Measurements:</strong> {len(stroke['measurements'])} | 
           <strong>Data Points:</strong> {len(stroke['combined_forceplot'])}</p>
        <div class="phase-info">
            <strong>Phases:</strong> {phases_str}
        </div>
    </div>
    <div class="plot">
        <img src="{filename}" alt="Complete Stroke {i+1}">
    </div>
""")
        
        f.write("""
</body>
</html>
""")
    
    print(f"\n🎉 Final force visualization complete!")
    print(f"   📁 Output directory: {args.output_dir}")
    print(f"   📊 Complete strokes: {len(combined_strokes)}")
    print(f"   📈 Comparison plot: {comparison_path}")
    print(f"   🔍 Phase analysis: {phase_path}")
    print(f"   🌐 HTML viewer: {html_path}")
    print(f"\n💡 Open {html_path} in your browser to view the final corrected plots!")

if __name__ == "__main__":
    main()
