#!/usr/bin/env python3
"""
Correlate Force Data with Body Pose Angles
Matches timestamps between force curves and body angles for comprehensive analysis
"""

import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os

def load_force_data():
    """Load the most recent force data"""
    # Find the most recent parsed force data
    force_files = glob.glob("pm5_py3row_parsed_*.csv")
    if not force_files:
        print("❌ No force data files found")
        return None
    
    latest_force_file = max(force_files, key=os.path.getctime)
    print(f"📊 Loading force data: {latest_force_file}")
    
    # Load CSV data
    df = pd.read_csv(latest_force_file)
    
    # Parse forceplot data
    force_data = []
    for idx, row in df.iterrows():
        if pd.notna(row['forceplot']) and row['forceplot'] != '':
            try:
                import json
                force_curve = json.loads(row['forceplot'])
                if force_curve:  # Only include non-empty force curves
                    force_data.append({
                        'timestamp': row['timestamp_iso'],
                        'elapsed_s': row['elapsed_s'],
                        'force_curve': force_curve,
                        'power': row['avg_power_w'],
                        'spm': row['spm'],
                        'distance': row['distance_m']
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    
    print(f"   Found {len(force_data)} complete force curves")
    return force_data

def load_pose_data():
    """Load the most recent pose data"""
    # Find the most recent pose data
    pose_files = glob.glob("rowing_pose_data_*.json")
    if not pose_files:
        print("❌ No pose data files found")
        return None
    
    latest_pose_file = max(pose_files, key=os.path.getctime)
    print(f"📊 Loading pose data: {latest_pose_file}")
    
    with open(latest_pose_file, 'r') as f:
        pose_data = json.load(f)
    
    print(f"   Found {len(pose_data)} pose frames")
    return pose_data

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    try:
        # Handle different timestamp formats
        if 'T' in timestamp_str:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            return datetime.fromisoformat(timestamp_str)
    except:
        return None

def find_closest_pose_frame(force_timestamp, pose_data, time_window=2.0):
    """Find the closest pose frame to a force timestamp within time window"""
    force_time = parse_timestamp(force_timestamp)
    if force_time is None:
        return None
    
    closest_frame = None
    min_time_diff = float('inf')
    
    for frame in pose_data:
        frame_time = parse_timestamp(frame['frame_time'])
        if frame_time is None:
            continue
        
        time_diff = abs((force_time - frame_time).total_seconds())
        
        if time_diff < time_window and time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_frame = frame
    
    return closest_frame, min_time_diff

def analyze_force_pose_correlation(force_data, pose_data):
    """Analyze correlation between force curves and body angles"""
    print("\n🔍 Analyzing Force-Pose Correlations...")
    
    correlations = []
    
    for i, force_entry in enumerate(force_data):
        print(f"   Processing force curve {i+1}/{len(force_data)}")
        
        # Find closest pose frame
        closest_pose, time_diff = find_closest_pose_frame(
            force_entry['timestamp'], pose_data
        )
        
        if closest_pose is None:
            continue
        
        # Extract force curve characteristics
        force_curve = force_entry['force_curve']
        if not force_curve:
            continue
        
        force_stats = {
            'peak_force': max(force_curve),
            'avg_force': np.mean(force_curve),
            'force_duration': len(force_curve),
            'force_curve': force_curve
        }
        
        # Extract pose characteristics
        pose_stats = {
            'left_arm_angle': closest_pose.get('left_arm_angle'),
            'right_arm_angle': closest_pose.get('right_arm_angle'),
            'left_leg_angle': closest_pose.get('left_leg_angle'),
            'right_leg_angle': closest_pose.get('right_leg_angle'),
            'torso_lean_angle': closest_pose.get('torso_lean_angle'),
            'frame_number': closest_pose['frame_number'],
            'pose_timestamp': closest_pose['frame_time']
        }
        
        # Combine data
        correlation_entry = {
            'force_timestamp': force_entry['timestamp'],
            'elapsed_s': force_entry['elapsed_s'],
            'time_diff': time_diff,
            'power': force_entry['power'],
            'spm': force_entry['spm'],
            'distance': force_entry['distance'],
            **force_stats,
            **pose_stats
        }
        
        correlations.append(correlation_entry)
    
    return correlations

def create_correlation_plots(correlations):
    """Create plots showing force-pose correlations"""
    if not correlations:
        print("❌ No correlations found to plot")
        return
    
    print(f"\n📊 Creating correlation plots for {len(correlations)} data points...")
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(correlations)
    
    # Create output directory
    os.makedirs('correlation_plots', exist_ok=True)
    
    # 1. Force vs Body Angles
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Force vs Body Angles Correlation', fontsize=16)
    
    angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle', 'torso_lean_angle']
    
    for i, angle in enumerate(angles):
        if angle in df.columns:
            row = i // 3
            col = i % 3
            
            # Filter out None values
            valid_data = df.dropna(subset=[angle, 'peak_force'])
            
            if len(valid_data) > 0:
                axes[row, col].scatter(valid_data[angle], valid_data['peak_force'], alpha=0.6)
                axes[row, col].set_xlabel(f'{angle} (degrees)')
                axes[row, col].set_ylabel('Peak Force')
                axes[row, col].set_title(f'Peak Force vs {angle}')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = valid_data[angle].corr(valid_data['peak_force'])
                axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', 
                                  transform=axes[row, col].transAxes, 
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplot
    if len(angles) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('correlation_plots/force_vs_angles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Force Curves with Body Angles Overlay
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Force Curves with Body Angles', fontsize=16)
    
    # Select a few representative force curves
    sample_indices = np.linspace(0, len(correlations)-1, 4, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        if idx < len(correlations):
            row = i // 2
            col = i % 2
            
            corr = correlations[idx]
            force_curve = corr['force_curve']
            
            if force_curve:
                x = np.arange(len(force_curve))
                axes[row, col].plot(x, force_curve, 'b-', linewidth=2, label='Force')
                axes[row, col].set_xlabel('Force Point')
                axes[row, col].set_ylabel('Force')
                axes[row, col].set_title(f'Drive {idx+1} - Peak: {corr["peak_force"]:.1f}')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add angle information as text
                angle_text = f"L Arm: {corr.get('left_arm_angle', 'N/A'):.1f}°\n"
                angle_text += f"R Arm: {corr.get('right_arm_angle', 'N/A'):.1f}°\n"
                angle_text += f"L Leg: {corr.get('left_leg_angle', 'N/A'):.1f}°\n"
                angle_text += f"R Leg: {corr.get('right_leg_angle', 'N/A'):.1f}°\n"
                angle_text += f"Torso: {corr.get('torso_lean_angle', 'N/A'):.1f}°"
                
                axes[row, col].text(0.02, 0.98, angle_text, 
                                  transform=axes[row, col].transAxes, 
                                  verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('correlation_plots/force_curves_with_angles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Time Series of Force and Angles
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Force and Body Angles Over Time', fontsize=16)
    
    # Plot peak force over time
    axes[0].plot(df['elapsed_s'], df['peak_force'], 'b-', linewidth=2, label='Peak Force')
    axes[0].set_ylabel('Peak Force')
    axes[0].set_title('Peak Force Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot arm angles over time
    if 'left_arm_angle' in df.columns and 'right_arm_angle' in df.columns:
        valid_arm_data = df.dropna(subset=['left_arm_angle', 'right_arm_angle'])
        if len(valid_arm_data) > 0:
            axes[1].plot(valid_arm_data['elapsed_s'], valid_arm_data['left_arm_angle'], 
                        'g-', linewidth=2, label='Left Arm')
            axes[1].plot(valid_arm_data['elapsed_s'], valid_arm_data['right_arm_angle'], 
                        'r-', linewidth=2, label='Right Arm')
            axes[1].set_ylabel('Arm Angle (degrees)')
            axes[1].set_title('Arm Angles Over Time')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
    
    # Plot leg angles over time
    if 'left_leg_angle' in df.columns and 'right_leg_angle' in df.columns:
        valid_leg_data = df.dropna(subset=['left_leg_angle', 'right_leg_angle'])
        if len(valid_leg_data) > 0:
            axes[2].plot(valid_leg_data['elapsed_s'], valid_leg_data['left_leg_angle'], 
                        'g-', linewidth=2, label='Left Leg')
            axes[2].plot(valid_leg_data['elapsed_s'], valid_leg_data['right_leg_angle'], 
                        'r-', linewidth=2, label='Right Leg')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_ylabel('Leg Angle (degrees)')
            axes[2].set_title('Leg Angles Over Time')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('correlation_plots/force_angles_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Correlation plots saved to 'correlation_plots/' directory")

def save_correlation_data(correlations):
    """Save correlation data to CSV"""
    if not correlations:
        print("❌ No correlations to save")
        return
    
    # Convert to DataFrame and save
    df = pd.DataFrame(correlations)
    
    # Flatten force curves for CSV (store as JSON string)
    df['force_curve_json'] = df['force_curve'].apply(lambda x: json.dumps(x) if x else '')
    df = df.drop('force_curve', axis=1)  # Remove the list column
    
    output_file = 'force_pose_correlations.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Correlation data saved to: {output_file}")
    print(f"   📊 {len(correlations)} correlations, {len(df.columns)} columns")

def main():
    """Main correlation analysis"""
    print("🚣‍♂️ Force-Pose Correlation Analysis")
    print("=" * 50)
    
    # Load data
    force_data = load_force_data()
    if force_data is None:
        return
    
    pose_data = load_pose_data()
    if pose_data is None:
        return
    
    # Analyze correlations
    correlations = analyze_force_pose_correlation(force_data, pose_data)
    
    if not correlations:
        print("❌ No correlations found between force and pose data")
        return
    
    print(f"\n✅ Found {len(correlations)} force-pose correlations")
    
    # Create visualizations
    create_correlation_plots(correlations)
    
    # Save data
    save_correlation_data(correlations)
    
    # Summary statistics
    print(f"\n📊 Correlation Summary:")
    df = pd.DataFrame(correlations)
    
    print(f"   Time differences: {df['time_diff'].mean():.2f}s ± {df['time_diff'].std():.2f}s")
    print(f"   Peak force range: {df['peak_force'].min():.1f} - {df['peak_force'].max():.1f}")
    
    # Show angle ranges
    angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle', 'torso_lean_angle']
    for angle in angles:
        if angle in df.columns:
            valid_data = df.dropna(subset=[angle])
            if len(valid_data) > 0:
                print(f"   {angle}: {valid_data[angle].min():.1f}° - {valid_data[angle].max():.1f}°")

if __name__ == "__main__":
    main()
