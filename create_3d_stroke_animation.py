#!/usr/bin/env python3
"""
3D Animated Stroke Visualization
Creates an interactive 3D animation showing stroke biomechanics in angle-space
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
import glob

def load_stroke_data(analysis_dir):
    """Load pose data and combined strokes for animation"""
    print(f"📊 Loading data from: {analysis_dir}")
    
    # Load pose frames
    pose_json_files = sorted(glob.glob(os.path.join(analysis_dir, "pose_data_*.json")))
    if not pose_json_files:
        print("❌ No pose data found")
        return None
    
    with open(pose_json_files[-1], 'r') as f:
        pose_frames = json.load(f)
    print(f"   Loaded {len(pose_frames)} pose frames")
    
    # Try to load frame timestamps from various sources
    frame_timestamps = []
    
    # First try: look for frames CSV
    frames_csv = glob.glob(os.path.join(analysis_dir, "*_frames.csv"))
    if frames_csv:
        with open(frames_csv[0], 'r') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_timestamps.append({
                        'frame_idx': int(row.get('frame_idx') or row.get('frame_number') or 0),
                        'timestamp': pd.to_datetime(row['ts_iso'])
                    })
                except:
                    continue
        print(f"   Loaded {len(frame_timestamps)} frame timestamps from frames CSV")
    
    # Second try: use raw CSV data to estimate frame timestamps
    if not frame_timestamps:
        raw_csv_files = glob.glob(os.path.join(analysis_dir, "*_raw.csv"))
        if raw_csv_files:
            print("   No frames CSV found, estimating timestamps from raw data...")
            # Read raw data to get timestamp range
            raw_data = pd.read_csv(raw_csv_files[0])
            if 'ts_iso' in raw_data.columns:
                start_time = pd.to_datetime(raw_data['ts_iso'].iloc[0])
                end_time = pd.to_datetime(raw_data['ts_iso'].iloc[-1])
                
                # Estimate frame timestamps based on pose frame count
                num_frames = len(pose_frames)
                time_range = (end_time - start_time).total_seconds()
                frame_interval = time_range / num_frames if num_frames > 0 else 0.033  # Default 30fps
                
                frame_timestamps = []
                for i in range(num_frames):
                    frame_time = start_time + pd.Timedelta(seconds=i * frame_interval)
                    frame_timestamps.append({
                        'frame_idx': i,
                        'timestamp': frame_time
                    })
                print(f"   Generated {len(frame_timestamps)} estimated frame timestamps")
    
    if not frame_timestamps:
        print("❌ No frame timestamps found or generated")
        return None
    
    # Load combined strokes for force data
    combined_json = os.path.join(analysis_dir, 'pm5_combined_strokes.json')
    combined_strokes = None
    if os.path.exists(combined_json):
        with open(combined_json, 'r') as f:
            data = json.load(f)
        combined_strokes = []
        for s in data:
            combined_strokes.append({
                'start_timestamp': pd.to_datetime(s['start_timestamp_iso']),
                'end_timestamp': pd.to_datetime(s['end_timestamp_iso']),
                'combined_forceplot': s.get('combined_forceplot', [])
            })
        print(f"   Loaded {len(combined_strokes)} strokes")
    
    return {
        'pose_frames': pose_frames,
        'frame_timestamps': frame_timestamps,
        'combined_strokes': combined_strokes
    }

def create_3d_stroke_animation(data, stroke_number, output_path):
    """Create an animated 3D visualization of a single stroke"""
    print(f"🎬 Creating 3D animation for Stroke #{stroke_number}")
    
    pose_frames = data['pose_frames']
    frame_timestamps = data['frame_timestamps']
    combined_strokes = data['combined_strokes']
    
    if not combined_strokes or stroke_number > len(combined_strokes):
        print(f"❌ Stroke {stroke_number} not found")
        return None
    
    stroke = combined_strokes[stroke_number - 1]
    
    # Get frames for this stroke
    stroke_data = []
    for i, ft in enumerate(frame_timestamps):
        if stroke['start_timestamp'] <= ft['timestamp'] <= stroke['end_timestamp']:
            if i < len(pose_frames):
                frame = pose_frames[i]
                
                # Calculate average angles
                leg_angle = np.mean([v for v in [frame.get('left_leg_angle'), frame.get('right_leg_angle')] if v is not None])
                back_angle = frame.get('back_vertical_angle')
                arm_angle = np.mean([v for v in [frame.get('left_arm_angle'), frame.get('right_arm_angle')] if v is not None])
                
                if not np.isnan(leg_angle) and back_angle is not None and not np.isnan(arm_angle):
                    stroke_data.append({
                        'time': (ft['timestamp'] - stroke['start_timestamp']).total_seconds(),
                        'leg_angle': leg_angle,
                        'back_angle': back_angle,
                        'arm_angle': arm_angle,
                        'frame_idx': ft['frame_idx']
                    })
    
    if len(stroke_data) == 0:
        print(f"❌ No valid data for stroke {stroke_number}")
        return None
    
    # Map force data to frames
    force_curve = stroke['combined_forceplot']
    if force_curve:
        max_force = max(force_curve)
        for i, frame_data in enumerate(stroke_data):
            # Map frame time to force curve index
            time_ratio = frame_data['time'] / stroke_data[-1]['time']
            force_idx = int(time_ratio * (len(force_curve) - 1))
            force_idx = min(force_idx, len(force_curve) - 1)
            frame_data['force'] = force_curve[force_idx] / max_force if max_force > 0 else 0
    else:
        for frame_data in stroke_data:
            frame_data['force'] = 0
    
    # Extract all data for plotting
    leg_angles = [d['leg_angle'] for d in stroke_data]
    back_angles = [d['back_angle'] for d in stroke_data]
    arm_angles = [d['arm_angle'] for d in stroke_data]
    forces = [d['force'] for d in stroke_data]
    times = [d['time'] for d in stroke_data]
    
    # Pre-calculate global min/max for consistent axis ranges
    leg_min, leg_max = min(leg_angles), max(leg_angles)
    back_min, back_max = min(back_angles), max(back_angles)
    arm_min, arm_max = min(arm_angles), max(arm_angles)
    
    # Add some padding to the ranges
    leg_range = [leg_min - 10, leg_max + 10]
    back_range = [back_min - 10, back_max + 10]
    arm_range = [arm_min - 10, arm_max + 10]
    
    # Create animation frames
    frames = []
    
    # Create frames for animation
    for i in range(len(stroke_data)):
        # Trail: show path up to current point
        trail_leg = leg_angles[:i+1]
        trail_back = back_angles[:i+1]
        trail_arm = arm_angles[:i+1]
        trail_force = forces[:i+1]
        
        frame = go.Frame(
            data=[
                # Trail (path taken so far)
                go.Scatter3d(
                    x=trail_leg,
                    y=trail_back,
                    z=trail_arm,
                    mode='lines+markers',
                    marker=dict(
                        size=3,
                        color=trail_force,
                        colorscale='Viridis',
                        cmin=0,
                        cmax=1,
                        showscale=False
                    ),
                    line=dict(
                        color='rgba(100,100,100,0.5)',
                        width=2
                    ),
                    name='Trail',
                    showlegend=False
                ),
                # Current position (larger marker)
                go.Scatter3d(
                    x=[leg_angles[i]],
                    y=[back_angles[i]],
                    z=[arm_angles[i]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=[forces[i]],
                        colorscale='Viridis',
                        cmin=0,
                        cmax=1,
                        colorbar=dict(
                            title="Force<br>(normalized)",
                            x=1.1
                        ),
                        line=dict(color='white', width=2)
                    ),
                    name='Current',
                    showlegend=False
                )
            ],
            name=f"frame_{i}",
            layout=go.Layout(
                title=f"Stroke #{stroke_number} - Time: {times[i]:.2f}s | Frame: {stroke_data[i]['frame_idx']}",
                scene=dict(
                    xaxis=dict(range=leg_range, autorange=False),
                    yaxis=dict(range=back_range, autorange=False),
                    zaxis=dict(range=arm_range, autorange=False),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)
                    ),
                    aspectmode='cube'
                )
            )
        )
        frames.append(frame)
    
    # Create initial figure
    fig = go.Figure(
        data=[
            # Initial trail (just first point)
            go.Scatter3d(
                x=[leg_angles[0]],
                y=[back_angles[0]],
                z=[arm_angles[0]],
                mode='lines+markers',
                marker=dict(size=3, color=[forces[0]], colorscale='Viridis', cmin=0, cmax=1, showscale=False),
                line=dict(color='rgba(100,100,100,0.5)', width=2),
                name='Trail'
            ),
            # Initial current position
            go.Scatter3d(
                x=[leg_angles[0]],
                y=[back_angles[0]],
                z=[arm_angles[0]],
                mode='markers',
                marker=dict(
                    size=12,
                    color=[forces[0]],
                    colorscale='Viridis',
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Force<br>(normalized)", x=1.1),
                    line=dict(color='white', width=2)
                ),
                name='Current'
            )
        ],
        frames=frames,
        layout=go.Layout(
            title=f"Stroke #{stroke_number} - 3D Biomechanics Animation",
            scene=dict(
                xaxis=dict(title="Leg Angle (°)", range=leg_range, autorange=False),
                yaxis=dict(title="Back Angle (°)", range=back_range, autorange=False),
                zaxis=dict(title="Arm Angle (°)", range=arm_range, autorange=False),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube'
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 50, "redraw": True},
                                         "fromcurrent": True,
                                         "transition": {"duration": 0}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}])
                    ],
                    x=0.1,
                    y=1.15
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    y=-0.1,
                    xanchor="left",
                    currentvalue=dict(
                        prefix="Time: ",
                        visible=True,
                        xanchor="right"
                    ),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    steps=[
                        dict(
                            args=[[f"frame_{k}"],
                                  {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}],
                            method="animate",
                            label=f"{times[k]:.2f}s"
                        )
                        for k in range(len(stroke_data))
                    ]
                )
            ],
            height=800,
            width=1200
        )
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"   ✅ Saved animation: {os.path.basename(output_path)}")
    
    return output_path

def create_all_stroke_animations(analysis_dir):
    """Create 3D animations for all strokes"""
    print("🎬 Creating 3D Stroke Animations")
    print("=" * 50)
    
    data = load_stroke_data(analysis_dir)
    if not data:
        return
    
    # Create output directory
    output_dir = os.path.join(analysis_dir, "3d_animations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate animation for each stroke
    num_strokes = len(data['combined_strokes'])
    generated = []
    
    for stroke_num in range(1, num_strokes + 1):
        output_path = os.path.join(output_dir, f"stroke_{stroke_num:02d}_3d_animation.html")
        result = create_3d_stroke_animation(data, stroke_num, output_path)
        if result:
            generated.append(result)
    
    print(f"\n🎉 Generated {len(generated)} 3D animations!")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n💡 Open any HTML file in your browser to view the interactive 3D animation")
    
    return generated

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate 3D animated stroke visualizations")
    parser.add_argument("analysis_dir", help="Analysis directory path containing the data")
    parser.add_argument("--stroke", type=int, help="Generate animation for specific stroke number only")
    
    args = parser.parse_args()
    
    # Check if analysis directory exists
    if not os.path.exists(args.analysis_dir):
        print(f"❌ Analysis directory not found: {args.analysis_dir}")
        return
    
    if args.stroke:
        # Generate single stroke animation
        data = load_stroke_data(args.analysis_dir)
        if data:
            output_dir = os.path.join(args.analysis_dir, "3d_animations")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"stroke_{args.stroke:02d}_3d_animation.html")
            create_3d_stroke_animation(data, args.stroke, output_path)
    else:
        # Generate all animations
        create_all_stroke_animations(args.analysis_dir)

if __name__ == "__main__":
    main()
