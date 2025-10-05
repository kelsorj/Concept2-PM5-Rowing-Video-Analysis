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
import plotly.express as px
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

def create_combined_stroke_analysis(analysis_dir):
    """Create a combined 3D visualization showing all strokes"""
    print("🎬 Creating Combined Stroke Analysis")
    print("=" * 50)
    
    data = load_stroke_data(analysis_dir)
    if not data:
        return
    
    pose_frames = data['pose_frames']
    frame_timestamps = data['frame_timestamps']
    combined_strokes = data['combined_strokes']
    
    if not combined_strokes:
        print("❌ No stroke data found")
        return
    
    print(f"📊 Processing {len(combined_strokes)} strokes...")
    
    # Collect all stroke data
    all_stroke_data = []
    all_leg_angles = []
    all_back_angles = []
    all_arm_angles = []
    all_forces = []
    
    for stroke_idx, stroke in enumerate(combined_strokes):
        stroke_data = []
        
        # Get frames for this stroke
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
                            'frame_idx': ft['frame_idx'],
                            'stroke_number': stroke_idx + 1
                        })
        
        if len(stroke_data) > 0:
            # Map force data to frames
            force_curve = stroke['combined_forceplot']
            if force_curve:
                for i, frame_data in enumerate(stroke_data):
                    time_ratio = frame_data['time'] / stroke_data[-1]['time']
                    force_idx = int(time_ratio * (len(force_curve) - 1))
                    force_idx = min(force_idx, len(force_curve) - 1)
                    frame_data['force'] = force_curve[force_idx]
            else:
                for frame_data in stroke_data:
                    frame_data['force'] = 0
            
            all_stroke_data.append(stroke_data)
            
            # Collect for global ranges
            leg_angles = [d['leg_angle'] for d in stroke_data]
            back_angles = [d['back_angle'] for d in stroke_data]
            arm_angles = [d['arm_angle'] for d in stroke_data]
            forces = [d['force'] for d in stroke_data]
            
            all_leg_angles.extend(leg_angles)
            all_back_angles.extend(back_angles)
            all_arm_angles.extend(arm_angles)
            all_forces.extend(forces)
    
    if not all_stroke_data:
        print("❌ No valid stroke data found")
        return
    
    # Calculate global ranges
    leg_min, leg_max = min(all_leg_angles), max(all_leg_angles)
    back_min, back_max = min(all_back_angles), max(all_back_angles)
    arm_min, arm_max = min(all_arm_angles), max(all_arm_angles)
    force_min, force_max = min(all_forces), max(all_forces)
    
    leg_range = [leg_min - 10, leg_max + 10]
    back_range = [back_min - 10, back_max + 10]
    arm_range = [arm_min - 10, arm_max + 10]
    
    # Create traces for each stroke
    traces = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for stroke_idx, stroke_data in enumerate(all_stroke_data):
        leg_angles = [d['leg_angle'] for d in stroke_data]
        back_angles = [d['back_angle'] for d in stroke_data]
        arm_angles = [d['arm_angle'] for d in stroke_data]
        forces = [d['force'] for d in stroke_data]
        
        color = colors[stroke_idx % len(colors)]
        
        # Create stroke trace
        trace = go.Scatter3d(
            x=leg_angles,
            y=back_angles,
            z=arm_angles,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=forces,
                colorscale='Viridis',
                cmin=force_min,
                cmax=force_max,
                showscale=(stroke_idx == 0),  # Only show colorbar for first stroke
                colorbar=dict(
                    title="Force (N)",
                    x=1.05,
                    len=0.7,
                    y=0.5,
                    yanchor='middle'
                ),
                line=dict(color=color, width=1)
            ),
            line=dict(
                color=color,
                width=3
            ),
            name=f'Stroke {stroke_idx + 1}',
            hovertemplate=f'<b>Stroke {stroke_idx + 1}</b><br>' +
                         'Leg: %{x:.1f}°<br>' +
                         'Back: %{y:.1f}°<br>' +
                         'Arm: %{z:.1f}°<br>' +
                         'Force: %{marker.color:.1f} N<br>' +
                         '<extra></extra>'
        )
        traces.append(trace)
    
    # Create the figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"Combined Stroke Analysis - All {len(all_stroke_data)} Strokes",
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
            height=800,
            width=1400,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                yanchor='top',
                xanchor='left',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            margin=dict(l=0, r=120, t=50, b=0)
        )
    )
    
    # Save to HTML
    output_dir = os.path.join(analysis_dir, "3d_animations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "combined_stroke_analysis.html")
    fig.write_html(output_path)
    
    print(f"   ✅ Saved combined analysis: {os.path.basename(output_path)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n💡 Open the HTML file in your browser to view the interactive combined analysis")
    
    return output_path

def create_interactive_axis_dashboard(analysis_dir):
    """Create an interactive dashboard where users can choose which angle goes on which axis"""
    print("🎛️ Creating Interactive Axis Dashboard")
    print("=" * 50)
    
    data = load_stroke_data(analysis_dir)
    if not data:
        return
    
    pose_frames = data['pose_frames']
    frame_timestamps = data['frame_timestamps']
    combined_strokes = data['combined_strokes']
    
    if not combined_strokes:
        print("❌ No stroke data found")
        return
    
    print(f"📊 Processing {len(combined_strokes)} strokes...")
    
    # Collect all stroke data with all available angles
    all_stroke_data = []
    all_angles = {
        'leg_angle': [],
        'back_angle': [],
        'arm_angle': [],
        'left_leg_angle': [],
        'right_leg_angle': [],
        'left_arm_angle': [],
        'right_arm_angle': [],
        'left_shoulder_angle': [],
        'right_shoulder_angle': [],
        'left_hip_angle': [],
        'right_hip_angle': [],
        'left_knee_angle': [],
        'right_knee_angle': [],
        'left_elbow_angle': [],
        'right_elbow_angle': []
    }
    all_forces = []
    
    for stroke_idx, stroke in enumerate(combined_strokes):
        stroke_data = []
        
        # Get frames for this stroke
        for i, ft in enumerate(frame_timestamps):
            if stroke['start_timestamp'] <= ft['timestamp'] <= stroke['end_timestamp']:
                if i < len(pose_frames):
                    frame = pose_frames[i]
                    
                    # Calculate all available angles
                    angles = {}
                    
                    # Basic angles (averaged)
                    left_leg = frame.get('left_leg_angle')
                    right_leg = frame.get('right_leg_angle')
                    left_arm = frame.get('left_arm_angle')
                    right_arm = frame.get('right_arm_angle')
                    
                    angles['leg_angle'] = np.mean([v for v in [left_leg, right_leg] if v is not None]) if any(v is not None for v in [left_leg, right_leg]) else None
                    angles['back_angle'] = frame.get('back_vertical_angle')
                    angles['arm_angle'] = np.mean([v for v in [left_arm, right_arm] if v is not None]) if any(v is not None for v in [left_arm, right_arm]) else None
                    
                    # Individual limb angles
                    angles['left_leg_angle'] = left_leg
                    angles['right_leg_angle'] = right_leg
                    angles['left_arm_angle'] = left_arm
                    angles['right_arm_angle'] = right_arm
                    
                    # Additional angles if available
                    angles['left_shoulder_angle'] = frame.get('left_shoulder_angle')
                    angles['right_shoulder_angle'] = frame.get('right_shoulder_angle')
                    angles['left_hip_angle'] = frame.get('left_hip_angle')
                    angles['right_hip_angle'] = frame.get('right_hip_angle')
                    angles['left_knee_angle'] = frame.get('left_knee_angle')
                    angles['right_knee_angle'] = frame.get('right_knee_angle')
                    angles['left_elbow_angle'] = frame.get('left_elbow_angle')
                    angles['right_elbow_angle'] = frame.get('right_elbow_angle')
                    
                    # Check if we have at least the basic angles
                    if not np.isnan(angles['leg_angle']) and angles['back_angle'] is not None and not np.isnan(angles['arm_angle']):
                        frame_data = {
                            'time': (ft['timestamp'] - stroke['start_timestamp']).total_seconds(),
                            'frame_idx': ft['frame_idx'],
                            'stroke_number': stroke_idx + 1,
                            **angles
                        }
                        stroke_data.append(frame_data)
        
        if len(stroke_data) > 0:
            # Map force data to frames
            force_curve = stroke['combined_forceplot']
            if force_curve:
                for i, frame_data in enumerate(stroke_data):
                    time_ratio = frame_data['time'] / stroke_data[-1]['time']
                    force_idx = int(time_ratio * (len(force_curve) - 1))
                    force_idx = min(force_idx, len(force_curve) - 1)
                    frame_data['force'] = force_curve[force_idx]
            else:
                for frame_data in stroke_data:
                    frame_data['force'] = 0
            
            all_stroke_data.append(stroke_data)
            
            # Collect all angle values for range calculation
            for angle_name in all_angles.keys():
                values = [d[angle_name] for d in stroke_data if d[angle_name] is not None and not np.isnan(d[angle_name])]
                all_angles[angle_name].extend(values)
            
            # Collect forces
            forces = [d['force'] for d in stroke_data]
            all_forces.extend(forces)
    
    if not all_stroke_data:
        print("❌ No valid stroke data found")
        return
    
    # Calculate ranges for all angles
    angle_ranges = {}
    for angle_name, values in all_angles.items():
        if values:
            min_val, max_val = min(values), max(values)
            angle_ranges[angle_name] = [min_val - 10, max_val + 10]
        else:
            angle_ranges[angle_name] = [0, 180]  # Default range
    
    force_min, force_max = min(all_forces), max(all_forces)
    
    # Create HTML with interactive controls
    stroke_data_json = json.dumps(all_stroke_data)
    angle_ranges_json = json.dumps(angle_ranges)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .controls {{ 
                background-color: #f0f0f0; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;
                display: flex;
                gap: 30px;
                flex-wrap: wrap;
            }}
            .axis-control {{ 
                display: flex; 
                flex-direction: column; 
                min-width: 200px;
            }}
            .axis-control h3 {{ margin: 0 0 10px 0; color: #333; }}
            .checkbox-group {{ display: flex; flex-direction: column; gap: 5px; }}
            .checkbox-item {{ display: flex; align-items: center; gap: 8px; }}
            .checkbox-item input {{ margin: 0; }}
            .checkbox-item label {{ margin: 0; font-size: 14px; }}
            #plot {{ width: 100%; height: 800px; }}
            .info {{ 
                background-color: #e8f4f8; 
                padding: 15px; 
                border-radius: 5px; 
                margin-bottom: 20px;
                border-left: 4px solid #2196F3;
            }}
        </style>
    </head>
    <body>
        <h1>Interactive Rowing Stroke Analysis Dashboard</h1>
        
        <div class="info">
            <strong>Instructions:</strong> Select which angle measurements you want to display on each axis. 
            You can choose from individual limb angles or averaged values. The color represents force in Newtons.
        </div>
        
        <div class="controls">
            <div class="axis-control">
                <h3>X-Axis (Horizontal)</h3>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="radio" name="x-axis" value="leg_angle" id="x-leg" checked>
                        <label for="x-leg">Leg Angle (avg)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="x-axis" value="back_angle" id="x-back">
                        <label for="x-back">Back Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="x-axis" value="arm_angle" id="x-arm">
                        <label for="x-arm">Arm Angle (avg)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="x-axis" value="left_leg_angle" id="x-left-leg">
                        <label for="x-left-leg">Left Leg Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="x-axis" value="right_leg_angle" id="x-right-leg">
                        <label for="x-right-leg">Right Leg Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="x-axis" value="left_arm_angle" id="x-left-arm">
                        <label for="x-left-arm">Left Arm Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="x-axis" value="right_arm_angle" id="x-right-arm">
                        <label for="x-right-arm">Right Arm Angle</label>
                    </div>
                </div>
            </div>
            
            <div class="axis-control">
                <h3>Y-Axis (Depth)</h3>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="radio" name="y-axis" value="back_angle" id="y-back" checked>
                        <label for="y-back">Back Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="y-axis" value="leg_angle" id="y-leg">
                        <label for="y-leg">Leg Angle (avg)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="y-axis" value="arm_angle" id="y-arm">
                        <label for="y-arm">Arm Angle (avg)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="y-axis" value="left_leg_angle" id="y-left-leg">
                        <label for="y-left-leg">Left Leg Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="y-axis" value="right_leg_angle" id="y-right-leg">
                        <label for="y-right-leg">Right Leg Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="y-axis" value="left_arm_angle" id="y-left-arm">
                        <label for="y-left-arm">Left Arm Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="y-axis" value="right_arm_angle" id="y-right-arm">
                        <label for="y-right-arm">Right Arm Angle</label>
                    </div>
                </div>
            </div>
            
            <div class="axis-control">
                <h3>Z-Axis (Vertical)</h3>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="radio" name="z-axis" value="arm_angle" id="z-arm" checked>
                        <label for="z-arm">Arm Angle (avg)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="z-axis" value="leg_angle" id="z-leg">
                        <label for="z-leg">Leg Angle (avg)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="z-axis" value="back_angle" id="z-back">
                        <label for="z-back">Back Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="z-axis" value="left_leg_angle" id="z-left-leg">
                        <label for="z-left-leg">Left Leg Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="z-axis" value="right_leg_angle" id="z-right-leg">
                        <label for="z-right-leg">Right Leg Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="z-axis" value="left_arm_angle" id="z-left-arm">
                        <label for="z-left-arm">Left Arm Angle</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="radio" name="z-axis" value="right_arm_angle" id="z-right-arm">
                        <label for="z-right-arm">Right Arm Angle</label>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="plot"></div>
        
        <script>
            // Data from Python
            const strokeData = {stroke_data_json};
            const angleRanges = {angle_ranges_json};
            const forceMin = {force_min};
            const forceMax = {force_max};
            
            function updatePlot() {{
                const xAxis = document.querySelector('input[name="x-axis"]:checked').value;
                const yAxis = document.querySelector('input[name="y-axis"]:checked').value;
                const zAxis = document.querySelector('input[name="z-axis"]:checked').value;
                
                const traces = [];
                const colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'];
                
                strokeData.forEach((stroke, strokeIdx) => {{
                    const xValues = stroke.map(d => d[xAxis]).filter(v => v !== null && !isNaN(v));
                    const yValues = stroke.map(d => d[yAxis]).filter(v => v !== null && !isNaN(v));
                    const zValues = stroke.map(d => d[zAxis]).filter(v => v !== null && !isNaN(v));
                    const forces = stroke.map(d => d.force);
                    
                    if (xValues.length > 0 && yValues.length > 0 && zValues.length > 0) {{
                        const color = colors[strokeIdx % colors.length];
                        
                        traces.push({{
                            x: xValues,
                            y: yValues,
                            z: zValues,
                            type: 'scatter3d',
                            mode: 'lines+markers',
                            marker: {{
                                size: 4,
                                color: forces,
                                colorscale: 'Viridis',
                                cmin: forceMin,
                                cmax: forceMax,
                                showscale: strokeIdx === 0,
                                colorbar: {{
                                    title: "Force (N)",
                                    x: 1.05,
                                    len: 0.7,
                                    y: 0.5,
                                    yanchor: 'middle'
                                }},
                                line: {{ color: color, width: 1 }}
                            }},
                            line: {{
                                color: color,
                                width: 3
                            }},
                            name: `Stroke ${{strokeIdx + 1}}`,
                            hovertemplate: `<b>Stroke ${{strokeIdx + 1}}</b><br>` +
                                         `${{xAxis}}: %{{x:.1f}}°<br>` +
                                         `${{yAxis}}: %{{y:.1f}}°<br>` +
                                         `${{zAxis}}: %{{z:.1f}}°<br>` +
                                         `Force: %{{marker.color:.1f}} N<br>` +
                                         `<extra></extra>`
                        }});
                    }}
                }});
                
                const layout = {{
                    title: `Interactive Stroke Analysis - ${{strokeData.length}} Strokes`,
                    scene: {{
                        xaxis: {{ title: `${{xAxis.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase())}} (°)`, range: angleRanges[xAxis], autorange: false }},
                        yaxis: {{ title: `${{yAxis.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase())}} (°)`, range: angleRanges[yAxis], autorange: false }},
                        zaxis: {{ title: `${{zAxis.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase())}} (°)`, range: angleRanges[zAxis], autorange: false }},
                        camera: {{
                            eye: {{ x: 1.5, y: 1.5, z: 1.2 }},
                            center: {{ x: 0, y: 0, z: 0 }},
                            up: {{ x: 0, y: 0, z: 1 }}
                        }},
                        aspectmode: 'cube'
                    }},
                    height: 800,
                    width: 1400,
                    showlegend: true,
                    legend: {{
                        x: 0.02,
                        y: 0.98,
                        yanchor: 'top',
                        xanchor: 'left',
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: 'rgba(0,0,0,0.2)',
                        borderwidth: 1
                    }},
                    margin: {{ l: 0, r: 120, t: 50, b: 0 }}
                }};
                
                Plotly.newPlot('plot', traces, layout);
            }}
            
            // Add event listeners
            document.querySelectorAll('input[name="x-axis"], input[name="y-axis"], input[name="z-axis"]').forEach(input => {{
                input.addEventListener('change', updatePlot);
            }});
            
            // Initial plot
            updatePlot();
        </script>
    </body>
    </html>
    """
    
    # Save to HTML
    output_dir = os.path.join(analysis_dir, "3d_animations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "interactive_axis_dashboard.html")
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"   ✅ Saved interactive dashboard: {os.path.basename(output_path)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n💡 Open the HTML file in your browser to interactively select which angles to display on each axis")
    
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
    parser.add_argument("--combined", action="store_true", help="Generate combined analysis showing all strokes")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive dashboard with axis selection")
    
    args = parser.parse_args()
    
    # Check if analysis directory exists
    if not os.path.exists(args.analysis_dir):
        print(f"❌ Analysis directory not found: {args.analysis_dir}")
        return
    
    if args.interactive:
        # Generate interactive dashboard
        create_interactive_axis_dashboard(args.analysis_dir)
    elif args.combined:
        # Generate combined analysis
        create_combined_stroke_analysis(args.analysis_dir)
    elif args.stroke:
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
