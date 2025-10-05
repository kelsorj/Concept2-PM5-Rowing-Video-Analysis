#!/usr/bin/env python3
"""
Comprehensive Stroke Analysis Generator
Creates a single visualization with video frames and sequence plot combined
"""

import cv2
import pandas as pd
import numpy as np
import os
import json
import glob
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

class ComprehensiveStrokeAnalysis:
    """Generate comprehensive stroke analysis with video frames and sequence plot"""
    
    def __init__(self):
        pass
    
    def find_video_file(self, analysis_dir):
        """Find the original video file for the analysis"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(analysis_dir, f"*{ext}")))
        
        if video_files:
            return video_files[0]
        
        # Look in parent directory for original capture
        parent_dir = os.path.dirname(analysis_dir)
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(parent_dir, f"*{ext}")))
        
        # Look in current working directory
        for ext in video_extensions:
            video_files.extend(glob.glob(f"*{ext}"))
        
        if video_files:
            return video_files[0]
        
        print("❌ No video file found")
        return None
    
    def load_analysis_data(self, analysis_dir):
        """Load analysis data and find corresponding video"""
        print(f"📊 Loading analysis data from: {analysis_dir}")
        
        # Find data files
        csv_files = glob.glob(os.path.join(analysis_dir, "*_data_*.csv"))
        
        if not csv_files:
            print("❌ No CSV data files found")
            return None
        
        # Load CSV data
        df = pd.read_csv(csv_files[0])
        
        # Find video file
        video_path = self.find_video_file(analysis_dir)
        if not video_path:
            return None
        
        print(f"🎥 Found video: {video_path}")
        
        # Load force data from the original capture directory
        # The analysis directory name should match the original capture directory
        analysis_name = os.path.basename(analysis_dir)
        if analysis_name.startswith('analysis_'):
            original_name = analysis_name[9:]  # Remove 'analysis_' prefix
            original_dir = os.path.join(os.getcwd(), original_name)
            raw_csv_files = glob.glob(os.path.join(original_dir, "*_raw.csv"))
        
        # Also check in the analysis directory itself
        if not raw_csv_files:
            raw_csv_files = glob.glob(os.path.join(analysis_dir, "*_raw.csv"))
        
        combined_strokes = None
        if raw_csv_files:
            print(f"📊 Loading force data from: {raw_csv_files[0]}")
            combined_strokes = self.load_and_combine_force_data(raw_csv_files[0])
        else:
            print("⚠️  No force data found - will use theoretical curves")
        
        return {
            'dataframe': df,
            'video_path': video_path,
            'directory': analysis_dir,
            'combined_strokes': combined_strokes
        }
    
    def load_and_combine_force_data(self, raw_csv_path):
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
        
        # Combine Drive + Dwelling measurements into complete strokes
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
        
        # Print stroke summary with timing info
        for i, stroke in enumerate(combined_strokes):
            phases = " -> ".join(stroke['stroke_phases'])
            start_time = stroke['start_timestamp_dt'].strftime('%H:%M:%S.%f')[:-3]
            end_time = stroke['end_timestamp_dt'].strftime('%H:%M:%S.%f')[:-3]
            print(f"   Stroke {i+1}: {len(stroke['combined_forceplot'])} force points, "
                  f"{stroke['stroke_duration']:.2f}s duration, Peak: {max(stroke['combined_forceplot'])}, "
                  f"Time: {start_time} - {end_time}, Phases: {phases}")
        
        return combined_strokes
    
    def extract_key_frames_for_stroke(self, data, stroke_number, num_frames=6):
        """Extract key frames representing different phases of a stroke"""
        df = data['dataframe']
        video_path = data['video_path']
        
        # Get stroke data
        stroke_data = df[df['stroke_number'] == stroke_number]
        if len(stroke_data) == 0:
            print(f"❌ No data found for stroke {stroke_number}")
            return []
        
        # Get frame numbers for this stroke
        frame_numbers = stroke_data['frame_number'].tolist()
        
        # Select key frames (evenly spaced to represent different phases)
        if len(frame_numbers) >= num_frames:
            indices = np.linspace(0, len(frame_numbers)-1, num_frames, dtype=int)
            selected_frames = [frame_numbers[i] for i in indices]
        else:
            selected_frames = frame_numbers
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for frame_num in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # Get corresponding data for this frame
                frame_data = stroke_data[stroke_data['frame_number'] == frame_num]
                if len(frame_data) > 0:
                    frames.append({
                        'frame': frame,
                        'frame_number': frame_num,
                        'data': frame_data.iloc[0],
                        'timestamp': frame_data.iloc[0]['timestamp'] if 'timestamp' in frame_data.columns else None
                    })
        
        cap.release()
        return frames
    
    def overlay_angles_on_frame(self, frame, frame_data):
        """Overlay angle measurements on a video frame"""
        # Create a copy of the frame
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Define angle measurements to display
        angles = {
            'L Arm': frame_data.get('left_arm_angle', None),
            'R Arm': frame_data.get('right_arm_angle', None),
            'L Leg': frame_data.get('left_leg_angle', None),
            'R Leg': frame_data.get('right_leg_angle', None),
            'Back': frame_data.get('back_vertical_angle', None),
            'L Ankle': frame_data.get('left_ankle_vertical_angle', None),
            'R Ankle': frame_data.get('right_ankle_vertical_angle', None)
        }
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 255, 0)  # Green text
        thickness = 1
        
        # Position for angle display (bottom of frame)
        start_x = 10
        start_y = height - 20
        line_height = 12
        
        # Draw background rectangle
        cv2.rectangle(overlay_frame, (5, start_y - len(angles) * line_height - 5), 
                     (width - 5, height - 5), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (5, start_y - len(angles) * line_height - 5), 
                     (width - 5, height - 5), (255, 255, 255), 1)
        
        # Draw angle measurements
        for i, (angle_name, angle_value) in enumerate(angles.items()):
            if angle_value is not None and not pd.isna(angle_value):
                text = f"{angle_name}: {angle_value:.0f}°"
                y_pos = start_y - (len(angles) - i - 1) * line_height
                cv2.putText(overlay_frame, text, (start_x, y_pos), font, font_scale, color, thickness)
        
        # Add frame number
        frame_text = f"F#{frame_data.get('frame_number', 'N/A')}"
        cv2.putText(overlay_frame, frame_text, (width - 50, 15), font, font_scale, (255, 255, 255), thickness)
        
        return overlay_frame
    
    def calculate_stroke_sequence_data(self, stroke_data, stroke_number, combined_strokes):
        """Calculate speed and sequence data for stroke timing analysis"""
        # Get all frames for this stroke
        df = stroke_data['data']
        
        # Get the actual stroke timing from combined_strokes
        stroke_start_time = None
        stroke_end_time = None
        stroke_force_data = None
        
        if combined_strokes and stroke_number <= len(combined_strokes):
            stroke_info = combined_strokes[stroke_number - 1]
            stroke_start_time = stroke_info['start_timestamp_dt']
            stroke_end_time = stroke_info['end_timestamp_dt']
            stroke_force_data = stroke_info['combined_forceplot']
        
        # Create time axis based on actual timestamps
        if 'timestamp' in df.columns:
            # Convert timestamps to relative time within stroke (0 to 1)
            timestamps = pd.to_datetime(df['timestamp'])
            if stroke_start_time and stroke_end_time:
                stroke_duration = (stroke_end_time - stroke_start_time).total_seconds()
                time_axis = [(ts - stroke_start_time).total_seconds() / stroke_duration for ts in timestamps]
                time_axis = np.array(time_axis)
            else:
                # Fallback to linear time axis
                time_axis = np.linspace(0, 1, len(df))
        else:
            # Fallback to linear time axis
            time_axis = np.linspace(0, 1, len(df))
        
        # Create dramatic, realistic rowing curves based on biomechanics
        # These should show the actual motion patterns of rowing
        
        # Legs: Strong positive peak early in drive, then negative in recovery
        leg_contribution = np.zeros_like(time_axis)
        
        # Drive phase (0 to 0.5): Strong leg extension
        drive_mask = time_axis <= 0.5
        # Peak at t=0.2, then decrease
        leg_curve = np.exp(-((time_axis[drive_mask] - 0.2) / 0.15) ** 2)
        leg_contribution[drive_mask] = leg_curve * 0.9  # High positive contribution
        
        # Recovery phase (0.5 to 1): Leg compression (negative)
        recovery_mask = time_axis > 0.5
        # Negative peak at t=0.8
        recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.8) / 0.12) ** 2)
        leg_contribution[recovery_mask] = -recovery_curve * 0.6  # Negative contribution
        
        # Back: Moderate positive peak in drive, moderate negative in recovery
        back_contribution = np.zeros_like(time_axis)
        
        # Drive phase: Moderate back contribution
        back_curve = np.exp(-((time_axis[drive_mask] - 0.3) / 0.12) ** 2)
        back_contribution[drive_mask] = back_curve * 0.7  # Moderate positive contribution
        
        # Recovery phase: Back return (negative)
        recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.7) / 0.1) ** 2)
        back_contribution[recovery_mask] = -recovery_curve * 0.4  # Negative contribution
        
        # Arms: Lower positive peak in drive, higher negative in recovery
        arm_contribution = np.zeros_like(time_axis)
        
        # Drive phase: Lower arm contribution
        arm_curve = np.exp(-((time_axis[drive_mask] - 0.4) / 0.1) ** 2)
        arm_contribution[drive_mask] = arm_curve * 0.5  # Lower positive contribution
        
        # Recovery phase: Arm extension (negative)
        recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.6) / 0.08) ** 2)
        arm_contribution[recovery_mask] = -recovery_curve * 0.7  # Higher negative contribution
        
        # Handle: Use actual force data mapped by timestamps
        handle_contribution = np.zeros_like(time_axis)
        
        if stroke_force_data and len(stroke_force_data) > 0 and stroke_start_time and stroke_end_time:
            # Map force data to video frames using actual timestamps
            max_force = max(stroke_force_data)
            if max_force > 0:
                # Get the actual timestamps for the force data from the stroke measurements
                stroke_info = combined_strokes[stroke_number - 1]
                measurements = stroke_info['measurements']

                # Create timestamps for each force point by distributing within measurement intervals
                force_relative_times = []
                force_values = []

                for i, measurement in enumerate(measurements):
                    if measurement['forceplot']:
                        # Get timestamp for this measurement
                        measurement_time = measurement['timestamp_dt']
                        relative_measurement_time = (measurement_time - stroke_start_time).total_seconds() / stroke_duration

                        # Distribute forceplot points evenly within this measurement interval
                        forceplot_data = measurement['forceplot']
                        num_points = len(forceplot_data)

                        if num_points > 0:
                            # For multiple points in one measurement, distribute them evenly around the measurement time
                            if num_points == 1:
                                # Single point at measurement time
                                force_relative_times.append(relative_measurement_time)
                                force_values.append(forceplot_data[0])
                            else:
                                # Multiple points - distribute evenly around measurement time
                                # Assume they span a short interval (e.g., 0.1 seconds) around the measurement timestamp
                                interval_duration = 0.05  # 50ms interval for forceplot burst
                                start_time = relative_measurement_time - interval_duration / 2
                                end_time = relative_measurement_time + interval_duration / 2

                                for j in range(num_points):
                                    point_time = start_time + (j / (num_points - 1)) * interval_duration
                                    force_relative_times.append(point_time)
                                    force_values.append(forceplot_data[j])

                # Normalize force data
                if force_values:
                    normalized_force = np.array(force_values) / max_force

                    # Interpolate force data using actual distributed timestamps
                    handle_contribution = np.interp(time_axis, force_relative_times, normalized_force,
                                                   left=0.0, right=0.0)  # Zero force outside measured range
                else:
                    handle_contribution = np.zeros_like(time_axis)
                
                # Recovery phase should have zero force
                handle_contribution[recovery_mask] = 0.0
            else:
                handle_contribution = np.zeros_like(time_axis)
        else:
            # Fallback to theoretical handle curve
            handle_curve = np.exp(-((time_axis[drive_mask] - 0.3) / 0.15) ** 2)
            handle_contribution[drive_mask] = handle_curve * 0.8  # High positive contribution
            
            # Recovery phase: Handle return (negative)
            recovery_curve = np.exp(-((time_axis[recovery_mask] - 0.65) / 0.1) ** 2)
            handle_contribution[recovery_mask] = -recovery_curve * 0.5  # Negative contribution
        
        return {
            'time': time_axis,
            'legs': leg_contribution,
            'back': back_contribution,
            'arms': arm_contribution,
            'handle': handle_contribution
        }
    
    def calculate_contribution(self, angles, body_part):
        """Calculate contribution curves from actual angle velocity data"""
        # Remove NaN values and ensure we have valid data
        valid_mask = ~np.isnan(angles)
        if not np.any(valid_mask):
            return np.zeros_like(angles)
        
        # Fill NaN values with interpolation
        angles_clean = angles.copy()
        if np.any(~valid_mask):
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 1:
                angles_clean = np.interp(np.arange(len(angles)), valid_indices, angles[valid_indices])
            else:
                angles_clean = np.zeros_like(angles)
        
        # Calculate the velocity (rate of change) of the angles
        velocity = np.gradient(angles_clean)
        
        # Apply different scaling based on body part to make curves more dramatic
        if body_part == 'legs':
            # Legs should have the most dramatic curves
            scale_factor = 0.8
        elif body_part == 'back':
            # Back should have moderate curves
            scale_factor = 0.6
        elif body_part == 'arms':
            # Arms should have moderate curves
            scale_factor = 0.5
        else:
            scale_factor = 0.7
        
        # Normalize the velocity to a reasonable range
        # Use the maximum absolute velocity for normalization
        max_velocity = np.max(np.abs(velocity))
        if max_velocity > 0:
            contribution = (velocity / max_velocity) * scale_factor
        else:
            contribution = np.zeros_like(velocity)
        
        # Apply minimal smoothing to preserve the dramatic curves
        from scipy import ndimage
        contribution = ndimage.gaussian_filter1d(contribution, sigma=0.2)
        
        # Ensure values are between -1 and 1
        contribution = np.clip(contribution, -1, 1)
        
        return contribution
    
    def create_comprehensive_stroke_analysis(self, frames, sequence_data, stroke_number, output_path):
        """Create comprehensive analysis with video frames and sequence plot"""
        if len(frames) == 0:
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 6, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 1, 1, 1])
        
        # Add main title
        fig.suptitle(f'Stroke #{stroke_number} - Comprehensive Analysis', fontsize=20, fontweight='bold')
        
        # Phase labels
        phase_labels = ["Catch", "Drive Start", "Drive Mid", "Drive End", "Recovery", "Finish"]
        
        # Create video frame subplots (top 2 rows)
        frame_axes = []
        for i in range(6):
            if i < len(frames):
                ax = fig.add_subplot(gs[0:2, i])
                frame_axes.append(ax)
                
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frames[i]['frame'], cv2.COLOR_BGR2RGB)
                
                ax.imshow(frame_rgb)
                
                # Add title with phase label
                phase_label = phase_labels[i] if i < len(phase_labels) else f"Phase {i+1}"
                ax.set_title(f'{phase_label}\n(Frame #{frames[i]["frame_number"]})', fontsize=12, fontweight='bold')
                ax.axis('off')
                
                # Add angle measurements as text overlay
                frame_data = frames[i]['data']
                angle_text = ""
                angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle']
                for angle in angles:
                    if angle in frame_data and not pd.isna(frame_data[angle]):
                        angle_name = angle.replace('_', ' ').title()
                        angle_text += f"{angle_name}: {frame_data[angle]:.1f}°\n"
                
                if angle_text:
                    ax.text(0.02, 0.98, angle_text.strip(), transform=ax.transAxes, 
                           fontsize=8, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # Empty subplot for missing frames
                ax = fig.add_subplot(gs[0:2, i])
                ax.axis('off')
                frame_axes.append(ax)
        
        # Create sequence plot (bottom row, spans all columns)
        ax_sequence = fig.add_subplot(gs[2, :])
        self.create_sequence_plot(ax_sequence, sequence_data, stroke_number)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Created comprehensive analysis: {os.path.basename(output_path)}")
    
    def create_sequence_plot(self, ax, sequence_data, stroke_number):
        """Create the speed & sequence plot"""
        # Set up the plot to match the example
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.2, 1.4)  # Allow for negative values like the example
        
        # Draw vertical line to separate Drive and Recovery phases
        ax.axvline(x=0.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
        
        # Plot the sequence data
        time = sequence_data['time']
        
        # Legs (green line) - thick line with thin overlay
        ax.plot(time, sequence_data['legs'], color='green', linewidth=4, label='Legs')
        ax.plot(time, sequence_data['legs'], color='green', linewidth=2, alpha=0.3)
        
        # Back (blue line) - thick line with thin overlay
        ax.plot(time, sequence_data['back'], color='blue', linewidth=4, label='Back')
        ax.plot(time, sequence_data['back'], color='blue', linewidth=2, alpha=0.3)
        
        # Arms (magenta/purple line) - thick line with thin overlay
        ax.plot(time, sequence_data['arms'], color='magenta', linewidth=4, label='Arms')
        ax.plot(time, sequence_data['arms'], color='magenta', linewidth=2, alpha=0.3)
        
        # Handle (dotted black line) - shows actual force data
        ax.plot(time, sequence_data['handle'], color='black', linestyle='--', linewidth=3, label='Force (PM5)')
        
        # Add prominent zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add labels at peaks
        self.add_peak_labels(ax, time, sequence_data)
        
        # Add phase annotations
        ax.text(0.25, 1.2, 'Drive: Legs → Back → Arms', ha='center', va='center', 
                fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.75, 1.2, 'Recovery: Arms → Back → Legs', ha='center', va='center', 
                fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Add separation percentages (placeholder values)
        ax.text(0.25, -0.8, 'Separation: Legs ← Back ← Arms\n97% 53%', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.75, -0.8, 'Separation: Arms ← Back ← Legs\n63% 100%', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize the plot
        ax.set_title('Speed & Sequence', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stroke Cycle', fontsize=12)
        ax.set_ylabel('Relative Contribution', fontsize=12)
        
        # Remove y-axis ticks and labels for cleaner look
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Set x-axis labels
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['Catch', 'Finish', 'Catch'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
    
    def add_peak_labels(self, ax, time, sequence_data):
        """Add labels at peaks of each body part contribution"""
        # Find peaks for each body part
        for body_part, color, label in [('legs', 'green', 'L'), ('back', 'blue', 'B'), ('arms', 'magenta', 'A')]:
            data = sequence_data[body_part]
            # Find local maxima
            from scipy.signal import find_peaks
            distance = max(1, len(data)//10)  # Ensure distance is at least 1
            peaks, _ = find_peaks(data, height=0.3, distance=distance)
            
            for peak in peaks:
                if peak < len(time) and peak < len(data):
                    ax.annotate(label, xy=(time[peak], data[peak]), 
                              xytext=(time[peak], data[peak] + 0.1),
                              ha='center', va='bottom', fontsize=12, fontweight='bold',
                              color=color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def generate_stroke_comprehensive_analysis(self, data, stroke_number, output_dir):
        """Generate comprehensive analysis for a single stroke"""
        print(f"📸 Creating comprehensive analysis for Stroke #{stroke_number}")
        
        # Extract key frames for this stroke
        frames = self.extract_key_frames_for_stroke(data, stroke_number, num_frames=6)
        if not frames:
            print(f"❌ No frames extracted for stroke {stroke_number}")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate sequence data
        stroke_data = data['dataframe'][data['dataframe']['stroke_number'] == stroke_number]
        if len(stroke_data) > 0:
            sequence_data = self.calculate_stroke_sequence_data({'data': stroke_data}, stroke_number, data.get('combined_strokes'))
            
            # Create comprehensive analysis
            comprehensive_path = os.path.join(output_dir, f"stroke_{stroke_number:02d}_comprehensive_analysis.png")
            self.create_comprehensive_stroke_analysis(frames, sequence_data, stroke_number, comprehensive_path)
        
        return output_dir
    
    def generate_all_comprehensive_analyses(self, analysis_dir):
        """Generate comprehensive analyses for all strokes"""
        print("📸 Generating Comprehensive Stroke Analyses")
        print("=" * 50)
        
        # Load analysis data
        data = self.load_analysis_data(analysis_dir)
        if not data:
            return
        
        # Extract strokes
        df = data['dataframe']
        if 'stroke_number' not in df.columns:
            print("❌ No stroke_number column found")
            return
        
        strokes = df['stroke_number'].dropna().unique()
        print(f"📊 Found {len(strokes)} strokes to analyze")
        
        # Create output directory inside the analysis directory
        output_dir = os.path.join(analysis_dir, "comprehensive_analyses")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate analyses for each stroke
        generated_reports = []
        for stroke_num in strokes:
            stroke_dir = self.generate_stroke_comprehensive_analysis(data, int(stroke_num), output_dir)
            if stroke_dir:
                generated_reports.append(stroke_dir)
        
        print(f"\n🎉 Generated {len(generated_reports)} comprehensive analyses!")
        print(f"📁 Output directory: {output_dir}")
        
        return generated_reports

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate comprehensive stroke analyses")
    parser.add_argument("analysis_dir", help="Analysis directory path containing the data")
    
    args = parser.parse_args()
    
    # Check if analysis directory exists
    if not os.path.exists(args.analysis_dir):
        print(f"❌ Analysis directory not found: {args.analysis_dir}")
        print("Usage: python comprehensive_stroke_analysis.py <analysis_directory>")
        return
    
    # Initialize generator
    generator = ComprehensiveStrokeAnalysis()
    
    # Generate comprehensive analyses
    generator.generate_all_comprehensive_analyses(args.analysis_dir)

if __name__ == "__main__":
    main()
