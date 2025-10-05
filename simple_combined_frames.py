#!/usr/bin/env python3
"""
Simple Combined Frames Generator
Generates only the combined frames visualization for each stroke
"""

import cv2
import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

class SimpleCombinedFramesGenerator:
    """Generate only combined frames visualization for each stroke"""
    
    def __init__(self):
        pass
    
    def find_video_file(self, analysis_dir):
        """Find the original video file for the analysis"""
        # Look for video files in the analysis directory
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
        
        return {
            'dataframe': df,
            'video_path': video_path,
            'directory': analysis_dir
        }
    
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
        font_scale = 0.5
        color = (0, 255, 0)  # Green text
        thickness = 1
        
        # Position for angle display (bottom of frame)
        start_x = 10
        start_y = height - 20
        line_height = 15
        
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
        cv2.putText(overlay_frame, frame_text, (width - 60, 20), font, font_scale, (255, 255, 255), thickness)
        
        return overlay_frame
    
    def create_combined_stroke_visualization(self, frames, stroke_number, output_path):
        """Create a combined visualization showing all frames for a stroke"""
        if len(frames) == 0:
            return
        
        # Create figure with subplots
        num_frames = len(frames)
        cols = min(3, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Stroke #{stroke_number} - Key Frames with Angle Measurements', 
                    fontsize=16, fontweight='bold')
        
        # Phase labels
        phase_labels = ["Catch", "Drive Start", "Drive Mid", "Drive End", "Recovery", "Finish"]
        
        for i, frame_info in enumerate(frames):
            if i >= len(axes):
                break
                
            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame_info['frame'], cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(frame_rgb)
            
            # Add title with phase label
            phase_label = phase_labels[i] if i < len(phase_labels) else f"Phase {i+1}"
            axes[i].set_title(f'{phase_label} (Frame #{frame_info["frame_number"]})')
            axes[i].axis('off')
            
            # Add angle measurements as text overlay
            frame_data = frame_info['data']
            angle_text = ""
            angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle']
            for angle in angles:
                if angle in frame_data and not pd.isna(frame_data[angle]):
                    angle_name = angle.replace('_', ' ').title()
                    angle_text += f"{angle_name}: {frame_data[angle]:.1f}°\n"
            
            if angle_text:
                axes[i].text(0.02, 0.98, angle_text.strip(), transform=axes[i].transAxes, 
                           fontsize=8, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(frames), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save combined visualization
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Created combined frames: {os.path.basename(output_path)}")
    
    def generate_stroke_combined_frames(self, data, stroke_number, output_dir):
        """Generate only the combined frames visualization for a stroke"""
        print(f"📸 Creating combined frames for Stroke #{stroke_number}")
        
        # Extract key frames for this stroke
        frames = self.extract_key_frames_for_stroke(data, stroke_number, num_frames=6)
        if not frames:
            print(f"❌ No frames extracted for stroke {stroke_number}")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create combined visualization
        combined_path = os.path.join(output_dir, f"stroke_{stroke_number:02d}_combined_frames.png")
        self.create_combined_stroke_visualization(frames, stroke_number, combined_path)
        
        return combined_path
    
    def generate_all_combined_frames(self, analysis_dir):
        """Generate combined frames for all strokes"""
        print("📸 Generating Combined Frames for All Strokes")
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
        output_dir = os.path.join(analysis_dir, "combined_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate combined frames for each stroke
        generated_files = []
        for stroke_num in strokes:
            combined_path = self.generate_stroke_combined_frames(data, int(stroke_num), output_dir)
            if combined_path:
                generated_files.append(combined_path)
        
        print(f"\n🎉 Generated {len(generated_files)} combined frame visualizations!")
        print(f"📁 Output directory: {output_dir}")
        
        return generated_files

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate combined frames for all strokes")
    parser.add_argument("analysis_dir", help="Analysis directory path containing the data")
    
    args = parser.parse_args()
    
    # Check if analysis directory exists
    if not os.path.exists(args.analysis_dir):
        print(f"❌ Analysis directory not found: {args.analysis_dir}")
        print("Usage: python simple_combined_frames.py <analysis_directory>")
        return
    
    # Initialize generator
    generator = SimpleCombinedFramesGenerator()
    
    # Generate combined frames
    generator.generate_all_combined_frames(args.analysis_dir)

if __name__ == "__main__":
    main()
