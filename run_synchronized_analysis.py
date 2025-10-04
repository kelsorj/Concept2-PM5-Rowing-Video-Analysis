#!/usr/bin/env python3
"""
Run advanced analysis using synchronized force and pose data
"""

import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import glob
import os
from scipy import signal
from scipy.stats import pearsonr
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings
warnings.filterwarnings('ignore')
import argparse

class SynchronizedRowingAnalyzer:
    def __init__(self):
        self.synchronized_data = None
        self.pose_data = []
        self.force_data = []
        self.analysis_results = {}
        self.plot_scale = 0.75
        self.time_offset_s = 0.0
        
    def load_synchronized_data(self, sync_json_path):
        """Load synchronized data from JSON file"""
        print(f"📊 Loading synchronized data: {sync_json_path}")
        
        with open(sync_json_path, 'r') as f:
            self.synchronized_data = json.load(f)
        
        # Extract pose and force data
        for entry in self.synchronized_data:
            if entry.get('pose_data'):
                pose_frame = entry['pose_data'].copy()
                pose_frame['timestamp'] = entry['frame_elapsed_s']
                pose_frame['frame_idx'] = entry['frame_idx']
                self.pose_data.append(pose_frame)
            
            if entry.get('force_data'):
                force_entry = entry['force_data'].copy()
                force_entry['elapsed_s'] = entry['frame_elapsed_s']
                force_entry['timestamp_iso'] = entry['frame_timestamp_iso']
                self.force_data.append(force_entry)
        
        print(f"   Loaded {len(self.pose_data)} pose frames")
        print(f"   Loaded {len(self.force_data)} force data points")
        
        return True
    
    def analyze_rowing_technique(self):
        """Perform comprehensive rowing technique analysis"""
        print("\n🔍 Performing Comprehensive Rowing Analysis...")
        
        if not self.pose_data:
            print("❌ No pose data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.pose_data)
        df = df.dropna(subset=['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle'])
        
        if len(df) == 0:
            print("❌ No valid pose data after filtering")
            return
        
        # Calculate comprehensive metrics
        self._calculate_joint_kinematics(df)
        self._calculate_symmetry_analysis(df)
        self._calculate_timing_analysis(df)
        self._calculate_technique_quality(df)
        self._calculate_force_analysis(df)
        
        print("✅ Comprehensive analysis completed")
    
    def _calculate_joint_kinematics(self, df):
        """Calculate detailed joint kinematics"""
        print("   📊 Calculating joint kinematics...")
        
        kinematics = {}
        
        # Arm kinematics
        kinematics['left_arm'] = {
            'range_of_motion': df['left_arm_angle'].max() - df['left_arm_angle'].min(),
            'mean_angle': df['left_arm_angle'].mean(),
            'std_angle': df['left_arm_angle'].std(),
            'min_angle': df['left_arm_angle'].min(),
            'max_angle': df['left_arm_angle'].max(),
            'angular_velocity': np.mean(np.abs(np.diff(df['left_arm_angle'])))
        }
        
        kinematics['right_arm'] = {
            'range_of_motion': df['right_arm_angle'].max() - df['right_arm_angle'].min(),
            'mean_angle': df['right_arm_angle'].mean(),
            'std_angle': df['right_arm_angle'].std(),
            'min_angle': df['right_arm_angle'].min(),
            'max_angle': df['right_arm_angle'].max(),
            'angular_velocity': np.mean(np.abs(np.diff(df['right_arm_angle'])))
        }
        
        # Leg kinematics
        kinematics['left_leg'] = {
            'range_of_motion': df['left_leg_angle'].max() - df['left_leg_angle'].min(),
            'mean_angle': df['left_leg_angle'].mean(),
            'std_angle': df['left_leg_angle'].std(),
            'min_angle': df['left_leg_angle'].min(),
            'max_angle': df['left_leg_angle'].max(),
            'angular_velocity': np.mean(np.abs(np.diff(df['left_leg_angle'])))
        }
        
        kinematics['right_leg'] = {
            'range_of_motion': df['right_leg_angle'].max() - df['right_leg_angle'].min(),
            'mean_angle': df['right_leg_angle'].mean(),
            'std_angle': df['right_leg_angle'].std(),
            'min_angle': df['right_leg_angle'].min(),
            'max_angle': df['right_leg_angle'].max(),
            'angular_velocity': np.mean(np.abs(np.diff(df['right_leg_angle'])))
        }
        
        # Torso kinematics (if available)
        if 'torso_lean_angle' in df.columns:
            kinematics['torso'] = {
                'range_of_motion': df['torso_lean_angle'].max() - df['torso_lean_angle'].min(),
                'mean_angle': df['torso_lean_angle'].mean(),
                'std_angle': df['torso_lean_angle'].std(),
                'min_angle': df['torso_lean_angle'].min(),
                'max_angle': df['torso_lean_angle'].max(),
                'angular_velocity': np.mean(np.abs(np.diff(df['torso_lean_angle'])))
            }
        
        self.analysis_results['joint_kinematics'] = kinematics
    
    def _calculate_symmetry_analysis(self, df):
        """Calculate bilateral symmetry analysis"""
        print("   📊 Calculating symmetry analysis...")
        
        symmetry = {}
        
        # Arm symmetry
        arm_diff = df['left_arm_angle'] - df['right_arm_angle']
        symmetry['arm_symmetry'] = {
            'mean_difference': arm_diff.mean(),
            'std_difference': arm_diff.std(),
            'max_difference': arm_diff.abs().max(),
            'correlation': pearsonr(df['left_arm_angle'], df['right_arm_angle'])[0],
            'symmetry_index': 100 - (arm_diff.abs().mean() / df['left_arm_angle'].mean() * 100)
        }
        
        # Leg symmetry
        leg_diff = df['left_leg_angle'] - df['right_leg_angle']
        symmetry['leg_symmetry'] = {
            'mean_difference': leg_diff.mean(),
            'std_difference': leg_diff.std(),
            'max_difference': leg_diff.abs().max(),
            'correlation': pearsonr(df['left_leg_angle'], df['right_leg_angle'])[0],
            'symmetry_index': 100 - (leg_diff.abs().mean() / df['left_leg_angle'].mean() * 100)
        }
        
        # Overall symmetry score
        symmetry['overall_symmetry_score'] = (symmetry['arm_symmetry']['symmetry_index'] + 
                                            symmetry['leg_symmetry']['symmetry_index']) / 2
        
        self.analysis_results['symmetry_analysis'] = symmetry
    
    def _calculate_timing_analysis(self, df):
        """Calculate timing and rhythm analysis"""
        print("   📊 Calculating timing analysis...")
        
        timing = {}
        
        # Calculate stroke rate from leg angle cycles
        leg_angles = (df['left_leg_angle'] + df['right_leg_angle']) / 2
        
        # Find peaks in leg angle (legs extended)
        peaks, _ = signal.find_peaks(leg_angles, distance=10, prominence=5)
        
        if len(peaks) > 1:
            # Calculate stroke rate
            total_time = df['timestamp'].max() - df['timestamp'].min()
            stroke_rate = (len(peaks) - 1) / (total_time / 60)  # strokes per minute
            timing['stroke_rate'] = stroke_rate
            
            # Calculate stroke duration
            timing['stroke_duration'] = total_time / (len(peaks) - 1)
            
            # Calculate rhythm consistency
            if len(peaks) > 2:
                peak_intervals = np.diff(peaks)
                timing['rhythm_consistency'] = 100 - (np.std(peak_intervals) / np.mean(peak_intervals) * 100)
            else:
                timing['rhythm_consistency'] = 0
        else:
            timing['stroke_rate'] = 0
            timing['stroke_duration'] = 0
            timing['rhythm_consistency'] = 0
        
        # Calculate phase analysis
        self._calculate_phase_analysis(df, timing)
        
        self.analysis_results['timing_analysis'] = timing
    
    def _calculate_phase_analysis(self, df, timing):
        """Calculate drive and recovery phase analysis"""
        leg_angles = (df['left_leg_angle'] + df['right_leg_angle']) / 2
        
        # Find peaks and valleys
        peaks, _ = signal.find_peaks(leg_angles, distance=10, prominence=5)
        valleys, _ = signal.find_peaks(-leg_angles, distance=10, prominence=5)
        
        if len(peaks) > 0 and len(valleys) > 0:
            # Calculate drive and recovery phases
            drive_frames = 0
            recovery_frames = 0
            
            for i in range(min(len(peaks), len(valleys))):
                if valleys[i] < peaks[i]:
                    drive_frames += peaks[i] - valleys[i]
                    if i < len(valleys) - 1:
                        recovery_frames += valleys[i+1] - peaks[i]
            
            total_frames = drive_frames + recovery_frames
            if total_frames > 0:
                timing['drive_ratio'] = drive_frames / total_frames
                timing['recovery_ratio'] = recovery_frames / total_frames
            else:
                timing['drive_ratio'] = 0.5
                timing['recovery_ratio'] = 0.5
        else:
            timing['drive_ratio'] = 0.5
            timing['recovery_ratio'] = 0.5
    
    def _calculate_technique_quality(self, df):
        """Calculate overall technique quality metrics"""
        print("   📊 Calculating technique quality...")
        
        quality = {}
        
        # Range of motion quality
        arm_rom = (df['left_arm_angle'].max() - df['left_arm_angle'].min() + 
                  df['right_arm_angle'].max() - df['right_arm_angle'].min()) / 2
        leg_rom = (df['left_leg_angle'].max() - df['left_leg_angle'].min() + 
                  df['right_leg_angle'].max() - df['right_leg_angle'].min()) / 2
        
        # Ideal ranges (typical rowing values)
        ideal_arm_rom = 120  # degrees
        ideal_leg_rom = 140  # degrees
        
        quality['arm_rom_score'] = min(100, (arm_rom / ideal_arm_rom) * 100)
        quality['leg_rom_score'] = min(100, (leg_rom / ideal_leg_rom) * 100)
        
        # Smoothness score (based on angular velocity consistency)
        arm_velocities = np.abs(np.diff(df['left_arm_angle']))
        leg_velocities = np.abs(np.diff(df['left_leg_angle']))
        
        quality['arm_smoothness'] = 100 - (np.std(arm_velocities) / np.mean(arm_velocities) * 100)
        quality['leg_smoothness'] = 100 - (np.std(leg_velocities) / np.mean(leg_velocities) * 100)
        
        # Overall technique score
        quality['overall_technique_score'] = (
            quality['arm_rom_score'] * 0.2 +
            quality['leg_rom_score'] * 0.3 +
            quality['arm_smoothness'] * 0.2 +
            quality['leg_smoothness'] * 0.3
        )
        
        self.analysis_results['technique_quality'] = quality
    
    def _calculate_force_analysis(self, df):
        """Calculate force-related analysis"""
        print("   📊 Calculating force analysis...")
        
        force_analysis = {}
        
        if self.force_data:
            powers = [f['power'] for f in self.force_data]
            peak_forces = []
            avg_forces = []
            
            for force_entry in self.force_data:
                if force_entry.get('forceplot'):
                    force_curve = force_entry['forceplot']
                    if force_curve:
                        peak_forces.append(max(force_curve))
                        avg_forces.append(np.mean(force_curve))
            
            force_analysis['power_metrics'] = {
                'average_power': np.mean(powers),
                'max_power': np.max(powers),
                'power_std': np.std(powers),
                'power_consistency': 100 - (np.std(powers) / np.mean(powers) * 100)
            }
            
            if peak_forces:
                force_analysis['force_metrics'] = {
                    'average_peak_force': np.mean(peak_forces),
                    'max_peak_force': np.max(peak_forces),
                    'average_force': np.mean(avg_forces),
                    'force_consistency': 100 - (np.std(peak_forces) / np.mean(peak_forces) * 100)
                }
        else:
            force_analysis['power_metrics'] = None
            force_analysis['force_metrics'] = None
        
        self.analysis_results['force_analysis'] = force_analysis
    
    def create_advanced_report(self):
        """Create advanced analysis report"""
        print("\n📊 Creating Advanced Analysis Report...")
        
        # Create output directory
        os.makedirs('synchronized_analysis', exist_ok=True)
        
        # Create comprehensive report
        self._create_comprehensive_report()
        
        # Create advanced visualizations
        self._create_advanced_visualizations()
        
        # Create technique recommendations
        self._create_technique_recommendations()
        
        print("✅ Advanced analysis report created")
    
    def _create_comprehensive_report(self):
        """Create comprehensive text report"""
        report_path = 'synchronized_analysis/comprehensive_rowing_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("🚣‍♂️ COMPREHENSIVE ROWING ANALYSIS REPORT (SYNCHRONIZED DATA)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Analyzed: {len(self.pose_data)}\n")
            f.write(f"Force Data Points: {len(self.force_data)}\n\n")
            
            # Joint Kinematics
            if 'joint_kinematics' in self.analysis_results:
                f.write("JOINT KINEMATICS ANALYSIS\n")
                f.write("-" * 40 + "\n")
                kinematics = self.analysis_results['joint_kinematics']
                
                for joint, metrics in kinematics.items():
                    f.write(f"\n{joint.upper()} JOINT:\n")
                    f.write(f"  Range of Motion: {metrics['range_of_motion']:.1f}°\n")
                    f.write(f"  Mean Angle: {metrics['mean_angle']:.1f}°\n")
                    f.write(f"  Angular Velocity: {metrics['angular_velocity']:.2f}°/frame\n")
                    f.write(f"  Angle Range: {metrics['min_angle']:.1f}° - {metrics['max_angle']:.1f}°\n")
                f.write("\n")
            
            # Symmetry Analysis
            if 'symmetry_analysis' in self.analysis_results:
                f.write("SYMMETRY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                symmetry = self.analysis_results['symmetry_analysis']
                
                f.write(f"Overall Symmetry Score: {symmetry['overall_symmetry_score']:.1f}%\n\n")
                
                f.write("Arm Symmetry:\n")
                f.write(f"  Mean Difference: {symmetry['arm_symmetry']['mean_difference']:.1f}°\n")
                f.write(f"  Correlation: {symmetry['arm_symmetry']['correlation']:.3f}\n")
                f.write(f"  Symmetry Index: {symmetry['arm_symmetry']['symmetry_index']:.1f}%\n\n")
                
                f.write("Leg Symmetry:\n")
                f.write(f"  Mean Difference: {symmetry['leg_symmetry']['mean_difference']:.1f}°\n")
                f.write(f"  Correlation: {symmetry['leg_symmetry']['correlation']:.3f}\n")
                f.write(f"  Symmetry Index: {symmetry['leg_symmetry']['symmetry_index']:.1f}%\n\n")
            
            # Timing Analysis
            if 'timing_analysis' in self.analysis_results:
                f.write("TIMING ANALYSIS\n")
                f.write("-" * 40 + "\n")
                timing = self.analysis_results['timing_analysis']
                
                f.write(f"Stroke Rate: {timing['stroke_rate']:.1f} spm\n")
                f.write(f"Stroke Duration: {timing['stroke_duration']:.2f} seconds\n")
                f.write(f"Rhythm Consistency: {timing['rhythm_consistency']:.1f}%\n")
                f.write(f"Drive Ratio: {timing['drive_ratio']:.2%}\n")
                f.write(f"Recovery Ratio: {timing['recovery_ratio']:.2%}\n\n")
            
            # Technique Quality
            if 'technique_quality' in self.analysis_results:
                f.write("TECHNIQUE QUALITY ASSESSMENT\n")
                f.write("-" * 40 + "\n")
                quality = self.analysis_results['technique_quality']
                
                f.write(f"Overall Technique Score: {quality['overall_technique_score']:.1f}%\n")
                f.write(f"Arm ROM Score: {quality['arm_rom_score']:.1f}%\n")
                f.write(f"Leg ROM Score: {quality['leg_rom_score']:.1f}%\n")
                f.write(f"Arm Smoothness: {quality['arm_smoothness']:.1f}%\n")
                f.write(f"Leg Smoothness: {quality['leg_smoothness']:.1f}%\n\n")
            
            # Force Analysis
            if 'force_analysis' in self.analysis_results and self.analysis_results['force_analysis']['power_metrics']:
                f.write("FORCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                force = self.analysis_results['force_analysis']
                
                if force['power_metrics']:
                    f.write(f"Average Power: {force['power_metrics']['average_power']:.1f}W\n")
                    f.write(f"Maximum Power: {force['power_metrics']['max_power']:.1f}W\n")
                    f.write(f"Power Consistency: {force['power_metrics']['power_consistency']:.1f}%\n")
                
                if force['force_metrics']:
                    f.write(f"Average Peak Force: {force['force_metrics']['average_peak_force']:.1f}\n")
                    f.write(f"Maximum Peak Force: {force['force_metrics']['max_peak_force']:.1f}\n")
                    f.write(f"Force Consistency: {force['force_metrics']['force_consistency']:.1f}%\n")
        
        print(f"   📋 Comprehensive report: {report_path}")
    
    def _create_advanced_visualizations(self):
        """Create advanced visualization plots"""
        if not self.pose_data:
            return
        
        df = pd.DataFrame(self.pose_data)
        df = df.dropna(subset=['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle'])
        
        # Set up plotting style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        # Create comprehensive analysis figure
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('🚣‍♂️ Advanced Rowing Biomechanical Analysis (Synchronized Data)', fontsize=24, color='white')
        
        # 1. Joint Angles Over Time
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(df['timestamp'], df['left_arm_angle'], 'lime', label='Left Arm', linewidth=2)
        ax1.plot(df['timestamp'], df['right_arm_angle'], 'cyan', label='Right Arm', linewidth=2)
        ax1.set_title('Arm Angles Over Time', color='white', fontsize=14)
        ax1.set_xlabel('Time (s)', color='white')
        ax1.set_ylabel('Angle (degrees)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Leg Angles Over Time
        ax2 = plt.subplot(3, 4, 2)
        ax2.plot(df['timestamp'], df['left_leg_angle'], 'lime', label='Left Leg', linewidth=2)
        ax2.plot(df['timestamp'], df['right_leg_angle'], 'cyan', label='Right Leg', linewidth=2)
        ax2.set_title('Leg Angles Over Time', color='white', fontsize=14)
        ax2.set_xlabel('Time (s)', color='white')
        ax2.set_ylabel('Angle (degrees)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Force and Power Over Time
        ax3 = plt.subplot(3, 4, 3)
        if self.force_data:
            times = [f['elapsed_s'] for f in self.force_data]
            powers = [f['power'] for f in self.force_data]
            ax3.plot(times, powers, 'yellow', linewidth=2, marker='o', markersize=4)
            ax3.set_title('Power Output Over Time', color='white', fontsize=14)
            ax3.set_xlabel('Time (s)', color='white')
            ax3.set_ylabel('Power (W)', color='white')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Force Data\nNot Available', ha='center', va='center', 
                    transform=ax3.transAxes, color='white', fontsize=12)
            ax3.set_title('Force Analysis', color='white', fontsize=14)
        
        # 4. Range of Motion Comparison
        ax4 = plt.subplot(3, 4, 4)
        joints = ['Left Arm', 'Right Arm', 'Left Leg', 'Right Leg']
        ranges = [
            df['left_arm_angle'].max() - df['left_arm_angle'].min(),
            df['right_arm_angle'].max() - df['right_arm_angle'].min(),
            df['left_leg_angle'].max() - df['left_leg_angle'].min(),
            df['right_leg_angle'].max() - df['right_leg_angle'].min()
        ]
        colors = ['lime', 'cyan', 'lime', 'cyan']
        bars = ax4.bar(joints, ranges, color=colors, alpha=0.7)
        ax4.set_title('Range of Motion', color='white', fontsize=14)
        ax4.set_ylabel('Angle Range (degrees)', color='white')
        ax4.tick_params(axis='x', rotation=45, colors='white')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, ranges):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}°', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 5. Symmetry Analysis
        ax5 = plt.subplot(3, 4, 5)
        arm_diff = df['left_arm_angle'] - df['right_arm_angle']
        leg_diff = df['left_leg_angle'] - df['right_leg_angle']
        ax5.plot(df['timestamp'], arm_diff, 'orange', label='Arm Difference', linewidth=2)
        ax5.plot(df['timestamp'], leg_diff, 'red', label='Leg Difference', linewidth=2)
        ax5.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax5.set_title('Bilateral Symmetry', color='white', fontsize=14)
        ax5.set_xlabel('Time (s)', color='white')
        ax5.set_ylabel('Angle Difference (degrees)', color='white')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Force Curves
        ax6 = plt.subplot(3, 4, 6)
        if self.force_data:
            # Show a few sample force curves
            for i, force_entry in enumerate(self.force_data[:5]):
                if force_entry.get('forceplot'):
                    force_curve = force_entry['forceplot']
                    if force_curve:
                        x = np.arange(len(force_curve))
                        ax6.plot(x, force_curve, alpha=0.7, linewidth=1)
            ax6.set_title('Sample Force Curves', color='white', fontsize=14)
            ax6.set_xlabel('Stroke Position', color='white')
            ax6.set_ylabel('Force', color='white')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Force Curves\nNot Available', ha='center', va='center', 
                    transform=ax6.transAxes, color='white', fontsize=12)
            ax6.set_title('Force Curves', color='white', fontsize=14)
        
        # 7. Technique Quality Scores
        ax7 = plt.subplot(3, 4, 7)
        if 'technique_quality' in self.analysis_results:
            quality = self.analysis_results['technique_quality']
            scores = [quality['arm_rom_score'], quality['leg_rom_score'], 
                     quality['arm_smoothness'], quality['leg_smoothness']]
            labels = ['Arm ROM', 'Leg ROM', 'Arm Smooth', 'Leg Smooth']
            
            bars = ax7.bar(labels, scores, color=['lime', 'cyan', 'yellow', 'magenta'], alpha=0.7)
            ax7.set_title('Technique Quality Scores', color='white', fontsize=14)
            ax7.set_ylabel('Score (%)', color='white')
            ax7.tick_params(axis='x', rotation=45, colors='white')
            ax7.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, scores):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 8. Symmetry Scores
        ax8 = plt.subplot(3, 4, 8)
        if 'symmetry_analysis' in self.analysis_results:
            symmetry = self.analysis_results['symmetry_analysis']
            scores = [symmetry['arm_symmetry']['symmetry_index'], 
                     symmetry['leg_symmetry']['symmetry_index']]
            labels = ['Arm Symmetry', 'Leg Symmetry']
            
            bars = ax8.bar(labels, scores, color=['orange', 'red'], alpha=0.7)
            ax8.set_title('Symmetry Scores', color='white', fontsize=14)
            ax8.set_ylabel('Symmetry Index (%)', color='white')
            ax8.tick_params(axis='x', rotation=45, colors='white')
            ax8.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, scores):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 9. Performance Summary
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')
        
        summary_text = "PERFORMANCE SUMMARY\n\n"
        
        if 'technique_quality' in self.analysis_results:
            quality = self.analysis_results['technique_quality']
            summary_text += f"Overall Score: {quality['overall_technique_score']:.1f}%\n"
        
        if 'symmetry_analysis' in self.analysis_results:
            symmetry = self.analysis_results['symmetry_analysis']
            summary_text += f"Symmetry Score: {symmetry['overall_symmetry_score']:.1f}%\n"
        
        if 'timing_analysis' in self.analysis_results:
            timing = self.analysis_results['timing_analysis']
            summary_text += f"Stroke Rate: {timing['stroke_rate']:.1f} spm\n"
            summary_text += f"Rhythm: {timing['rhythm_consistency']:.1f}%\n"
        
        if self.force_data:
            powers = [f['power'] for f in self.force_data]
            summary_text += f"Avg Power: {np.mean(powers):.1f}W\n"
            summary_text += f"Max Power: {np.max(powers):.1f}W\n"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
                color='white', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # 10. Data Quality Assessment
        ax10 = plt.subplot(3, 4, 10)
        ax10.axis('off')
        
        quality_text = "DATA QUALITY ASSESSMENT\n\n"
        quality_text += f"Pose Frames: {len(self.pose_data)}\n"
        quality_text += f"Force Points: {len(self.force_data)}\n"
        quality_text += "Force-Pose Sync: Available\n"
        
        # Calculate data completeness
        valid_frames = len([f for f in self.pose_data if f.get('left_arm_angle') is not None])
        completeness = (valid_frames / len(self.pose_data)) * 100
        quality_text += f"Data Completeness: {completeness:.1f}%\n"
        
        if completeness > 90:
            quality_text += "Quality: Excellent\n"
        elif completeness > 75:
            quality_text += "Quality: Good\n"
        else:
            quality_text += "Quality: Fair\n"
        
        ax10.text(0.1, 0.9, quality_text, transform=ax10.transAxes, 
                color='white', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('synchronized_analysis/synchronized_rowing_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print("   📊 Synchronized analysis plot: synchronized_analysis/synchronized_rowing_analysis.png")
    
    def _create_technique_recommendations(self):
        """Create detailed technique recommendations"""
        recommendations_path = 'synchronized_analysis/technique_recommendations.txt'
        
        with open(recommendations_path, 'w') as f:
            f.write("🚣‍♂️ ROWING TECHNIQUE RECOMMENDATIONS (SYNCHRONIZED DATA)\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall assessment
            if 'technique_quality' in self.analysis_results:
                quality = self.analysis_results['technique_quality']
                overall_score = quality['overall_technique_score']
                
                f.write(f"OVERALL TECHNIQUE SCORE: {overall_score:.1f}%\n")
                f.write("-" * 40 + "\n")
                
                if overall_score >= 90:
                    f.write("Excellent technique! Your rowing form is very good.\n\n")
                elif overall_score >= 80:
                    f.write("Good technique with room for minor improvements.\n\n")
                elif overall_score >= 70:
                    f.write("Fair technique with several areas for improvement.\n\n")
                else:
                    f.write("Technique needs significant improvement.\n\n")
            
            # Specific recommendations
            f.write("SPECIFIC RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            
            if 'symmetry_analysis' in self.analysis_results:
                symmetry = self.analysis_results['symmetry_analysis']
                
                if symmetry['arm_symmetry']['symmetry_index'] < 90:
                    f.write("• ARM SYMMETRY: Focus on equal arm movement\n")
                    f.write("  - Practice with a mirror to check symmetry\n")
                    f.write("  - Ensure both arms move together\n")
                    f.write("  - Check for dominant side compensation\n\n")
                
                if symmetry['leg_symmetry']['symmetry_index'] < 90:
                    f.write("• LEG SYMMETRY: Focus on equal leg drive\n")
                    f.write("  - Ensure both legs push equally\n")
                    f.write("  - Check foot positioning on footrests\n")
                    f.write("  - Work on leg strength balance\n\n")
            
            if 'technique_quality' in self.analysis_results:
                quality = self.analysis_results['technique_quality']
                
                if quality['arm_rom_score'] < 80:
                    f.write("• ARM RANGE OF MOTION: Increase arm movement\n")
                    f.write("  - Focus on full arm extension and compression\n")
                    f.write("  - Practice arm-only rowing drills\n")
                    f.write("  - Ensure proper catch and finish positions\n\n")
                
                if quality['leg_rom_score'] < 80:
                    f.write("• LEG RANGE OF MOTION: Increase leg drive\n")
                    f.write("  - Focus on full leg compression and extension\n")
                    f.write("  - Practice leg-only rowing drills\n")
                    f.write("  - Ensure proper catch and finish positions\n\n")
                
                if quality['arm_smoothness'] < 80:
                    f.write("• ARM SMOOTHNESS: Improve arm movement flow\n")
                    f.write("  - Focus on smooth, controlled arm movement\n")
                    f.write("  - Avoid jerky or abrupt arm actions\n")
                    f.write("  - Practice with lighter resistance\n\n")
                
                if quality['leg_smoothness'] < 80:
                    f.write("• LEG SMOOTHNESS: Improve leg movement flow\n")
                    f.write("  - Focus on smooth, controlled leg drive\n")
                    f.write("  - Avoid jerky or abrupt leg actions\n")
                    f.write("  - Practice with lighter resistance\n\n")
            
            if 'timing_analysis' in self.analysis_results:
                timing = self.analysis_results['timing_analysis']
                
                if timing['rhythm_consistency'] < 80:
                    f.write("• RHYTHM CONSISTENCY: Improve stroke rhythm\n")
                    f.write("  - Practice with a metronome\n")
                    f.write("  - Focus on consistent stroke timing\n")
                    f.write("  - Count strokes to maintain rhythm\n\n")
                
                if timing['drive_ratio'] < 0.3:
                    f.write("• DRIVE PHASE: Increase drive duration\n")
                    f.write("  - Focus on accelerating through the drive\n")
                    f.write("  - Maintain connection between legs, body, and arms\n")
                    f.write("  - Don't rush the drive phase\n\n")
                elif timing['drive_ratio'] > 0.6:
                    f.write("• RECOVERY PHASE: Increase recovery duration\n")
                    f.write("  - Focus on controlled, relaxed recovery\n")
                    f.write("  - Don't rush back to the catch\n")
                    f.write("  - Use recovery for preparation\n\n")
            
            f.write("GENERAL IMPROVEMENT TIPS:\n")
            f.write("-" * 30 + "\n")
            f.write("• Practice regularly with video analysis\n")
            f.write("• Focus on one aspect at a time\n")
            f.write("• Work with a coach for personalized feedback\n")
            f.write("• Use drills to isolate specific movements\n")
            f.write("• Maintain consistent practice schedule\n")
            f.write("• Record sessions for progress tracking\n")
        
        print(f"   📋 Technique recommendations: {recommendations_path}")

def main():
    parser = argparse.ArgumentParser(description="Run synchronized rowing analysis")
    parser.add_argument("--sync-json", required=True, help="Path to synchronized data JSON file")
    
    args = parser.parse_args()
    
    print("🚣‍♂️ Synchronized Rowing Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SynchronizedRowingAnalyzer()
    
    # Load synchronized data
    if not analyzer.load_synchronized_data(args.sync_json):
        print("❌ Cannot proceed without synchronized data")
        return
    
    # Perform analysis
    analyzer.analyze_rowing_technique()
    
    # Create advanced report
    analyzer.create_advanced_report()
    
    print("\n🎉 Synchronized rowing analysis complete!")
    print("📁 Check the 'synchronized_analysis/' directory for results")

if __name__ == "__main__":
    main()
