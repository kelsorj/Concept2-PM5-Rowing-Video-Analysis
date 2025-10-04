#!/usr/bin/env python3
"""
Advanced Rowing Analysis System
Provides comprehensive biomechanical analysis with robust stroke cycle detection
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

class AdvancedRowingAnalyzer:
    def __init__(self):
        self.pose_data = None
        self.force_data = None
        self.analysis_results = {}
        self.plot_scale = 0.75
        self.time_offset_s = 0.0
        
    def load_data(self):
        """Load pose and force data"""
        # Load pose data
        pose_files = glob.glob("rowing_pose_data_*.json")
        if not pose_files:
            print("❌ No pose data files found")
            return False
        
        latest_pose_file = max(pose_files, key=os.path.getctime)
        print(f"📊 Loading pose data: {latest_pose_file}")
        
        with open(latest_pose_file, 'r') as f:
            self.pose_data = json.load(f)
        
        # Load force data
        force_files = glob.glob("pm5_py3row_parsed_*.csv")
        if force_files:
            latest_force_file = max(force_files, key=os.path.getctime)
            print(f"📊 Loading force data: {latest_force_file}")
            
            df = pd.read_csv(latest_force_file)
            force_data = []
            for idx, row in df.iterrows():
                if pd.notna(row['forceplot']) and row['forceplot'] != '':
                    try:
                        force_curve = json.loads(row['forceplot'])
                        if force_curve:
                            force_data.append({
                                'elapsed_s': row['elapsed_s'],
                                'force_curve': force_curve,
                                'power': row['avg_power_w'],
                                'spm': row['spm'],
                                'distance': row['distance_m'],
                                'timestamp_iso': row.get('timestamp_iso'),
                                'timestamp_dt': datetime.fromisoformat(row['timestamp_iso']) if pd.notna(row.get('timestamp_iso')) else None
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            self.force_data = force_data
            print(f"   Found {len(self.force_data)} complete force curves")
        
        print(f"   Found {len(self.pose_data)} pose frames")
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
                if force_entry['force_curve']:
                    peak_forces.append(max(force_entry['force_curve']))
                    avg_forces.append(np.mean(force_entry['force_curve']))
            
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
        os.makedirs('advanced_analysis', exist_ok=True)
        
        # Create comprehensive report
        self._create_comprehensive_report()
        
        # Create advanced visualizations
        self._create_advanced_visualizations()
        
        # Create technique recommendations
        self._create_technique_recommendations()
        
        # Create analysis video
        self._create_analysis_video()
        
        print("✅ Advanced analysis report created")
    
    def _create_comprehensive_report(self):
        """Create comprehensive text report"""
        report_path = 'advanced_analysis/comprehensive_rowing_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("🚣‍♂️ COMPREHENSIVE ROWING ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Analyzed: {len(self.pose_data)}\n\n")
            
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
        fig.suptitle('🚣‍♂️ Advanced Rowing Biomechanical Analysis', fontsize=24, color='white')
        
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
        
        # 3. Symmetry Analysis
        ax3 = plt.subplot(3, 4, 3)
        arm_diff = df['left_arm_angle'] - df['right_arm_angle']
        leg_diff = df['left_leg_angle'] - df['right_leg_angle']
        ax3.plot(df['timestamp'], arm_diff, 'orange', label='Arm Difference', linewidth=2)
        ax3.plot(df['timestamp'], leg_diff, 'red', label='Leg Difference', linewidth=2)
        ax3.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax3.set_title('Bilateral Symmetry', color='white', fontsize=14)
        ax3.set_xlabel('Time (s)', color='white')
        ax3.set_ylabel('Angle Difference (degrees)', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
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
        
        # 5. Angular Velocity Analysis
        ax5 = plt.subplot(3, 4, 5)
        arm_vel = np.abs(np.diff(df['left_arm_angle']))
        leg_vel = np.abs(np.diff(df['left_leg_angle']))
        ax5.plot(df['timestamp'][1:], arm_vel, 'yellow', label='Arm Velocity', linewidth=2)
        ax5.plot(df['timestamp'][1:], leg_vel, 'magenta', label='Leg Velocity', linewidth=2)
        ax5.set_title('Angular Velocity', color='white', fontsize=14)
        ax5.set_xlabel('Time (s)', color='white')
        ax5.set_ylabel('Angular Velocity (degrees/frame)', color='white')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Correlation Matrix
        ax6 = plt.subplot(3, 4, 6)
        corr_data = df[['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle']].corr()
        im = ax6.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(corr_data.columns)))
        ax6.set_yticks(range(len(corr_data.columns)))
        ax6.set_xticklabels(['L Arm', 'R Arm', 'L Leg', 'R Leg'], color='white')
        ax6.set_yticklabels(['L Arm', 'R Arm', 'L Leg', 'R Leg'], color='white')
        ax6.set_title('Joint Correlation Matrix', color='white', fontsize=14)
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                ax6.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
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
        
        # 8. Force Analysis (if available)
        ax8 = plt.subplot(3, 4, 8)
        if self.force_data:
            powers = [f['power'] for f in self.force_data]
            times = [f['elapsed_s'] for f in self.force_data]
            ax8.plot(times, powers, 'yellow', linewidth=2, marker='o', markersize=4)
            ax8.set_title('Power Output Over Time', color='white', fontsize=14)
            ax8.set_xlabel('Time (s)', color='white')
            ax8.set_ylabel('Power (W)', color='white')
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Force Data\nNot Available', ha='center', va='center', 
                    transform=ax8.transAxes, color='white', fontsize=12)
            ax8.set_title('Force Analysis', color='white', fontsize=14)
        
        # 9. Symmetry Scores
        ax9 = plt.subplot(3, 4, 9)
        if 'symmetry_analysis' in self.analysis_results:
            symmetry = self.analysis_results['symmetry_analysis']
            scores = [symmetry['arm_symmetry']['symmetry_index'], 
                     symmetry['leg_symmetry']['symmetry_index']]
            labels = ['Arm Symmetry', 'Leg Symmetry']
            
            bars = ax9.bar(labels, scores, color=['orange', 'red'], alpha=0.7)
            ax9.set_title('Symmetry Scores', color='white', fontsize=14)
            ax9.set_ylabel('Symmetry Index (%)', color='white')
            ax9.tick_params(axis='x', rotation=45, colors='white')
            ax9.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, scores):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 10. Performance Summary
        ax10 = plt.subplot(3, 4, 10)
        ax10.axis('off')
        
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
        
        ax10.text(0.1, 0.9, summary_text, transform=ax10.transAxes, 
                color='white', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # 11. Technique Recommendations
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        recommendations = "TECHNIQUE RECOMMENDATIONS\n\n"
        
        if 'symmetry_analysis' in self.analysis_results:
            symmetry = self.analysis_results['symmetry_analysis']
            if symmetry['arm_symmetry']['symmetry_index'] < 90:
                recommendations += "• Focus on arm symmetry\n"
            if symmetry['leg_symmetry']['symmetry_index'] < 90:
                recommendations += "• Focus on leg symmetry\n"
        
        if 'technique_quality' in self.analysis_results:
            quality = self.analysis_results['technique_quality']
            if quality['arm_rom_score'] < 80:
                recommendations += "• Increase arm range of motion\n"
            if quality['leg_rom_score'] < 80:
                recommendations += "• Increase leg range of motion\n"
            if quality['arm_smoothness'] < 80:
                recommendations += "• Improve arm movement smoothness\n"
            if quality['leg_smoothness'] < 80:
                recommendations += "• Improve leg movement smoothness\n"
        
        if not recommendations.endswith("\n\n"):
            recommendations += "• Technique looks good!\n"
        
        ax11.text(0.1, 0.9, recommendations, transform=ax11.transAxes, 
                color='white', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))
        
        # 12. Data Quality Assessment
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        quality_text = "DATA QUALITY ASSESSMENT\n\n"
        quality_text += f"Pose Frames: {len(self.pose_data)}\n"
        
        if self.force_data:
            quality_text += f"Force Curves: {len(self.force_data)}\n"
            quality_text += "Force-Pose Sync: Available\n"
        else:
            quality_text += "Force Curves: Not Available\n"
            quality_text += "Force-Pose Sync: Not Available\n"
        
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
        
        ax12.text(0.1, 0.9, quality_text, transform=ax12.transAxes, 
                color='white', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('advanced_analysis/advanced_rowing_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print("   📊 Advanced analysis plot: advanced_analysis/advanced_rowing_analysis.png")
    
    def _create_technique_recommendations(self):
        """Create detailed technique recommendations"""
        recommendations_path = 'advanced_analysis/technique_recommendations.txt'
        
        with open(recommendations_path, 'w') as f:
            f.write("🚣‍♂️ ROWING TECHNIQUE RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            
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
    
    def _create_analysis_video(self):
        """Create analysis video with animated force plots and body angles overlaid"""
        print("\n🎬 Creating Analysis Video with Animated Overlays...")
        
        if not self.pose_data:
            print("❌ No pose data available for video creation")
            return
        
        # Find the video file
        video_files = glob.glob("runs/pose/predict/*.mp4")
        if not video_files:
            print("❌ No video files found in runs/pose/predict/")
            return
        
        video_path = video_files[0]
        print(f"📹 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Derive absolute start time from filename when available (HHMMSS-HHMMSS)
        video_start_dt = None
        m = re.search(r"(\d{6})-(\d{6})", os.path.basename(video_path))
        if m:
            hhmmss = m.group(1)
            # Anchor to date from force data if available
            base_dt = None
            if self.force_data and self.force_data[0].get('timestamp_dt') is not None:
                base_dt = self.force_data[0]['timestamp_dt']
            if base_dt is None and self.force_data:
                # fall back to now if timestamps missing
                base_dt = datetime.now()
            if base_dt is not None:
                h = int(hhmmss[0:2]); mi = int(hhmmss[2:4]); s = int(hhmmss[4:6])
                video_start_dt = datetime(base_dt.year, base_dt.month, base_dt.day, h, mi, s)

        # If filename has a clip time window, filter force data to that window and align times
        self._compute_clip_window_and_filter_force(video_path)
        
        # Create output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'advanced_analysis/advanced_rowing_analysis_{timestamp}.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"📊 Processing frames with animated analysis overlays...")
        
        frame_count = 0
        overlays_added = 0
        power_overlays = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get pose data for this frame
            if frame_count <= len(self.pose_data):
                pose_frame = self.pose_data[frame_count - 1]
                # Base elapsed time from pose export
                elapsed_s = float(pose_frame.get('timestamp', 0))
                # If we have video absolute start time and force timestamps, compute absolute frame time
                if video_start_dt is not None:
                    frame_abs_dt = video_start_dt + timedelta(seconds=(frame_count - 1) / max(1, fps))
                    # Apply user-provided constant offset for sync
                    if getattr(self, 'time_offset_s', 0.0) != 0.0:
                        frame_abs_dt = frame_abs_dt + timedelta(seconds=self.time_offset_s)
                    # Replace elapsed to clip-relative seconds using force clip start when known
                    # Find clip start from first filtered force entry
                    if self.force_data and self.force_data[0].get('clip_elapsed_s') is not None and self.force_data[0].get('timestamp_dt') is not None:
                        clip_start_dt = self.force_data[0]['timestamp_dt'] - timedelta(seconds=self.force_data[0]['clip_elapsed_s'])
                        elapsed_s = (frame_abs_dt - clip_start_dt).total_seconds()
                
                # Create enhanced joint angles display
                angles_display = self._create_enhanced_joint_angles_display(pose_frame, frame_count, elapsed_s)
                
                # Create animated power curve display (if available)
                frame_abs_dt = None
                if 'video_start_dt' in locals() and video_start_dt is not None:
                    frame_abs_dt = video_start_dt + timedelta(seconds=(frame_count - 1) / max(1, fps))
                power_display = self._create_animated_power_curve_display(elapsed_s, frame_abs_dt)
                
                # Create analysis summary display
                summary_display = self._create_analysis_summary_display(pose_frame, elapsed_s)
                
                # Create angle trend chart
                angle_chart = self._create_angle_trend_chart(frame_count - 1)
                
                # Draw joint-level angle labels directly on the frame at each joint
                self._draw_joint_angles_on_frame(frame, pose_frame)
                
                # Overlay displays on frame
                if angles_display is not None:
                    self._overlay_display(frame, angles_display, 10, 10)
                    overlays_added += 1
                
                if power_display is not None:
                    ph, pw = power_display.shape[:2]
                    px = max(0, width - pw - 10)
                    py = 10
                    if px + pw <= width and py + ph <= height:
                        self._overlay_display(frame, power_display, px, py)
                        power_overlays += 1
                
                if angle_chart is not None:
                    ch, cw = angle_chart.shape[:2]
                    cx = max(0, width - cw - 10)
                    cy = max(0, height - ch - 10)
                    if cx + cw <= width and cy + ch <= height:
                        self._overlay_display(frame, angle_chart, cx, cy)
                
                if summary_display is not None:
                    self._overlay_display(frame, summary_display, 10, height - 200)
                
                # Add frame info
                info_text = f"Frame {frame_count}/{total_frames} | Time: {elapsed_s:.1f}s | Advanced Analysis"
                cv2.putText(frame, info_text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"\n🎉 Analysis video complete!")
        print(f"   📹 Output video: {output_path}")
        print(f"   📊 Total frames: {frame_count}")
        print(f"   📈 Overlays added: {overlays_added}")
        print(f"   📊 Overlay rate: {(overlays_added/frame_count)*100:.1f}%")
        print(f"   🔋 Power plot overlays: {power_overlays}")
    
    def _create_joint_angles_display(self, pose_frame, frame_num, elapsed_s):
        """Create joint angles display overlay"""
        # Create a semi-transparent background
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Define colors
        colors = {
            'white': (255, 255, 255),
            'lime': (0, 255, 0),
            'cyan': (255, 255, 0),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'magenta': (255, 0, 255)
        }
        
        # Title
        cv2.putText(img, "JOINT ANGLES ANALYSIS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['white'], 2)
        
        # Frame info
        cv2.putText(img, f"Frame: {frame_num}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
        cv2.putText(img, f"Time: {elapsed_s:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
        
        y_pos = 95
        line_height = 25
        
        # Arm angles
        cv2.putText(img, "ARM ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        if pose_frame.get('left_arm_angle') is not None:
            left_arm = pose_frame['left_arm_angle']
            cv2.putText(img, f"  L Arm: {left_arm:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
            y_pos += line_height
        
        if pose_frame.get('right_arm_angle') is not None:
            right_arm = pose_frame['right_arm_angle']
            cv2.putText(img, f"  R Arm: {right_arm:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
            y_pos += line_height
        
        # Leg angles
        cv2.putText(img, "LEG ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        if pose_frame.get('left_leg_angle') is not None:
            left_leg = pose_frame['left_leg_angle']
            cv2.putText(img, f"  L Leg: {left_leg:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
            y_pos += line_height
        
        if pose_frame.get('right_leg_angle') is not None:
            right_leg = pose_frame['right_leg_angle']
            cv2.putText(img, f"  R Leg: {right_leg:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
            y_pos += line_height
        
        # Torso angle
        cv2.putText(img, "TORSO:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        if pose_frame.get('torso_lean_angle') is not None:
            torso = pose_frame['torso_lean_angle']
            cv2.putText(img, f"  Lean: {torso:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
        
        return img
    
    def _create_enhanced_joint_angles_display(self, pose_frame, frame_num, elapsed_s):
        """Create enhanced joint angles display with more information"""
        # Create a semi-transparent background
        img = np.zeros((350, 400, 3), dtype=np.uint8)
        
        # Define colors
        colors = {
            'white': (255, 255, 255),
            'lime': (0, 255, 0),
            'cyan': (255, 255, 0),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'magenta': (255, 0, 255)
        }
        
        # Title
        cv2.putText(img, "ROWING ANALYSIS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['white'], 2)
        
        # Frame info
        cv2.putText(img, f"Frame: {frame_num}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
        cv2.putText(img, f"Time: {elapsed_s:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
        
        y_pos = 95
        line_height = 25
        
        # Arm angles
        cv2.putText(img, "ARM ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        if pose_frame.get('left_arm_angle') is not None:
            left_arm = pose_frame['left_arm_angle']
            cv2.putText(img, f"  L Arm: {left_arm:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
            y_pos += line_height
        
        if pose_frame.get('right_arm_angle') is not None:
            right_arm = pose_frame['right_arm_angle']
            cv2.putText(img, f"  R Arm: {right_arm:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
            y_pos += line_height
        
        # Leg angles
        cv2.putText(img, "LEG ANGLES:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        if pose_frame.get('left_leg_angle') is not None:
            left_leg = pose_frame['left_leg_angle']
            cv2.putText(img, f"  L Leg: {left_leg:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
            y_pos += line_height
        
        if pose_frame.get('right_leg_angle') is not None:
            right_leg = pose_frame['right_leg_angle']
            cv2.putText(img, f"  R Leg: {right_leg:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
            y_pos += line_height
        
        # Torso angle
        cv2.putText(img, "TORSO:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        y_pos += line_height
        
        if pose_frame.get('torso_lean_angle') is not None:
            torso = pose_frame['torso_lean_angle']
            cv2.putText(img, f"  Lean: {torso:.1f}°", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
        
        return img
    
    def _create_animated_power_curve_display(self, elapsed_s, frame_abs_dt=None):
        """Create animated power curve display with current position indicator"""
        if not self.force_data:
            return None
        
        # Use a bracketing-segment approach so we never show a static curve
        def ts_at(i):
            e = self.force_data[i]
            return e.get('clip_elapsed_s', e.get('elapsed_s', 0.0))
        n = len(self.force_data)
        if n < 2:
            return None
        # Find the nearest index by time
        closest_idx = int(np.argmin([abs(elapsed_s - ts_at(i)) for i in range(n)]))
        t_curr = float(elapsed_s)
        left_idx, right_idx = None, None
        if t_curr >= ts_at(closest_idx) and closest_idx + 1 < n:
            left_idx, right_idx = closest_idx, closest_idx + 1
        elif t_curr < ts_at(closest_idx) and closest_idx - 1 >= 0:
            left_idx, right_idx = closest_idx - 1, closest_idx
        else:
            return None  # no valid segment yet; hide the plot

        t_left = ts_at(left_idx)
        t_right = ts_at(right_idx)
        denom = max(1e-3, (t_right - t_left))
        phase = float(np.clip((t_curr - t_left) / denom, 0.0, 1.0))
        closest_force = self.force_data[left_idx]
        if not closest_force or not closest_force.get('force_curve'):
            return None
        num_pts = len(closest_force['force_curve'])
        current_idx = int(np.clip(round(phase * (num_pts - 1)), 0, num_pts - 1))

        # Delegate to the overlay-style plot for identical look and metadata, with animated dot
        force_curve = closest_force['force_curve']
        spm = closest_force.get('spm')
        distance = closest_force.get('distance')
        plot_elapsed = closest_force.get('clip_elapsed_s', closest_force.get('elapsed_s', 0.0))
        return self._create_force_curve_plot(force_curve, closest_force['power'], spm, distance, plot_elapsed, current_idx, frame_abs_dt)

    def _create_angle_trend_chart(self, current_frame_idx):
        """Create a mini chart showing angle trends"""
        if current_frame_idx < 10 or not self.pose_data:
            return None
        
        # Get recent angle data
        recent_frames = self.pose_data[max(0, current_frame_idx-30):current_frame_idx+1]
        
        # Extract angles
        left_arm_angles = [f.get('left_arm_angle') for f in recent_frames if f.get('left_arm_angle') is not None]
        right_arm_angles = [f.get('right_arm_angle') for f in recent_frames if f.get('right_arm_angle') is not None]
        
        if not left_arm_angles or not right_arm_angles:
            return None
        
        # Ensure both arrays have the same length
        min_length = min(len(left_arm_angles), len(right_arm_angles))
        left_arm_angles = left_arm_angles[:min_length]
        right_arm_angles = right_arm_angles[:min_length]
        
        # Create mini chart
        fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        x = np.arange(len(left_arm_angles))
        ax.plot(x, left_arm_angles, 'lime', linewidth=2, label='L Arm')
        ax.plot(x, right_arm_angles, 'cyan', linewidth=2, label='R Arm')
        
        ax.set_title('Arm Angles Trend', color='white', fontsize=10)
        ax.set_ylabel('Degrees', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=6)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3, color='gray')
        
        plt.tight_layout()
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    def _create_power_curve_display(self, elapsed_s, frame_abs_dt=None):
        """Create power curve display overlay"""
        if not self.force_data:
            return None
        
        # Use a bracketing-segment approach so we never show a static curve
        def ts_at(i):
            e = self.force_data[i]
            return e.get('clip_elapsed_s', e.get('elapsed_s', 0.0))
        n = len(self.force_data)
        if n < 2:
            return None
        # Find the nearest index by time
        closest_idx = int(np.argmin([abs(elapsed_s - ts_at(i)) for i in range(n)]))
        t_curr = float(elapsed_s)
        left_idx, right_idx = None, None
        if t_curr >= ts_at(closest_idx) and closest_idx + 1 < n:
            left_idx, right_idx = closest_idx, closest_idx + 1
        elif t_curr < ts_at(closest_idx) and closest_idx - 1 >= 0:
            left_idx, right_idx = closest_idx - 1, closest_idx
        else:
            return None  # no valid segment yet; hide the plot

        t_left = ts_at(left_idx)
        t_right = ts_at(right_idx)
        denom = max(1e-3, (t_right - t_left))
        phase = float(np.clip((t_curr - t_left) / denom, 0.0, 1.0))
        closest_force = self.force_data[left_idx]
        if not closest_force or not closest_force.get('force_curve'):
            return None
        num_pts = len(closest_force['force_curve'])
        current_idx = int(np.clip(round(phase * (num_pts - 1)), 0, num_pts - 1))

        # Delegate to the overlay-style plot for identical look and metadata, with animated dot
        force_curve = closest_force['force_curve']
        spm = closest_force.get('spm')
        distance = closest_force.get('distance')
        plot_elapsed = closest_force.get('clip_elapsed_s', closest_force.get('elapsed_s', 0.0))
        return self._create_force_curve_plot(force_curve, closest_force['power'], spm, distance, plot_elapsed, current_idx, frame_abs_dt)

    def _create_force_curve_plot(self, force_curve, power, spm, distance, elapsed_s, current_idx=None, frame_abs_dt=None):
        """Create a matplotlib plot of the force curve (parity with overlay_force_angles_video)."""
        if not force_curve:
            return None

        base_w, base_h = 4, 3
        scale = float(getattr(self, 'plot_scale', 1.0))
        fig, ax = plt.subplots(figsize=(base_w * scale, base_h * scale), dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        x = np.arange(len(force_curve))
        ax.plot(x, force_curve, 'lime', linewidth=3, label='Force')

        # Animated dot at current index; fall back to peak if not provided
        if current_idx is None:
            current_idx = int(np.argmax(force_curve))
        current_idx = int(np.clip(current_idx, 0, len(force_curve) - 1))
        current_force = float(force_curve[current_idx])
        ax.plot(current_idx, current_force, 'ro', markersize=8)

        ax.set_xlabel('Stroke Position', color='white', fontsize=10)
        ax.set_ylabel('Force', color='white', fontsize=10)
        # Include SPM in title to match original overlay
        spm_text = f", {int(spm)}spm" if spm is not None else ""
        ax.set_title(f'Force Curve - {int(power)}W{spm_text}', color='white', fontsize=int(12 * scale))
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white', labelsize=int(8 * scale))

        peak_force_val = int(np.max(force_curve))
        stats = f'Peak: {peak_force_val}\nAvg: {np.mean(force_curve):.1f}'
        if distance is not None:
            stats += f'\nDist: {float(distance):.1f}m'
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                verticalalignment='top', color='white', fontsize=int(9 * scale),
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # Timestamp label for troubleshooting
        if frame_abs_dt is not None:
            ts_text = frame_abs_dt.strftime('%H:%M:%S.%f')[:-3]
            ax.text(0.70, 0.02, ts_text, transform=ax.transAxes,
                    verticalalignment='bottom', color='white', fontsize=int(10 * scale),
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

        plt.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]
        plt.close(fig)
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    def _create_analysis_summary_display(self, pose_frame, elapsed_s):
        """Create analysis summary display"""
        # Create a semi-transparent background
        img = np.zeros((150, 500, 3), dtype=np.uint8)
        
        # Define colors
        colors = {
            'white': (255, 255, 255),
            'lime': (0, 255, 0),
            'cyan': (255, 255, 0),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'orange': (0, 165, 255)
        }
        
        # Title
        cv2.putText(img, "ANALYSIS SUMMARY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['white'], 2)
        
        y_pos = 50
        line_height = 20
        
        # Add analysis results if available
        if 'symmetry_analysis' in self.analysis_results:
            symmetry = self.analysis_results['symmetry_analysis']
            cv2.putText(img, f"Symmetry Score: {symmetry['overall_symmetry_score']:.1f}%", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['lime'], 1)
            y_pos += line_height
        
        if 'technique_quality' in self.analysis_results:
            quality = self.analysis_results['technique_quality']
            cv2.putText(img, f"Technique Score: {quality['overall_technique_score']:.1f}%", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
            y_pos += line_height
        
        if 'timing_analysis' in self.analysis_results:
            timing = self.analysis_results['timing_analysis']
            cv2.putText(img, f"Stroke Rate: {timing['stroke_rate']:.1f} spm", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
            y_pos += line_height
        
        if self.force_data:
            # Find current power
            current_power = 0
            for force_entry in self.force_data:
                t = force_entry.get('clip_elapsed_s', force_entry.get('elapsed_s', 0.0))
                if abs(elapsed_s - t) < 5.0:
                    current_power = force_entry['power']
                    break
            
            cv2.putText(img, f"Current Power: {current_power}W", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)
        
        return img
    
    def _overlay_display(self, frame, display, x_offset, y_offset):
        """Overlay a display on the frame"""
        if display is None:
            return
        
        display_height, display_width = display.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        
        # Ensure the display fits within the frame
        if (x_offset + display_width <= frame_width and 
            y_offset + display_height <= frame_height):
            frame[y_offset:y_offset+display_height, x_offset:x_offset+display_width] = display

    def _compute_clip_window_and_filter_force(self, video_path):
        """If the video filename contains HHMMSS-HHMMSS, filter force data to that absolute window
        and store clip-relative elapsed seconds as 'clip_elapsed_s'."""
        if not self.force_data:
            return
        # Determine base date from first available force timestamp
        first_dt = None
        for e in self.force_data:
            if e.get('timestamp_dt') is not None:
                first_dt = e['timestamp_dt']
                break
        if first_dt is None:
            return
        m = re.search(r"(\d{6})-(\d{6})", os.path.basename(video_path))
        if not m:
            return
        start_hhmmss, end_hhmmss = m.group(1), m.group(2)
        def to_dt(hhmmss):
            hh = int(hhmmss[0:2]); mm = int(hhmmss[2:4]); ss = int(hhmmss[4:6])
            return datetime(first_dt.year, first_dt.month, first_dt.day, hh, mm, ss)
        start_dt = to_dt(start_hhmmss)
        end_dt = to_dt(end_hhmmss)
        if end_dt < start_dt:
            end_dt = end_dt.replace(day=end_dt.day + 1)
        filtered = []
        for e in self.force_data:
            ts = e.get('timestamp_dt')
            if ts and start_dt <= ts <= end_dt:
                e['clip_elapsed_s'] = (ts - start_dt).total_seconds()
                filtered.append(e)
        if filtered:
            self.force_data = filtered

    def _draw_joint_angles_on_frame(self, frame, pose_frame):
        """Draw angles at the elbow, knee, hip (and torso lean) directly on the video frame.
        This makes it clear what each angle represents by rendering the numeric value at the
        joint vertex and drawing simple limb segments.
        """
        # Helper to fetch a keypoint triplet
        def get_kpt(name):
            x_key, y_key, c_key = f"{name}_x", f"{name}_y", f"{name}_confidence"
            if x_key in pose_frame and y_key in pose_frame and c_key in pose_frame:
                conf = pose_frame.get(c_key, 0)
                if conf is None:
                    conf = 0
                if conf > 0.5:
                    return (int(pose_frame[x_key]), int(pose_frame[y_key]))
            return None

        def calc_angle(p1, p2, p3):
            if p1 is None or p2 is None or p3 is None:
                return None
            a = np.array(p1, dtype=np.float32)
            b = np.array(p2, dtype=np.float32)
            c = np.array(p3, dtype=np.float32)
            v1 = a - b
            v2 = c - b
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom == 0:
                return None
            cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            return float(np.degrees(np.arccos(cosang)))

        def draw_segment(p, q, color, thickness=3):
            if p is not None and q is not None:
                cv2.line(frame, p, q, color, thickness)

        def draw_kpt(p, color):
            if p is not None:
                cv2.circle(frame, p, 6, color, -1)

        def draw_badge(p, text, bg_color, text_color=(0, 0, 0), draw_deg=False):
            if p is None:
                return
            x, y = p
            pad = 6
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x1, y1 = x - tw // 2 - pad, y - th - baseline - pad
            x2, y2 = x + tw // 2 + pad, y + pad
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1]-1, x2)
            y2 = min(frame.shape[0]-1, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50,50,50), 1)
            cv2.putText(frame, text, (x1 + pad, y2 - pad - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            # Manually render a degree symbol as a small circle to avoid unicode issues
            if draw_deg:
                cx = x1 + pad + tw + 6
                cy = y1 + th // 3  # slightly above midline
                cx = min(cx, frame.shape[1]-3)
                cy = max(3, min(cy, frame.shape[0]-3))
                cv2.circle(frame, (cx, cy), 4, text_color, 2)

        # Collect keypoints we care about
        ls = get_kpt('left_shoulder')
        rs = get_kpt('right_shoulder')
        le = get_kpt('left_elbow')
        re = get_kpt('right_elbow')
        lw = get_kpt('left_wrist')
        rw = get_kpt('right_wrist')
        lh = get_kpt('left_hip')
        rh = get_kpt('right_hip')
        lk = get_kpt('left_knee')
        rk = get_kpt('right_knee')
        la = get_kpt('left_ankle')
        ra = get_kpt('right_ankle')

        # Colors (BGR)
        color_arm = (0, 255, 255)   # yellow
        color_leg = (255, 153, 0)   # blue-ish? actually BGR; but we'll keep distinct
        color_left = (0, 255, 0)    # green for left joint dot
        color_right = (255, 0, 0)   # red for right joint dot

        # Draw simple skeleton segments for context
        draw_segment(ls, le, color_arm)
        draw_segment(le, lw, color_arm)
        draw_segment(rs, re, color_arm)
        draw_segment(re, rw, color_arm)
        draw_segment(lh, lk, color_leg)
        draw_segment(lk, la, color_leg)
        draw_segment(rh, rk, color_leg)
        draw_segment(rk, ra, color_leg)
        draw_segment(ls, lh, (200,200,200))
        draw_segment(rs, rh, (200,200,200))
        draw_segment(lh, rh, (200,200,200))

        # Draw keypoints
        for p, col in [(le, color_left), (re, color_right), (lk, color_left), (rk, color_right), (lh, (255,255,255))]:
            draw_kpt(p, col)

        # Compute angles and draw badges at the vertex
        ang_left_elbow = calc_angle(ls, le, lw)
        if ang_left_elbow is not None:
            draw_badge(le, f"{ang_left_elbow:.0f}", (0, 255, 255), draw_deg=True)

        ang_right_elbow = calc_angle(rs, re, rw)
        if ang_right_elbow is not None:
            draw_badge(re, f"{ang_right_elbow:.0f}", (0, 255, 255), draw_deg=True)

        ang_left_knee = calc_angle(lh, lk, la)
        if ang_left_knee is not None:
            draw_badge(lk, f"{ang_left_knee:.0f}", (255, 200, 0), draw_deg=True)

        ang_right_knee = calc_angle(rh, rk, ra)
        if ang_right_knee is not None:
            draw_badge(rk, f"{ang_right_knee:.0f}", (255, 200, 0), draw_deg=True)

        # Hip angle (shoulder-hip-knee) — draw at hip
        ang_left_hip = calc_angle(ls, lh, lk)
        if ang_left_hip is not None:
            draw_badge(lh, f"{ang_left_hip:.0f}", (200, 200, 200), draw_deg=True)

        ang_right_hip = calc_angle(rs, rh, rk)
        if ang_right_hip is not None:
            draw_badge(rh, f"{ang_right_hip:.0f}", (200, 200, 200), draw_deg=True)

        # Ankle dorsiflexion/plantarflexion approximation:
        # angle between shank (knee->ankle) and a vertical reference line through the ankle
        def draw_vertical_dotted(p_start, length=110, color=(180, 180, 180)):
            if p_start is None:
                return None
            x, y = p_start
            y2 = max(0, y - length)
            # dotted vertical line
            for t in range(y2, y, 8):
                cv2.line(frame, (x, t), (x, min(y, t+4)), color, 2)
            return (x, y2)

        # Left ankle angle (signed relative to vertical): positive when knee is forward (to the right)
        if la is not None and lk is not None:
            top = draw_vertical_dotted(la)
            if top is not None:
                # Signed angle using reference vector up (vertical) and shank vector
                v_ref = np.array([0.0, -1.0], dtype=np.float32)  # up
                v_shank = np.array([lk[0] - la[0], lk[1] - la[1]], dtype=np.float32)
                n = np.linalg.norm(v_shank)
                if n > 0:
                    v_shank /= n
                    cross_z = v_ref[0]*v_shank[1] - v_ref[1]*v_shank[0]
                    dot = v_ref[0]*v_shank[0] + v_ref[1]*v_shank[1]
                    signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                    draw_badge(la, f"{signed_deg:.0f}", (255, 255, 0), draw_deg=True)

        # Right ankle angle (signed relative to vertical): positive when knee is forward (to the right)
        if ra is not None and rk is not None:
            top = draw_vertical_dotted(ra)
            if top is not None:
                v_ref = np.array([0.0, -1.0], dtype=np.float32)
                v_shank = np.array([rk[0] - ra[0], rk[1] - ra[1]], dtype=np.float32)
                n = np.linalg.norm(v_shank)
                if n > 0:
                    v_shank /= n
                    cross_z = v_ref[0]*v_shank[1] - v_ref[1]*v_shank[0]
                    dot = v_ref[0]*v_shank[0] + v_ref[1]*v_shank[1]
                    signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                    draw_badge(ra, f"{signed_deg:.0f}", (255, 255, 0), draw_deg=True)

        # Thigh angle relative to vertical (hip->knee vs vertical), analogous to ankle logic
        # Left thigh
        if lh is not None and lk is not None:
            top = draw_vertical_dotted(lh)
            if top is not None:
                v_ref = np.array([0.0, -1.0], dtype=np.float32)  # up
                v_thigh = np.array([lk[0] - lh[0], lk[1] - lh[1]], dtype=np.float32)
                n = np.linalg.norm(v_thigh)
                if n > 0:
                    v_thigh /= n
                    cross_z = v_ref[0]*v_thigh[1] - v_ref[1]*v_thigh[0]
                    dot = v_ref[0]*v_thigh[0] + v_ref[1]*v_thigh[1]
                    signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                    # Use orange-ish badge for thigh
                    draw_badge(lh, f"{signed_deg:.0f}", (0, 165, 255), draw_deg=True)

        # Right thigh
        if rh is not None and rk is not None:
            top = draw_vertical_dotted(rh)
            if top is not None:
                v_ref = np.array([0.0, -1.0], dtype=np.float32)
                v_thigh = np.array([rk[0] - rh[0], rk[1] - rh[1]], dtype=np.float32)
                n = np.linalg.norm(v_thigh)
                if n > 0:
                    v_thigh /= n
                    cross_z = v_ref[0]*v_thigh[1] - v_ref[1]*v_thigh[0]
                    dot = v_ref[0]*v_thigh[0] + v_ref[1]*v_thigh[1]
                    signed_deg = float(np.degrees(np.arctan2(cross_z, dot)))
                    draw_badge(rh, f"{signed_deg:.0f}", (0, 165, 255), draw_deg=True)

        # Torso lean badge near midpoint between shoulders
        if ls is not None and rs is not None and lh is not None and rh is not None:
            shoulder_center = (int((ls[0] + rs[0]) / 2), int((ls[1] + rs[1]) / 2))
            hip_center = (int((lh[0] + rh[0]) / 2), int((lh[1] + rh[1]) / 2))
            # vertical ref from shoulder center straight down
            vertical_ref = (shoulder_center[0], shoulder_center[1] + 100)
            torso_angle = calc_angle(vertical_ref, shoulder_center, hip_center)
            if torso_angle is not None:
                draw_badge(shoulder_center, f"{torso_angle:.0f}", (0, 255, 128), draw_deg=True)

def main():
    """Main analysis function with CLI modes"""
    parser = argparse.ArgumentParser(description="Advanced Rowing Analysis")
    parser.add_argument("--mode", choices=["full", "overlay_only"], default="full",
                        help="Run full pipeline or only generate overlay video")
    parser.add_argument("--time-offset-ms", type=float, default=0.0,
                        help="Apply a constant time offset (ms) to video frame timestamps for sync")
    parser.add_argument("--plot-scale", type=float, default=0.75,
                        help="Scale factor for the force plot size (e.g., 0.75)")
    args = parser.parse_args()

    print("🚣‍♂️ Advanced Rowing Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AdvancedRowingAnalyzer()
    analyzer.time_offset_s = float(args.time_offset_ms) / 1000.0
    analyzer.plot_scale = float(args.plot_scale)
    
    # Load data
    if not analyzer.load_data():
        print("❌ Cannot proceed without data")
        return
    
    if args.mode == "overlay_only":
        analyzer._create_analysis_video()
    else:
        # Perform analysis
        analyzer.analyze_rowing_technique()
        
        # Create advanced report
        analyzer.create_advanced_report()
        
        print("\n🎉 Advanced rowing analysis complete!")
        print("📁 Check the 'advanced_analysis/' directory for results")

if __name__ == "__main__":
    main()
