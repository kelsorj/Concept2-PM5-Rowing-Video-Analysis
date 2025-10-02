#!/usr/bin/env python3
"""
Enhanced Rowing Kinematics Analysis
Provides detailed biomechanical analysis similar to professional rowing analysis software
"""

import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class RowingKinematicsAnalyzer:
    def __init__(self):
        self.pose_data = None
        self.force_data = None
        self.analysis_results = {}
        
    def load_pose_data(self):
        """Load the most recent pose data"""
        pose_files = glob.glob("rowing_pose_data_*.json")
        if not pose_files:
            print("❌ No pose data files found")
            return False
        
        latest_pose_file = max(pose_files, key=os.path.getctime)
        print(f"📊 Loading pose data: {latest_pose_file}")
        
        with open(latest_pose_file, 'r') as f:
            self.pose_data = json.load(f)
        
        print(f"   Found {len(self.pose_data)} pose frames")
        return True
    
    def load_force_data(self):
        """Load the most recent force data"""
        force_files = glob.glob("pm5_py3row_parsed_*.csv")
        if not force_files:
            print("❌ No force data files found")
            return False
        
        latest_force_file = max(force_files, key=os.path.getctime)
        print(f"📊 Loading force data: {latest_force_file}")
        
        df = pd.read_csv(latest_force_file)
        
        # Parse forceplot data
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
                            'distance': row['distance_m']
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
        
        self.force_data = force_data
        print(f"   Found {len(self.force_data)} complete force curves")
        return True
    
    def calculate_kinematic_metrics(self):
        """Calculate comprehensive kinematic metrics"""
        print("\n🔍 Calculating Kinematic Metrics...")
        
        if not self.pose_data:
            print("❌ No pose data available")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.pose_data)
        
        # Filter out frames with missing key data
        df = df.dropna(subset=['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle'])
        
        if len(df) == 0:
            print("❌ No valid pose data after filtering")
            return
        
        # Calculate stroke phases
        self._identify_stroke_phases(df)
        
        # Calculate joint angles and ranges
        self._calculate_joint_metrics(df)
        
        # Calculate symmetry metrics
        self._calculate_symmetry_metrics(df)
        
        # Calculate timing metrics
        self._calculate_timing_metrics(df)
        
        # Calculate power metrics (if force data available)
        if self.force_data:
            self._calculate_power_metrics(df)
        
        print("✅ Kinematic metrics calculated")
    
    def _identify_stroke_phases(self, df):
        """Identify drive and recovery phases"""
        print("   📊 Identifying stroke phases...")
        
        # Use leg angles to identify phases
        # Drive: legs extending (angle increasing)
        # Recovery: legs compressing (angle decreasing)
        
        leg_angles = (df['left_leg_angle'] + df['right_leg_angle']) / 2
        
        # Find peaks and valleys in leg angle
        peaks, _ = signal.find_peaks(leg_angles, distance=10, prominence=5)
        valleys, _ = signal.find_peaks(-leg_angles, distance=10, prominence=5)
        
        # Create phase labels
        phases = ['Recovery'] * len(df)
        
        # Mark drive phases (between valley and peak)
        for i, peak in enumerate(peaks):
            # Find the valley before this peak
            prev_valleys = valleys[valleys < peak]
            if len(prev_valleys) > 0:
                start_valley = prev_valleys[-1]
                phases[start_valley:peak] = ['Drive'] * (peak - start_valley)
        
        df['stroke_phase'] = phases
        df['leg_angle_avg'] = leg_angles
        
        self.analysis_results['stroke_phases'] = {
            'total_drive_frames': sum(1 for p in phases if p == 'Drive'),
            'total_recovery_frames': sum(1 for p in phases if p == 'Recovery'),
            'drive_ratio': sum(1 for p in phases if p == 'Drive') / len(phases)
        }
    
    def _calculate_joint_metrics(self, df):
        """Calculate joint angle metrics"""
        print("   📊 Calculating joint metrics...")
        
        joint_metrics = {}
        
        # Arm angles
        arm_metrics = {
            'left_arm_range': df['left_arm_angle'].max() - df['left_arm_angle'].min(),
            'right_arm_range': df['right_arm_angle'].max() - df['right_arm_angle'].min(),
            'left_arm_mean': df['left_arm_angle'].mean(),
            'right_arm_mean': df['right_arm_angle'].mean(),
            'left_arm_std': df['left_arm_angle'].std(),
            'right_arm_std': df['right_arm_angle'].std()
        }
        
        # Leg angles
        leg_metrics = {
            'left_leg_range': df['left_leg_angle'].max() - df['left_leg_angle'].min(),
            'right_leg_range': df['right_leg_angle'].max() - df['right_leg_angle'].min(),
            'left_leg_mean': df['left_leg_angle'].mean(),
            'right_leg_mean': df['right_leg_angle'].mean(),
            'left_leg_std': df['left_leg_angle'].std(),
            'right_leg_std': df['right_leg_angle'].std()
        }
        
        # Torso angle
        if 'torso_lean_angle' in df.columns:
            torso_metrics = {
                'torso_range': df['torso_lean_angle'].max() - df['torso_lean_angle'].min(),
                'torso_mean': df['torso_lean_angle'].mean(),
                'torso_std': df['torso_lean_angle'].std()
            }
        else:
            torso_metrics = {}
        
        joint_metrics.update(arm_metrics)
        joint_metrics.update(leg_metrics)
        joint_metrics.update(torso_metrics)
        
        self.analysis_results['joint_metrics'] = joint_metrics
    
    def _calculate_symmetry_metrics(self, df):
        """Calculate left-right symmetry metrics"""
        print("   📊 Calculating symmetry metrics...")
        
        symmetry_metrics = {}
        
        # Arm symmetry
        arm_diff = df['left_arm_angle'] - df['right_arm_angle']
        symmetry_metrics['arm_symmetry_mean'] = arm_diff.mean()
        symmetry_metrics['arm_symmetry_std'] = arm_diff.std()
        symmetry_metrics['arm_symmetry_max_diff'] = arm_diff.abs().max()
        
        # Leg symmetry
        leg_diff = df['left_leg_angle'] - df['right_leg_angle']
        symmetry_metrics['leg_symmetry_mean'] = leg_diff.mean()
        symmetry_metrics['leg_symmetry_std'] = leg_diff.std()
        symmetry_metrics['leg_symmetry_max_diff'] = leg_diff.abs().max()
        
        # Correlation between left and right
        symmetry_metrics['arm_correlation'] = pearsonr(df['left_arm_angle'], df['right_arm_angle'])[0]
        symmetry_metrics['leg_correlation'] = pearsonr(df['left_leg_angle'], df['right_leg_angle'])[0]
        
        self.analysis_results['symmetry_metrics'] = symmetry_metrics
    
    def _calculate_timing_metrics(self, df):
        """Calculate timing and rhythm metrics"""
        print("   📊 Calculating timing metrics...")
        
        timing_metrics = {}
        
        # Calculate stroke rate from phase changes
        phase_changes = (df['stroke_phase'] != df['stroke_phase'].shift()).sum()
        total_time = df['timestamp'].max() - df['timestamp'].min()
        
        if total_time > 0:
            timing_metrics['stroke_rate'] = (phase_changes / 2) / (total_time / 60)  # strokes per minute
            timing_metrics['stroke_duration'] = total_time / (phase_changes / 2)  # seconds per stroke
        
        # Calculate drive/recovery timing
        drive_frames = df[df['stroke_phase'] == 'Drive']
        recovery_frames = df[df['stroke_phase'] == 'Recovery']
        
        if len(drive_frames) > 0 and len(recovery_frames) > 0:
            timing_metrics['drive_time'] = drive_frames['timestamp'].max() - drive_frames['timestamp'].min()
            timing_metrics['recovery_time'] = recovery_frames['timestamp'].max() - recovery_frames['timestamp'].min()
            timing_metrics['drive_recovery_ratio'] = timing_metrics['drive_time'] / timing_metrics['recovery_time']
        
        self.analysis_results['timing_metrics'] = timing_metrics
    
    def _calculate_power_metrics(self, df):
        """Calculate power-related metrics"""
        print("   📊 Calculating power metrics...")
        
        if not self.force_data:
            return
        
        power_metrics = {}
        
        # Calculate average power from force data
        powers = [f['power'] for f in self.force_data]
        power_metrics['avg_power'] = np.mean(powers)
        power_metrics['max_power'] = np.max(powers)
        power_metrics['power_std'] = np.std(powers)
        
        # Calculate force curve metrics
        peak_forces = []
        avg_forces = []
        
        for force_entry in self.force_data:
            if force_entry['force_curve']:
                peak_forces.append(max(force_entry['force_curve']))
                avg_forces.append(np.mean(force_entry['force_curve']))
        
        if peak_forces:
            power_metrics['avg_peak_force'] = np.mean(peak_forces)
            power_metrics['max_peak_force'] = np.max(peak_forces)
            power_metrics['avg_force'] = np.mean(avg_forces)
        
        self.analysis_results['power_metrics'] = power_metrics
    
    def create_comprehensive_report(self):
        """Create a comprehensive analysis report"""
        print("\n📊 Creating Comprehensive Analysis Report...")
        
        # Create output directory
        os.makedirs('kinematics_analysis', exist_ok=True)
        
        # Create summary report
        self._create_summary_report()
        
        # Create detailed plots
        self._create_detailed_plots()
        
        # Create CSV export
        self._create_csv_export()
        
        print("✅ Comprehensive analysis report created")
    
    def _create_summary_report(self):
        """Create a text summary report"""
        report_path = 'kinematics_analysis/rowing_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("🚣‍♂️ ROWING KINEMATICS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Stroke Phases
            if 'stroke_phases' in self.analysis_results:
                f.write("STROKE PHASES:\n")
                f.write("-" * 20 + "\n")
                phases = self.analysis_results['stroke_phases']
                f.write(f"Drive Ratio: {phases['drive_ratio']:.2%}\n")
                f.write(f"Drive Frames: {phases['total_drive_frames']}\n")
                f.write(f"Recovery Frames: {phases['total_recovery_frames']}\n\n")
            
            # Joint Metrics
            if 'joint_metrics' in self.analysis_results:
                f.write("JOINT METRICS:\n")
                f.write("-" * 20 + "\n")
                joints = self.analysis_results['joint_metrics']
                f.write(f"Left Arm Range: {joints['left_arm_range']:.1f}°\n")
                f.write(f"Right Arm Range: {joints['right_arm_range']:.1f}°\n")
                f.write(f"Left Leg Range: {joints['left_leg_range']:.1f}°\n")
                f.write(f"Right Leg Range: {joints['right_leg_range']:.1f}°\n")
                if 'torso_range' in joints:
                    f.write(f"Torso Range: {joints['torso_range']:.1f}°\n")
                f.write("\n")
            
            # Symmetry Metrics
            if 'symmetry_metrics' in self.analysis_results:
                f.write("SYMMETRY METRICS:\n")
                f.write("-" * 20 + "\n")
                sym = self.analysis_results['symmetry_metrics']
                f.write(f"Arm Symmetry (Mean Diff): {sym['arm_symmetry_mean']:.1f}°\n")
                f.write(f"Leg Symmetry (Mean Diff): {sym['leg_symmetry_mean']:.1f}°\n")
                f.write(f"Arm Correlation: {sym['arm_correlation']:.3f}\n")
                f.write(f"Leg Correlation: {sym['leg_correlation']:.3f}\n\n")
            
            # Timing Metrics
            if 'timing_metrics' in self.analysis_results:
                f.write("TIMING METRICS:\n")
                f.write("-" * 20 + "\n")
                timing = self.analysis_results['timing_metrics']
                if 'stroke_rate' in timing:
                    f.write(f"Stroke Rate: {timing['stroke_rate']:.1f} spm\n")
                if 'drive_recovery_ratio' in timing:
                    f.write(f"Drive/Recovery Ratio: {timing['drive_recovery_ratio']:.2f}\n")
                f.write("\n")
            
            # Power Metrics
            if 'power_metrics' in self.analysis_results:
                f.write("POWER METRICS:\n")
                f.write("-" * 20 + "\n")
                power = self.analysis_results['power_metrics']
                f.write(f"Average Power: {power['avg_power']:.1f}W\n")
                f.write(f"Max Power: {power['max_power']:.1f}W\n")
                if 'avg_peak_force' in power:
                    f.write(f"Average Peak Force: {power['avg_peak_force']:.1f}\n")
                f.write("\n")
        
        print(f"   📋 Summary report: {report_path}")
    
    def _create_detailed_plots(self):
        """Create detailed analysis plots"""
        if not self.pose_data:
            return
        
        df = pd.DataFrame(self.pose_data)
        df = df.dropna(subset=['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle'])
        
        # Set up the plotting style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        # Create comprehensive analysis figure
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('🚣‍♂️ Comprehensive Rowing Kinematics Analysis', fontsize=20, color='white')
        
        # 1. Joint Angles Over Time
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(df['timestamp'], df['left_arm_angle'], 'lime', label='Left Arm', linewidth=2)
        ax1.plot(df['timestamp'], df['right_arm_angle'], 'cyan', label='Right Arm', linewidth=2)
        ax1.set_title('Arm Angles Over Time', color='white', fontsize=14)
        ax1.set_xlabel('Time (s)', color='white')
        ax1.set_ylabel('Angle (degrees)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Leg Angles Over Time
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(df['timestamp'], df['left_leg_angle'], 'lime', label='Left Leg', linewidth=2)
        ax2.plot(df['timestamp'], df['right_leg_angle'], 'cyan', label='Right Leg', linewidth=2)
        ax2.set_title('Leg Angles Over Time', color='white', fontsize=14)
        ax2.set_xlabel('Time (s)', color='white')
        ax2.set_ylabel('Angle (degrees)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Symmetry Analysis
        ax3 = plt.subplot(3, 3, 3)
        arm_diff = df['left_arm_angle'] - df['right_arm_angle']
        leg_diff = df['left_leg_angle'] - df['right_leg_angle']
        ax3.plot(df['timestamp'], arm_diff, 'orange', label='Arm Difference', linewidth=2)
        ax3.plot(df['timestamp'], leg_diff, 'red', label='Leg Difference', linewidth=2)
        ax3.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax3.set_title('Left-Right Symmetry', color='white', fontsize=14)
        ax3.set_xlabel('Time (s)', color='white')
        ax3.set_ylabel('Angle Difference (degrees)', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Joint Range of Motion
        ax4 = plt.subplot(3, 3, 4)
        joints = ['Left Arm', 'Right Arm', 'Left Leg', 'Right Leg']
        ranges = [
            df['left_arm_angle'].max() - df['left_arm_angle'].min(),
            df['right_arm_angle'].max() - df['right_arm_angle'].min(),
            df['left_leg_angle'].max() - df['left_leg_angle'].min(),
            df['right_leg_angle'].max() - df['right_leg_angle'].min()
        ]
        bars = ax4.bar(joints, ranges, color=['lime', 'cyan', 'lime', 'cyan'], alpha=0.7)
        ax4.set_title('Range of Motion', color='white', fontsize=14)
        ax4.set_ylabel('Angle Range (degrees)', color='white')
        ax4.tick_params(axis='x', rotation=45, colors='white')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, ranges):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}°', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 5. Correlation Matrix
        ax5 = plt.subplot(3, 3, 5)
        corr_data = df[['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle']].corr()
        im = ax5.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(corr_data.columns)))
        ax5.set_yticks(range(len(corr_data.columns)))
        ax5.set_xticklabels(['L Arm', 'R Arm', 'L Leg', 'R Leg'], color='white')
        ax5.set_yticklabels(['L Arm', 'R Arm', 'L Leg', 'R Leg'], color='white')
        ax5.set_title('Joint Correlation Matrix', color='white', fontsize=14)
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                ax5.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        # 6. Phase Analysis (if available)
        ax6 = plt.subplot(3, 3, 6)
        if 'stroke_phase' in df.columns:
            drive_data = df[df['stroke_phase'] == 'Drive']
            recovery_data = df[df['stroke_phase'] == 'Recovery']
            
            ax6.scatter(drive_data['left_leg_angle'], drive_data['left_arm_angle'], 
                       c='red', label='Drive', alpha=0.6, s=20)
            ax6.scatter(recovery_data['left_leg_angle'], recovery_data['left_arm_angle'], 
                       c='blue', label='Recovery', alpha=0.6, s=20)
            ax6.set_title('Leg vs Arm Angles by Phase', color='white', fontsize=14)
            ax6.set_xlabel('Leg Angle (degrees)', color='white')
            ax6.set_ylabel('Arm Angle (degrees)', color='white')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Phase Analysis\nNot Available', 
                    ha='center', va='center', transform=ax6.transAxes, 
                    color='white', fontsize=12)
            ax6.set_title('Phase Analysis', color='white', fontsize=14)
        
        # 7. Force Analysis (if available)
        ax7 = plt.subplot(3, 3, 7)
        if self.force_data:
            powers = [f['power'] for f in self.force_data]
            times = [f['elapsed_s'] for f in self.force_data]
            ax7.plot(times, powers, 'yellow', linewidth=2, marker='o', markersize=4)
            ax7.set_title('Power Over Time', color='white', fontsize=14)
            ax7.set_xlabel('Time (s)', color='white')
            ax7.set_ylabel('Power (W)', color='white')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Force Data\nNot Available', 
                    ha='center', va='center', transform=ax7.transAxes, 
                    color='white', fontsize=12)
            ax7.set_title('Force Analysis', color='white', fontsize=14)
        
        # 8. Statistical Summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        # Create summary text
        summary_text = "ANALYSIS SUMMARY\n\n"
        
        if 'joint_metrics' in self.analysis_results:
            joints = self.analysis_results['joint_metrics']
            summary_text += f"Arm Range: {joints['left_arm_range']:.1f}°\n"
            summary_text += f"Leg Range: {joints['left_leg_range']:.1f}°\n"
        
        if 'symmetry_metrics' in self.analysis_results:
            sym = self.analysis_results['symmetry_metrics']
            summary_text += f"Arm Symmetry: {sym['arm_symmetry_mean']:.1f}°\n"
            summary_text += f"Leg Symmetry: {sym['leg_symmetry_mean']:.1f}°\n"
        
        if 'timing_metrics' in self.analysis_results and 'stroke_rate' in self.analysis_results['timing_metrics']:
            timing = self.analysis_results['timing_metrics']
            summary_text += f"Stroke Rate: {timing['stroke_rate']:.1f} spm\n"
        
        if 'power_metrics' in self.analysis_results:
            power = self.analysis_results['power_metrics']
            summary_text += f"Avg Power: {power['avg_power']:.1f}W\n"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
                color='white', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # 9. Technique Recommendations
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        recommendations = "TECHNIQUE RECOMMENDATIONS\n\n"
        
        if 'symmetry_metrics' in self.analysis_results:
            sym = self.analysis_results['symmetry_metrics']
            if abs(sym['arm_symmetry_mean']) > 5:
                recommendations += "• Focus on arm symmetry\n"
            if abs(sym['leg_symmetry_mean']) > 5:
                recommendations += "• Focus on leg symmetry\n"
        
        if 'timing_metrics' in self.analysis_results and 'drive_recovery_ratio' in self.analysis_results['timing_metrics']:
            timing = self.analysis_results['timing_metrics']
            if timing['drive_recovery_ratio'] < 0.5:
                recommendations += "• Increase drive phase duration\n"
            elif timing['drive_recovery_ratio'] > 1.0:
                recommendations += "• Increase recovery phase duration\n"
        
        if not recommendations.endswith("\n\n"):
            recommendations += "• Technique looks good!\n"
        
        ax9.text(0.1, 0.9, recommendations, transform=ax9.transAxes, 
                color='white', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('kinematics_analysis/comprehensive_kinematics_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print("   📊 Comprehensive analysis plot: kinematics_analysis/comprehensive_kinematics_analysis.png")
    
    def _create_csv_export(self):
        """Create CSV export of all metrics"""
        if not self.analysis_results:
            return
        
        # Flatten the results dictionary
        flat_results = {}
        for category, metrics in self.analysis_results.items():
            for metric, value in metrics.items():
                flat_results[f"{category}_{metric}"] = value
        
        # Create DataFrame and save
        df_results = pd.DataFrame([flat_results])
        df_results.to_csv('kinematics_analysis/kinematics_metrics.csv', index=False)
        
        print("   📊 Metrics CSV: kinematics_analysis/kinematics_metrics.csv")

def main():
    """Main analysis function"""
    print("🚣‍♂️ Enhanced Rowing Kinematics Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = RowingKinematicsAnalyzer()
    
    # Load data
    pose_loaded = analyzer.load_pose_data()
    force_loaded = analyzer.load_force_data()
    
    if not pose_loaded:
        print("❌ Cannot proceed without pose data")
        return
    
    # Perform analysis
    analyzer.calculate_kinematic_metrics()
    
    # Create comprehensive report
    analyzer.create_comprehensive_report()
    
    print("\n🎉 Enhanced kinematics analysis complete!")
    print("📁 Check the 'kinematics_analysis/' directory for results")

if __name__ == "__main__":
    main()
