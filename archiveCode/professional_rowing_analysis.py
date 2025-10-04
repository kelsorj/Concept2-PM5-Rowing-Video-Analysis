#!/usr/bin/env python3
"""
Professional Rowing Analysis System
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
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class ProfessionalRowingAnalyzer:
    def __init__(self):
        self.pose_data = None
        self.force_data = None
        self.stroke_cycles = []
        self.analysis_results = {}
        
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
                                'distance': row['distance_m']
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            self.force_data = force_data
            print(f"   Found {len(self.force_data)} complete force curves")
        
        print(f"   Found {len(self.pose_data)} pose frames")
        return True
    
    def identify_stroke_cycles(self):
        """Identify individual stroke cycles from pose data"""
        print("\n🔍 Identifying Stroke Cycles...")
        
        df = pd.DataFrame(self.pose_data)
        df = df.dropna(subset=['left_leg_angle', 'right_leg_angle'])
        
        # Use leg angles to identify stroke cycles
        leg_angles = (df['left_leg_angle'] + df['right_leg_angle']) / 2
        
        # Find peaks (legs extended - end of drive) and valleys (legs compressed - catch)
        peaks, _ = signal.find_peaks(leg_angles, distance=15, prominence=10)
        valleys, _ = signal.find_peaks(-leg_angles, distance=15, prominence=10)
        
        # Create stroke cycles
        self.stroke_cycles = []
        
        for i in range(min(len(peaks), len(valleys))):
            if valleys[i] < peaks[i]:  # Valid cycle: valley to peak
                cycle_data = df.iloc[valleys[i]:peaks[i]+1].copy()
                cycle_data['cycle_phase'] = self._calculate_cycle_phases(cycle_data)
                cycle_data['cycle_id'] = i
                self.stroke_cycles.append(cycle_data)
        
        print(f"   Identified {len(self.stroke_cycles)} complete stroke cycles")
    
    def _calculate_cycle_phases(self, cycle_data):
        """Calculate detailed stroke phases within a cycle"""
        leg_angles = (cycle_data['left_leg_angle'] + cycle_data['right_leg_angle']) / 2
        
        # Find the point of maximum leg extension (end of drive)
        max_leg_idx = leg_angles.idxmax()
        max_leg_pos = cycle_data.index.get_loc(max_leg_idx)
        
        phases = ['Drive'] * len(cycle_data)
        phases[max_leg_pos:] = ['Recovery'] * (len(cycle_data) - max_leg_pos)
        
        return phases
    
    def calculate_biomechanical_metrics(self):
        """Calculate comprehensive biomechanical metrics"""
        print("\n🔍 Calculating Biomechanical Metrics...")
        
        if not self.stroke_cycles:
            print("❌ No stroke cycles identified")
            return
        
        # Calculate metrics for each cycle
        cycle_metrics = []
        
        for i, cycle in enumerate(self.stroke_cycles):
            metrics = self._analyze_single_cycle(cycle, i)
            cycle_metrics.append(metrics)
        
        # Calculate overall statistics
        self._calculate_overall_statistics(cycle_metrics)
        
        # Calculate technique consistency
        self._calculate_consistency_metrics(cycle_metrics)
        
        # Calculate efficiency metrics
        self._calculate_efficiency_metrics(cycle_metrics)
        
        print("✅ Biomechanical metrics calculated")
    
    def _analyze_single_cycle(self, cycle, cycle_id):
        """Analyze a single stroke cycle"""
        metrics = {'cycle_id': cycle_id}
        
        # Drive phase analysis
        drive_phase = cycle[cycle['cycle_phase'] == 'Drive']
        recovery_phase = cycle[cycle['cycle_phase'] == 'Recovery']
        
        if len(drive_phase) > 0:
            # Drive phase metrics
            metrics['drive_duration'] = len(drive_phase)
            metrics['drive_leg_range'] = drive_phase['left_leg_angle'].max() - drive_phase['left_leg_angle'].min()
            metrics['drive_arm_range'] = drive_phase['left_arm_angle'].max() - drive_phase['left_arm_angle'].min()
            metrics['drive_leg_velocity'] = self._calculate_angular_velocity(drive_phase['left_leg_angle'])
            metrics['drive_arm_velocity'] = self._calculate_angular_velocity(drive_phase['left_arm_angle'])
        
        if len(recovery_phase) > 0:
            # Recovery phase metrics
            metrics['recovery_duration'] = len(recovery_phase)
            metrics['recovery_leg_range'] = recovery_phase['left_leg_angle'].max() - recovery_phase['left_leg_angle'].min()
            metrics['recovery_arm_range'] = recovery_phase['left_arm_angle'].max() - recovery_phase['left_arm_angle'].min()
            metrics['recovery_leg_velocity'] = self._calculate_angular_velocity(recovery_phase['left_leg_angle'])
            metrics['recovery_arm_velocity'] = self._calculate_angular_velocity(recovery_phase['left_arm_angle'])
        
        # Overall cycle metrics
        metrics['total_duration'] = len(cycle)
        metrics['leg_range_total'] = cycle['left_leg_angle'].max() - cycle['left_leg_angle'].min()
        metrics['arm_range_total'] = cycle['left_arm_angle'].max() - cycle['left_arm_angle'].min()
        
        # Symmetry metrics
        metrics['arm_symmetry'] = abs(cycle['left_arm_angle'].mean() - cycle['right_arm_angle'].mean())
        metrics['leg_symmetry'] = abs(cycle['left_leg_angle'].mean() - cycle['right_leg_angle'].mean())
        
        return metrics
    
    def _calculate_angular_velocity(self, angles):
        """Calculate angular velocity from angle data"""
        if len(angles) < 2:
            return 0
        return np.mean(np.abs(np.diff(angles)))
    
    def _calculate_overall_statistics(self, cycle_metrics):
        """Calculate overall statistics from cycle metrics"""
        df_cycles = pd.DataFrame(cycle_metrics)
        
        self.analysis_results['overall_stats'] = {
            'total_cycles': len(cycle_metrics),
            'avg_drive_duration': df_cycles['drive_duration'].mean(),
            'avg_recovery_duration': df_cycles['recovery_duration'].mean(),
            'avg_drive_recovery_ratio': df_cycles['drive_duration'].mean() / df_cycles['recovery_duration'].mean(),
            'avg_leg_range': df_cycles['leg_range_total'].mean(),
            'avg_arm_range': df_cycles['arm_range_total'].mean(),
            'avg_arm_symmetry': df_cycles['arm_symmetry'].mean(),
            'avg_leg_symmetry': df_cycles['leg_symmetry'].mean(),
            'avg_drive_leg_velocity': df_cycles['drive_leg_velocity'].mean(),
            'avg_drive_arm_velocity': df_cycles['drive_arm_velocity'].mean()
        }
    
    def _calculate_consistency_metrics(self, cycle_metrics):
        """Calculate technique consistency metrics"""
        df_cycles = pd.DataFrame(cycle_metrics)
        
        consistency_metrics = {}
        
        # Calculate coefficient of variation (CV) for key metrics
        key_metrics = ['drive_duration', 'recovery_duration', 'leg_range_total', 'arm_range_total']
        
        for metric in key_metrics:
            if metric in df_cycles.columns:
                mean_val = df_cycles[metric].mean()
                std_val = df_cycles[metric].std()
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                consistency_metrics[f'{metric}_cv'] = cv
                consistency_metrics[f'{metric}_consistency'] = 100 - cv  # Higher is more consistent
        
        self.analysis_results['consistency_metrics'] = consistency_metrics
    
    def _calculate_efficiency_metrics(self, cycle_metrics):
        """Calculate rowing efficiency metrics"""
        df_cycles = pd.DataFrame(cycle_metrics)
        
        efficiency_metrics = {}
        
        # Calculate drive efficiency (leg work vs arm work)
        if 'drive_leg_velocity' in df_cycles.columns and 'drive_arm_velocity' in df_cycles.columns:
            leg_work = df_cycles['drive_leg_velocity'] * df_cycles['drive_leg_range']
            arm_work = df_cycles['drive_arm_velocity'] * df_cycles['drive_arm_range']
            
            efficiency_metrics['leg_arm_work_ratio'] = (leg_work / arm_work).mean()
            efficiency_metrics['drive_efficiency'] = leg_work.mean() / (leg_work.mean() + arm_work.mean())
        
        # Calculate phase timing efficiency
        if 'drive_duration' in df_cycles.columns and 'recovery_duration' in df_cycles.columns:
            ideal_ratio = 0.4  # Ideal drive:recovery ratio
            actual_ratio = df_cycles['drive_duration'] / df_cycles['recovery_duration']
            efficiency_metrics['timing_efficiency'] = 100 - abs(actual_ratio - ideal_ratio).mean() * 100
        
        self.analysis_results['efficiency_metrics'] = efficiency_metrics
    
    def create_professional_report(self):
        """Create a professional analysis report"""
        print("\n📊 Creating Professional Analysis Report...")
        
        # Create output directory
        os.makedirs('professional_analysis', exist_ok=True)
        
        # Create detailed report
        self._create_detailed_report()
        
        # Create advanced visualizations
        self._create_advanced_visualizations()
        
        # Create technique recommendations
        self._create_technique_recommendations()
        
        print("✅ Professional analysis report created")
    
    def _create_detailed_report(self):
        """Create detailed text report"""
        report_path = 'professional_analysis/detailed_rowing_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("🚣‍♂️ PROFESSIONAL ROWING ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Stroke Cycles Analyzed: {len(self.stroke_cycles)}\n\n")
            
            # Overall Statistics
            if 'overall_stats' in self.analysis_results:
                f.write("OVERALL PERFORMANCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                stats = self.analysis_results['overall_stats']
                f.write(f"Average Drive Duration: {stats['avg_drive_duration']:.1f} frames\n")
                f.write(f"Average Recovery Duration: {stats['avg_recovery_duration']:.1f} frames\n")
                f.write(f"Drive/Recovery Ratio: {stats['avg_drive_recovery_ratio']:.2f}\n")
                f.write(f"Average Leg Range of Motion: {stats['avg_leg_range']:.1f}°\n")
                f.write(f"Average Arm Range of Motion: {stats['avg_arm_range']:.1f}°\n")
                f.write(f"Average Arm Symmetry: {stats['avg_arm_symmetry']:.1f}°\n")
                f.write(f"Average Leg Symmetry: {stats['avg_leg_symmetry']:.1f}°\n\n")
            
            # Consistency Metrics
            if 'consistency_metrics' in self.analysis_results:
                f.write("TECHNIQUE CONSISTENCY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                consistency = self.analysis_results['consistency_metrics']
                
                for metric, value in consistency.items():
                    if 'consistency' in metric:
                        f.write(f"{metric.replace('_', ' ').title()}: {value:.1f}%\n")
                f.write("\n")
            
            # Efficiency Metrics
            if 'efficiency_metrics' in self.analysis_results:
                f.write("EFFICIENCY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                efficiency = self.analysis_results['efficiency_metrics']
                
                for metric, value in efficiency.items():
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.2f}\n")
                f.write("\n")
            
            # Force Analysis (if available)
            if self.force_data:
                f.write("FORCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                powers = [f['power'] for f in self.force_data]
                f.write(f"Average Power: {np.mean(powers):.1f}W\n")
                f.write(f"Maximum Power: {np.max(powers):.1f}W\n")
                f.write(f"Power Standard Deviation: {np.std(powers):.1f}W\n")
                f.write(f"Power Consistency: {100 - (np.std(powers)/np.mean(powers)*100):.1f}%\n\n")
        
        print(f"   📋 Detailed report: {report_path}")
    
    def _create_advanced_visualizations(self):
        """Create advanced visualization plots"""
        if not self.stroke_cycles:
            return
        
        # Set up plotting style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        # Create comprehensive analysis figure
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('🚣‍♂️ Professional Rowing Biomechanical Analysis', fontsize=24, color='white')
        
        # 1. Stroke Cycle Analysis
        ax1 = plt.subplot(3, 4, 1)
        cycle_durations = [len(cycle) for cycle in self.stroke_cycles]
        ax1.plot(range(len(cycle_durations)), cycle_durations, 'lime', linewidth=2, marker='o')
        ax1.set_title('Stroke Cycle Duration Consistency', color='white', fontsize=14)
        ax1.set_xlabel('Cycle Number', color='white')
        ax1.set_ylabel('Duration (frames)', color='white')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drive vs Recovery Timing
        ax2 = plt.subplot(3, 4, 2)
        drive_durations = []
        recovery_durations = []
        
        for cycle in self.stroke_cycles:
            drive_phase = cycle[cycle['cycle_phase'] == 'Drive']
            recovery_phase = cycle[cycle['cycle_phase'] == 'Recovery']
            drive_durations.append(len(drive_phase))
            recovery_durations.append(len(recovery_phase))
        
        ax2.scatter(drive_durations, recovery_durations, alpha=0.7, s=50)
        ax2.set_title('Drive vs Recovery Duration', color='white', fontsize=14)
        ax2.set_xlabel('Drive Duration (frames)', color='white')
        ax2.set_ylabel('Recovery Duration (frames)', color='white')
        ax2.grid(True, alpha=0.3)
        
        # 3. Range of Motion Analysis
        ax3 = plt.subplot(3, 4, 3)
        leg_ranges = [cycle['left_leg_angle'].max() - cycle['left_leg_angle'].min() for cycle in self.stroke_cycles]
        arm_ranges = [cycle['left_arm_angle'].max() - cycle['left_arm_angle'].min() for cycle in self.stroke_cycles]
        
        ax3.plot(range(len(leg_ranges)), leg_ranges, 'cyan', label='Leg ROM', linewidth=2)
        ax3.plot(range(len(arm_ranges)), arm_ranges, 'yellow', label='Arm ROM', linewidth=2)
        ax3.set_title('Range of Motion Consistency', color='white', fontsize=14)
        ax3.set_xlabel('Cycle Number', color='white')
        ax3.set_ylabel('Range of Motion (degrees)', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Symmetry Analysis
        ax4 = plt.subplot(3, 4, 4)
        arm_symmetry = [abs(cycle['left_arm_angle'].mean() - cycle['right_arm_angle'].mean()) for cycle in self.stroke_cycles]
        leg_symmetry = [abs(cycle['left_leg_angle'].mean() - cycle['right_leg_angle'].mean()) for cycle in self.stroke_cycles]
        
        ax4.plot(range(len(arm_symmetry)), arm_symmetry, 'red', label='Arm Asymmetry', linewidth=2)
        ax4.plot(range(len(leg_symmetry)), leg_symmetry, 'orange', label='Leg Asymmetry', linewidth=2)
        ax4.set_title('Bilateral Symmetry', color='white', fontsize=14)
        ax4.set_xlabel('Cycle Number', color='white')
        ax4.set_ylabel('Asymmetry (degrees)', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Phase Analysis Heatmap
        ax5 = plt.subplot(3, 4, 5)
        phase_data = []
        for cycle in self.stroke_cycles[:10]:  # Show first 10 cycles
            cycle_phases = []
            for _, row in cycle.iterrows():
                if row['cycle_phase'] == 'Drive':
                    cycle_phases.append(1)
                else:
                    cycle_phases.append(0)
            phase_data.append(cycle_phases)
        
        if phase_data:
            phase_matrix = np.array(phase_data)
            im = ax5.imshow(phase_matrix, cmap='RdYlBu', aspect='auto')
            ax5.set_title('Stroke Phase Heatmap', color='white', fontsize=14)
            ax5.set_xlabel('Frame in Cycle', color='white')
            ax5.set_ylabel('Cycle Number', color='white')
            ax5.set_yticks(range(len(phase_data)))
            ax5.set_yticklabels([f'Cycle {i+1}' for i in range(len(phase_data))], color='white')
        
        # 6. Angular Velocity Analysis
        ax6 = plt.subplot(3, 4, 6)
        drive_leg_velocities = []
        drive_arm_velocities = []
        
        for cycle in self.stroke_cycles:
            drive_phase = cycle[cycle['cycle_phase'] == 'Drive']
            if len(drive_phase) > 1:
                leg_vel = np.mean(np.abs(np.diff(drive_phase['left_leg_angle'])))
                arm_vel = np.mean(np.abs(np.diff(drive_phase['left_arm_angle'])))
                drive_leg_velocities.append(leg_vel)
                drive_arm_velocities.append(arm_vel)
        
        ax6.scatter(drive_leg_velocities, drive_arm_velocities, alpha=0.7, s=50)
        ax6.set_title('Drive Phase Angular Velocities', color='white', fontsize=14)
        ax6.set_xlabel('Leg Angular Velocity', color='white')
        ax6.set_ylabel('Arm Angular Velocity', color='white')
        ax6.grid(True, alpha=0.3)
        
        # 7. Consistency Metrics
        ax7 = plt.subplot(3, 4, 7)
        if 'consistency_metrics' in self.analysis_results:
            consistency = self.analysis_results['consistency_metrics']
            consistency_scores = [v for k, v in consistency.items() if 'consistency' in k]
            metric_names = [k.replace('_consistency', '').replace('_', ' ').title() for k, v in consistency.items() if 'consistency' in k]
            
            bars = ax7.bar(range(len(consistency_scores)), consistency_scores, color='lime', alpha=0.7)
            ax7.set_title('Technique Consistency Scores', color='white', fontsize=14)
            ax7.set_ylabel('Consistency (%)', color='white')
            ax7.set_xticks(range(len(metric_names)))
            ax7.set_xticklabels(metric_names, rotation=45, ha='right', color='white')
            ax7.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, consistency_scores):
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
        
        # 9. Efficiency Metrics
        ax9 = plt.subplot(3, 4, 9)
        if 'efficiency_metrics' in self.analysis_results:
            efficiency = self.analysis_results['efficiency_metrics']
            eff_scores = list(efficiency.values())
            eff_names = [k.replace('_', ' ').title() for k in efficiency.keys()]
            
            bars = ax9.bar(range(len(eff_scores)), eff_scores, color='cyan', alpha=0.7)
            ax9.set_title('Efficiency Metrics', color='white', fontsize=14)
            ax9.set_ylabel('Efficiency Score', color='white')
            ax9.set_xticks(range(len(eff_names)))
            ax9.set_xticklabels(eff_names, rotation=45, ha='right', color='white')
            ax9.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, eff_scores):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 10. Performance Summary
        ax10 = plt.subplot(3, 4, 10)
        ax10.axis('off')
        
        summary_text = "PERFORMANCE SUMMARY\n\n"
        
        if 'overall_stats' in self.analysis_results:
            stats = self.analysis_results['overall_stats']
            summary_text += f"Total Cycles: {stats['total_cycles']}\n"
            summary_text += f"Drive/Recovery Ratio: {stats['avg_drive_recovery_ratio']:.2f}\n"
            summary_text += f"Leg ROM: {stats['avg_leg_range']:.1f}°\n"
            summary_text += f"Arm ROM: {stats['avg_arm_range']:.1f}°\n"
            summary_text += f"Arm Symmetry: {stats['avg_arm_symmetry']:.1f}°\n"
            summary_text += f"Leg Symmetry: {stats['avg_leg_symmetry']:.1f}°\n"
        
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
        
        if 'overall_stats' in self.analysis_results:
            stats = self.analysis_results['overall_stats']
            
            if stats['avg_drive_recovery_ratio'] < 0.3:
                recommendations += "• Increase drive phase duration\n"
            elif stats['avg_drive_recovery_ratio'] > 0.6:
                recommendations += "• Increase recovery phase duration\n"
            
            if stats['avg_arm_symmetry'] > 10:
                recommendations += "• Focus on arm symmetry\n"
            
            if stats['avg_leg_symmetry'] > 10:
                recommendations += "• Focus on leg symmetry\n"
        
        if 'consistency_metrics' in self.analysis_results:
            consistency = self.analysis_results['consistency_metrics']
            low_consistency = [k for k, v in consistency.items() if 'consistency' in k and v < 80]
            if low_consistency:
                recommendations += "• Improve technique consistency\n"
        
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
        quality_text += f"Stroke Cycles: {len(self.stroke_cycles)}\n"
        
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
        plt.savefig('professional_analysis/professional_rowing_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print("   📊 Professional analysis plot: professional_analysis/professional_rowing_analysis.png")
    
    def _create_technique_recommendations(self):
        """Create detailed technique recommendations"""
        recommendations_path = 'professional_analysis/technique_recommendations.txt'
        
        with open(recommendations_path, 'w') as f:
            f.write("🚣‍♂️ ROWING TECHNIQUE RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            
            if 'overall_stats' in self.analysis_results:
                stats = self.analysis_results['overall_stats']
                
                f.write("TIMING RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                if stats['avg_drive_recovery_ratio'] < 0.3:
                    f.write("• Your drive phase is too short relative to recovery\n")
                    f.write("• Focus on accelerating through the drive phase\n")
                    f.write("• Maintain connection between legs, body, and arms\n")
                elif stats['avg_drive_recovery_ratio'] > 0.6:
                    f.write("• Your drive phase is too long relative to recovery\n")
                    f.write("• Focus on quick, efficient recovery\n")
                    f.write("• Don't rush the drive - maintain good technique\n")
                else:
                    f.write("• Your drive/recovery timing is well balanced\n")
                f.write("\n")
                
                f.write("SYMMETRY RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                if stats['avg_arm_symmetry'] > 10:
                    f.write("• Significant arm asymmetry detected\n")
                    f.write("• Focus on equal arm movement\n")
                    f.write("• Check for dominant side compensation\n")
                else:
                    f.write("• Good arm symmetry maintained\n")
                
                if stats['avg_leg_symmetry'] > 10:
                    f.write("• Significant leg asymmetry detected\n")
                    f.write("• Focus on equal leg drive\n")
                    f.write("• Check foot positioning and leg strength\n")
                else:
                    f.write("• Good leg symmetry maintained\n")
                f.write("\n")
            
            if 'consistency_metrics' in self.analysis_results:
                consistency = self.analysis_results['consistency_metrics']
                low_consistency = [k for k, v in consistency.items() if 'consistency' in k and v < 80]
                
                if low_consistency:
                    f.write("CONSISTENCY RECOMMENDATIONS:\n")
                    f.write("-" * 30 + "\n")
                    f.write("• Technique consistency needs improvement\n")
                    f.write("• Focus on maintaining consistent stroke patterns\n")
                    f.write("• Practice with metronome for rhythm\n")
                    f.write("• Video analysis can help identify variations\n")
                    f.write("\n")
            
            if 'efficiency_metrics' in self.analysis_results:
                efficiency = self.analysis_results['efficiency_metrics']
                
                f.write("EFFICIENCY RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                if 'drive_efficiency' in efficiency:
                    if efficiency['drive_efficiency'] < 0.6:
                        f.write("• Drive efficiency can be improved\n")
                        f.write("• Focus on leg drive initiation\n")
                        f.write("• Maintain connection through the stroke\n")
                    else:
                        f.write("• Good drive efficiency maintained\n")
                f.write("\n")
            
            f.write("GENERAL RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("• Regular video analysis helps identify technique changes\n")
            f.write("• Focus on one aspect at a time for improvement\n")
            f.write("• Consistent practice leads to better technique\n")
            f.write("• Consider working with a coach for personalized feedback\n")
        
        print(f"   📋 Technique recommendations: {recommendations_path}")

def main():
    """Main analysis function"""
    print("🚣‍♂️ Professional Rowing Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ProfessionalRowingAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        print("❌ Cannot proceed without data")
        return
    
    # Identify stroke cycles
    analyzer.identify_stroke_cycles()
    
    # Calculate biomechanical metrics
    analyzer.calculate_biomechanical_metrics()
    
    # Create professional report
    analyzer.create_professional_report()
    
    print("\n🎉 Professional rowing analysis complete!")
    print("📁 Check the 'professional_analysis/' directory for results")

if __name__ == "__main__":
    main()
