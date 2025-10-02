#!/usr/bin/env python3
"""
Synchronized Rowing Data Capture Guide
Shows how to capture force and pose data simultaneously for perfect correlation
"""

import json
import csv
import time
from datetime import datetime

def create_demo_correlation():
    """Create a demonstration of what the correlation would look like with synchronized data"""
    print("🚣‍♂️ Creating Demo Force-Pose Correlation")
    print("=" * 50)
    
    # Create sample synchronized data
    demo_data = []
    
    # Simulate 10 strokes with realistic force and angle data
    for i in range(10):
        elapsed_s = 60 + i * 3.5  # 3.5 second intervals
        
        # Simulate force curve (typical rowing pattern)
        force_curve = []
        for j in range(30):  # 30 force points per stroke
            # Typical rowing force curve: starts low, peaks in middle, ends low
            progress = j / 29.0
            if progress < 0.3:  # Drive phase
                force = 50 + (progress / 0.3) * 150  # Ramp up
            elif progress < 0.7:  # Peak phase
                force = 200 - (progress - 0.3) / 0.4 * 50  # Peak then decline
            else:  # Recovery phase
                force = 150 - (progress - 0.7) / 0.3 * 150  # Rapid decline
        
            force_curve.append(int(force + (i % 3 - 1) * 20))  # Add some variation
        
        # Simulate body angles (realistic rowing ranges)
        left_arm_angle = 120 + (i % 3 - 1) * 15  # 105-135 degrees
        right_arm_angle = 125 + (i % 3 - 1) * 10  # 115-135 degrees
        left_leg_angle = 140 + (i % 3 - 1) * 20   # 120-160 degrees
        right_leg_angle = 135 + (i % 3 - 1) * 15  # 120-150 degrees
        torso_lean_angle = -15 + (i % 3 - 1) * 10  # -25 to -5 degrees
        
        demo_data.append({
            'elapsed_s': elapsed_s,
            'timestamp': f"2025-09-28T12:24:{int(elapsed_s % 60):02d}.000000",
            'force_curve': force_curve,
            'peak_force': max(force_curve),
            'avg_force': sum(force_curve) / len(force_curve),
            'left_arm_angle': left_arm_angle,
            'right_arm_angle': right_arm_angle,
            'left_leg_angle': left_leg_angle,
            'right_leg_angle': right_leg_angle,
            'torso_lean_angle': torso_lean_angle,
            'power': 150 + (i % 3 - 1) * 30,  # 120-180 watts
            'spm': 20 + (i % 3 - 1) * 2,     # 18-22 strokes per minute
        })
    
    # Save demo data
    with open('demo_synchronized_data.json', 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    # Create CSV version
    with open('demo_synchronized_data.csv', 'w', newline='') as f:
        fieldnames = ['elapsed_s', 'timestamp', 'peak_force', 'avg_force', 
                     'left_arm_angle', 'right_arm_angle', 'left_leg_angle', 
                     'right_leg_angle', 'torso_lean_angle', 'power', 'spm', 'force_curve_json']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in demo_data:
            csv_row = row.copy()
            csv_row['force_curve_json'] = json.dumps(row['force_curve'])
            del csv_row['force_curve']
            writer.writerow(csv_row)
    
    print("✅ Demo data created:")
    print("   📋 demo_synchronized_data.json")
    print("   📊 demo_synchronized_data.csv")
    
    return demo_data

def create_correlation_plots_demo(demo_data):
    """Create correlation plots using demo data"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("\n📊 Creating demo correlation plots...")
        
        # Create output directory
        import os
        os.makedirs('demo_correlation_plots', exist_ok=True)
        
        # 1. Force vs Body Angles
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Demo: Force vs Body Angles Correlation', fontsize=16)
        
        angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle', 'torso_lean_angle']
        
        for i, angle in enumerate(angles):
            row = i // 3
            col = i % 3
            
            angle_values = [d[angle] for d in demo_data]
            force_values = [d['peak_force'] for d in demo_data]
            
            axes[row, col].scatter(angle_values, force_values, alpha=0.7, s=100)
            axes[row, col].set_xlabel(f'{angle} (degrees)')
            axes[row, col].set_ylabel('Peak Force')
            axes[row, col].set_title(f'Peak Force vs {angle}')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(angle_values, force_values)[0, 1]
            axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', 
                              transform=axes[row, col].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig('demo_correlation_plots/force_vs_angles_demo.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Force Curves with Angles
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Demo: Force Curves with Body Angles', fontsize=16)
        
        for i in range(4):
            row = i // 2
            col = i % 2
            
            data = demo_data[i]
            force_curve = data['force_curve']
            
            x = np.arange(len(force_curve))
            axes[row, col].plot(x, force_curve, 'b-', linewidth=2, label='Force')
            axes[row, col].set_xlabel('Force Point')
            axes[row, col].set_ylabel('Force')
            axes[row, col].set_title(f'Stroke {i+1} - Peak: {data["peak_force"]:.1f}')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add angle information
            angle_text = f"L Arm: {data['left_arm_angle']:.1f}°\n"
            angle_text += f"R Arm: {data['right_arm_angle']:.1f}°\n"
            angle_text += f"L Leg: {data['left_leg_angle']:.1f}°\n"
            angle_text += f"R Leg: {data['right_leg_angle']:.1f}°\n"
            angle_text += f"Torso: {data['torso_lean_angle']:.1f}°"
            
            axes[row, col].text(0.02, 0.98, angle_text, 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('demo_correlation_plots/force_curves_with_angles_demo.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Demo plots saved to 'demo_correlation_plots/' directory")
        
    except ImportError:
        print("⚠️  matplotlib not available for plotting")

def print_synchronization_guide():
    """Print guide for capturing synchronized data"""
    print("\n" + "="*60)
    print("🎯 GUIDE: How to Capture Synchronized Force + Pose Data")
    print("="*60)
    
    print("""
📋 STEP-BY-STEP PROCESS:

1. 🚀 START BOTH CAPTURES SIMULTANEOUSLY:
   Terminal 1: sudo ./rowing_env/bin/python3 py3row_usb_capture.py
   Terminal 2: python3 simple_pose_export.py

2. ⏱️  TIMING IS CRITICAL:
   - Start both scripts within 1-2 seconds of each other
   - Use the same video file for pose analysis
   - Ensure video recording starts at the same time as force capture

3. 📹 VIDEO RECORDING:
   - Start video recording BEFORE starting force capture
   - Use consistent naming: YYYYMMDD_HHMMSS.mp4
   - Record for the same duration as your rowing session

4. 🔄 DATA SYNCHRONIZATION:
   - Force data: Uses elapsed time from workout start
   - Pose data: Uses elapsed time from video start
   - Both should start at t=0.0s for perfect correlation

5. 📊 ANALYSIS:
   - Run: python3 correlate_force_pose_elapsed.py
   - This will match force curves with body angles by elapsed time
   - Time differences should be < 0.5 seconds for good correlation

💡 TIPS FOR SUCCESS:
   - Practice the timing with a short test session first
   - Use a countdown timer to start both captures simultaneously
   - Keep video camera in a fixed position
   - Ensure good lighting for pose detection
   - Record at least 30 seconds of rowing for meaningful data

🎯 EXPECTED RESULTS:
   - 20-50 force curves per minute of rowing
   - 30-60 pose frames per second of video
   - Perfect time synchronization for correlation analysis
   - Detailed force-angle relationships for technique analysis
""")

def main():
    """Main function"""
    print("🚣‍♂️ Synchronized Rowing Data Capture Guide")
    print("=" * 50)
    
    # Create demo data
    demo_data = create_demo_correlation()
    
    # Create demo plots
    create_correlation_plots_demo(demo_data)
    
    # Print synchronization guide
    print_synchronization_guide()
    
    print(f"\n🎉 Demo complete! You now have:")
    print(f"   📋 Sample synchronized data files")
    print(f"   📊 Demo correlation plots")
    print(f"   📖 Complete synchronization guide")
    print(f"\n💡 Next time you row, follow the guide above for perfect force-pose correlation!")

if __name__ == "__main__":
    main()
