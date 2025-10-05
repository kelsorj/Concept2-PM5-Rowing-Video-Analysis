# Rowing Biomechanical Analysis System - Complete Instructions

## Overview

This system provides comprehensive biomechanical analysis of rowing technique by combining:
- **Video capture** with embedded timestamps
- **PM5 force data** capture and synchronization
- **Pose estimation** using YOLO11
- **Synchronized overlay videos** with animated force curves and body angles
- **Comprehensive reports** with force-angle correlations

## Prerequisites

### Hardware Requirements
- **Concept2 PM5** rowing machine
- **Computer** with USB port for PM5 connection
- **Camera** (webcam or external camera)
- **USB cable** to connect PM5 to computer

### Software Requirements
- **Python 3.8+** with the following packages:
  - `opencv-python`
  - `ultralytics` (for YOLO11 pose estimation)
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `scipy`
  - `py3row` (for PM5 communication)
- **FFmpeg** (for video encoding with metadata)

### Installation
```bash
pip install opencv-python ultralytics numpy matplotlib pandas scipy py3row
# Install FFmpeg (varies by OS)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

## Step 1: Video and Force Data Capture

### 1.1 Setup
1. **Connect PM5** to computer via USB cable
2. **Position camera** to capture side view of rower (recommended: right side view)
3. **Ensure good lighting** for pose detection
4. **Start PM5** and select a workout mode (recommended: "Just Row" or "Workout")

### 1.2 Run Capture Script
```bash
sudo python py3row_usb_video_capture.py
```

**Important Notes:**
- The script will create a timestamped directory (e.g., `py3rowcap_20251004_123520/`)
- **Start rowing** when prompted - the script captures both video and force data simultaneously
- **Stop rowing** and press Ctrl+C to end capture
- The script generates:
  - `py3rowcap_YYYYMMDD_HHMMSS.mp4` - Video with embedded timestamps
  - `py3rowcap_YYYYMMDD_HHMMSS_raw.csv` - Raw PM5 data
  - `py3rowcap_YYYYMMDD_HHMMSS_frames.csv` - Frame timestamps

### 1.3 Verify Capture
Check that the generated files contain data:
```bash
# Check video file
ls -la py3rowcap_*/py3rowcap_*.mp4

# Check force data (should have multiple rows)
wc -l py3rowcap_*/py3rowcap_*_raw.csv

# Check frame timestamps
head -5 py3rowcap_*/py3rowcap_*_frames.csv
```

## Step 2: Complete Biomechanical Analysis

### 2.1 Run Complete Analysis
```bash
python create_complete_kinematics_overlay.py --session-dir py3rowcap_YYYYMMDD_HHMMSS
```

**What this does:**
1. **Automatically finds** the video and CSV files in the session directory
2. **Runs kinematics analysis** on the video using YOLO11 pose estimation
3. **Processes force data** and combines Drive + Dwelling phases into complete strokes
4. **Creates synchronized overlay video** with:
   - Animated force curves (red dot moving through stroke)
   - Skeleton overlay with joint angle badges
   - Text display with all body angles
5. **Generates comprehensive report** with statistics and correlations

**Note:** The script automatically creates an `analysis_` directory in your current working directory to avoid permission issues with the original capture directory.

### 2.2 Output Files
The analysis creates a new directory `analysis_py3rowcap_YYYYMMDD_HHMMSS/` with:
- `complete_kinematics_overlay_YYYYMMDD_HHMMSS.mp4` - **Final overlay video**
- `pose_data_YYYYMMDD_HHMMSS.json` - Pose estimation data
- `rowing_analysis_report_YYYYMMDD_HHMMSS.txt` - **Comprehensive report**
- `rowing_analysis_data_YYYYMMDD_HHMMSS.csv` - **Detailed CSV data**

## Step 3: Understanding the Results

### 3.1 Overlay Video Features
The final video includes:

**Skeleton Overlay:**
- **Yellow badges**: Elbow angles and ankle vertical angles
- **Orange badges**: Knee angles
- **Gray badges**: Hip angles
- **Green badge**: Back vertical angle (torso lean)

**Force Curve Display:**
- **Animated red dot**: Shows current position in stroke
- **Peak force**: Maximum force achieved
- **Current force**: Force at current video frame
- **Stroke number**: Which stroke is being analyzed

**Text Display:**
- **Frame number** and **elapsed time**
- **All body angles** with consistent signs
- **Vertical angles**: Ankle and back angles relative to vertical

### 3.2 Report Analysis
The comprehensive report includes:

**Stroke Summary:**
- Duration, peak force, power, stroke rate for each stroke
- Timing information and phase breakdown

**Body Angle Statistics:**
- Mean, standard deviation, min, max, range for each angle
- Count of valid measurements

**Force-Angle Correlations:**
- Per-stroke analysis showing average angles during each stroke
- Force curve metrics correlated with body position

### 3.3 CSV Data Structure
The detailed CSV contains:
- `frame_number`: Video frame index
- `timestamp`: Absolute timestamp
- `left_arm_angle`, `right_arm_angle`: Elbow joint angles
- `left_leg_angle`, `right_leg_angle`: Knee joint angles
- `back_vertical_angle`: Torso lean relative to vertical (signed)
- `left_ankle_vertical_angle`, `right_ankle_vertical_angle`: Ankle angles relative to vertical (signed)
- `force_peak`, `force_avg`: Force metrics for the stroke
- `stroke_number`, `stroke_phase`: Which stroke and phase (Drive/Dwelling/Recovery)

## Step 4: Troubleshooting

### 4.1 Common Issues

**No Force Data Captured:**
- Ensure PM5 is in a workout mode (not just displaying)
- Check USB connection
- Verify PM5 is actively rowing (not just sitting idle)

**Poor Pose Detection:**
- Ensure good lighting
- Position camera for clear side view
- Avoid cluttered backgrounds
- Ensure rower is fully visible in frame

**Synchronization Issues:**
- The system uses embedded video timestamps for perfect sync
- If issues persist, check that the capture script completed successfully

**Missing Angles:**
- Some angles may be missing if pose detection confidence is low
- The system only displays angles with >50% confidence
- Check lighting and camera positioning

### 4.2 Performance Tips

**For Better Results:**
- Use consistent lighting
- Position camera at rower's side (not front/back)
- Ensure full body is visible
- Use a stable camera mount
- Row at consistent pace for better analysis

**For Faster Processing:**
- Shorter video segments (2-5 minutes) process faster
- The system processes ~30 frames per second
- Full analysis typically takes 2-3x the video duration

## Step 5: Advanced Usage

### 5.1 Custom Analysis
You can modify the analysis by editing `create_complete_kinematics_overlay.py`:
- Adjust angle calculation methods
- Modify overlay display options
- Change report generation parameters

### 5.2 Batch Processing
To analyze multiple sessions:
```bash
# Create a batch script
for session in py3rowcap_*/; do
  python create_complete_kinematics_overlay.py --session-dir "$session"
done
```

### 5.3 Data Export
The CSV data can be imported into:
- **Excel/Google Sheets** for manual analysis
- **R/Python** for statistical analysis
- **MATLAB** for signal processing
- **Custom analysis tools**

## File Structure

```
rowingIA/
├── py3row_usb_video_capture.py          # Main capture script
├── create_complete_kinematics_overlay.py # Main analysis script
├── instructions.md                       # This file
├── py3rowcap_YYYYMMDD_HHMMSS/           # Capture session directory
│   ├── py3rowcap_YYYYMMDD_HHMMSS.mp4    # Original video with timestamps
│   ├── py3rowcap_YYYYMMDD_HHMMSS_raw.csv # Raw PM5 data
│   └── py3rowcap_YYYYMMDD_HHMMSS_frames.csv # Frame timestamps
└── analysis_py3rowcap_YYYYMMDD_HHMMSS/  # Analysis output directory
    ├── complete_kinematics_overlay_YYYYMMDD_HHMMSS.mp4 # Final overlay video
    ├── pose_data_YYYYMMDD_HHMMSS.json   # Pose estimation data
    ├── rowing_analysis_report_YYYYMMDD_HHMMSS.txt # Comprehensive report
    └── rowing_analysis_data_YYYYMMDD_HHMMSS.csv # Detailed CSV data
```

## Quick Start Summary

1. **Connect PM5** via USB and start a workout
2. **Run capture**: `sudo python py3row_usb_video_capture.py`
3. **Row for 2-5 minutes** then stop capture
4. **Run analysis**: `python create_complete_kinematics_overlay.py --session-dir py3rowcap_YYYYMMDD_HHMMSS`
5. **Review results** in the generated overlay video and report

## Support

For issues or questions:
- Check the troubleshooting section above
- Verify all prerequisites are installed
- Ensure PM5 is properly connected and in workout mode
- Check that video capture completed successfully

The system provides comprehensive biomechanical analysis with perfect synchronization between video, force data, and body angles for detailed rowing technique analysis.
