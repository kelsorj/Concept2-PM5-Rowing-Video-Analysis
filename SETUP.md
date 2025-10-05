# Rowing Biomechanical Analysis System - Setup Guide

## Quick Installation

```bash
# Clone or download this repository
cd rowingIA

# Install Python dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

## Detailed Requirements

### Python Dependencies

The `requirements.txt` file includes all necessary Python packages:

- **opencv-python** - Computer vision and video processing
- **ultralytics** - YOLO11 pose estimation models
- **numpy** - Numerical computing
- **pandas** - Data analysis and CSV handling
- **scipy** - Scientific computing (for smoothing algorithms)
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **py3row** - Concept2 PM5 rowing machine communication
- **Pillow** - Image processing

### System Requirements

- **Python 3.8+** (tested with Python 3.13)
- **FFmpeg** - For video encoding with metadata
- **USB port** - For PM5 connection
- **Camera** - Webcam or external camera

### Hardware Requirements

- **Concept2 PM5** rowing machine
- **Computer** with USB port
- **Camera** (webcam or external camera)
- **USB cable** to connect PM5 to computer

## Installation Steps

### 1. Python Environment (Recommended)

```bash
# Create virtual environment
python -m venv rowing_env

# Activate virtual environment
# macOS/Linux:
source rowing_env/bin/activate
# Windows:
# rowing_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Test YOLO model download
python -c "from ultralytics import YOLO; model = YOLO('yolo11n-pose.pt'); print('✅ YOLO11 model ready')"

# Test OpenCV
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} ready')"

# Test PM5 communication (with PM5 connected)
python -c "import py3row; print('✅ PM5 communication ready')"
```

### 3. Download Pose Models

The system will automatically download YOLO11 pose models on first use:
- `yolo11n-pose.pt` (nano - default, fastest)
- `yolo11m-pose.pt` (medium - better accuracy)
- `yolo11l-pose.pt` (large - highest accuracy)

## Troubleshooting

### Common Issues

**"No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**"FFmpeg not found"**
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: Download from https://ffmpeg.org/

**"Permission denied" for USB**
- macOS/Linux: Run with `sudo` for USB access
- Windows: Install USB drivers if needed

**"PM5 not detected"**
- Ensure PM5 is in workout mode (not just displaying)
- Check USB cable connection
- Try different USB port

### Performance Optimization

**For faster processing:**
- Use `yolo11n-pose.pt` (nano model)
- Process shorter video segments (2-5 minutes)
- Ensure good lighting for pose detection

**For better accuracy:**
- Use `yolo11m-pose.pt` or `yolo11l-pose.pt`
- Ensure good camera positioning
- Use consistent lighting

## Usage

Once installed, follow the main instructions in `instructions.md`:

1. **Capture data**: `sudo python py3row_usb_video_capture.py`
2. **Analyze**: `python create_complete_kinematics_overlay.py --session-dir py3rowcap_YYYYMMDD_HHMMSS`

## Support

For issues:
1. Check this setup guide
2. Verify all requirements are installed
3. Ensure PM5 is properly connected
4. Check camera permissions and lighting

---

# Complete Usage Instructions

## Overview

This system provides comprehensive biomechanical analysis of rowing technique by combining:
- **Video capture** with embedded timestamps
- **PM5 force data** capture and synchronization
- **Pose estimation** using YOLO11 with advanced smoothing
- **Synchronized overlay videos** with animated force curves and body angles
- **Comprehensive reports** with force-angle correlations

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
2. **Runs kinematics analysis** on the video using YOLO11 pose estimation with smoothing
3. **Processes force data** and combines Drive + Dwelling phases into complete strokes
4. **Creates synchronized overlay video** with:
   - Animated force curves (red dot moving through stroke)
   - Smoothed skeleton overlay with joint angle badges
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

**Smoothed Skeleton Overlay:**
- **Green skeleton lines** - Smooth, stable pose detection
- **White keypoints** - Joint markers
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
- Adjust smoothing parameters for different stability/responsiveness

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
├── requirements.txt                      # Python dependencies
├── SETUP.md                             # This file
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

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Connect PM5** via USB and start a workout
3. **Run capture**: `sudo python py3row_usb_video_capture.py`
4. **Row for 2-5 minutes** then stop capture
5. **Run analysis**: `python create_complete_kinematics_overlay.py --session-dir py3rowcap_YYYYMMDD_HHMMSS`
6. **Review results** in the generated overlay video and report

## Final Support

For issues or questions:
- Check the troubleshooting section above
- Verify all prerequisites are installed
- Ensure PM5 is properly connected and in workout mode
- Check that video capture completed successfully

The system provides comprehensive biomechanical analysis with perfect synchronization between video, force data, and body angles for detailed rowing technique analysis.
