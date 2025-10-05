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
# Standard capture at 30 FPS (default)
sudo python py3row_usb_video_capture.py

# High frame rate capture for detailed analysis
sudo python py3row_usb_video_capture.py --fps 60

# Custom frame rate (e.g., 24 FPS for slower processing)
sudo python py3row_usb_video_capture.py --fps 24
```

**Frame Rate Options:**
- **30 FPS (default)**: Good balance of detail and processing speed
- **60 FPS**: Higher detail for precise biomechanical analysis (larger files, longer processing)
- **24 FPS**: Faster processing, good for longer sessions
- **15 FPS**: Fastest processing, basic analysis

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
- `pm5_combined_strokes.json` - High-fidelity per-stroke force and timestamps (new)
- `py3rowcap_YYYYMMDD_HHMMSS_raw.csv` - A copy of the original PM5 raw CSV (for portability)
- `rowing_analysis_report_YYYYMMDD_HHMMSS.txt` - **Comprehensive report with coaching metrics** (new metrics)
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

**Coaching Metrics (per stroke):**
- Finish: layback angle, legs unbent, handle height at torso (%)
- Catch: shins angle, forward body angle, elbows unbent
- Sequence: Legs→Back separation %, Back→Arms separation %, Drive duration ratio %

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
- Ensure PM5 is in a "just row" mode (not just on), sometimes I've had to go back to menu and back into "just row"
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
3. **Run capture**: `sudo python py3row_usb_video_capture.py` (or `--fps 60` for higher detail)
4. **Row for 2-5 minutes** then stop capture
5. **Run comprehensive analysis**: `python comprehensive_stroke_analysis.py analysis_py3rowcap_YYYYMMDD_HHMMSS`
6. **Review results**:
   - **Overlay video**: `complete_kinematics_overlay_*.mp4`
   - **Comprehensive reports**: `comprehensive_analyses/stroke_*_comprehensive_analysis.png`
   - **3D visualizations**: `3d_animations/interactive_axis_dashboard.html`
   - **Session summary**: `session_summary.png`

## 📸 Combined Frames Generator

After running your analysis, you can generate combined frame visualizations for each stroke using the `simple_combined_frames.py` script.

### What It Does

The combined frames generator creates a single visualization for each stroke that shows:
- **6 key video frames** from each stroke (representing different phases)
- **Angle measurements overlaid** on each frame
- **Phase labels** (Catch, Drive Start, Drive Mid, Drive End, Recovery, Finish)
- **Frame numbers** and timestamps

### Usage

```bash
# Generate combined frames for all strokes
python simple_combined_frames.py analysis_py3rowcap_20251004_123520
```

### Output

The script creates a `combined_frames` folder inside your analysis directory:
```
analysis_py3rowcap_20251004_123520/
├── combined_frames/
│   ├── stroke_01_combined_frames.png
│   ├── stroke_02_combined_frames.png
│   ├── stroke_03_combined_frames.png
│   └── ...
├── rowing_analysis_data_*.csv
├── pose_data_*.json
└── complete_kinematics_overlay_*.mp4
```

### What's Displayed

#### Angle Measurements
- **L Arm**: Left arm angle
- **R Arm**: Right arm angle  
- **L Leg**: Left leg angle
- **R Leg**: Right leg angle
- **Back**: Back vertical angle
- **L Ankle**: Left ankle vertical angle
- **R Ankle**: Right ankle vertical angle

#### Phase Labels
- **Catch**: Beginning of stroke
- **Drive Start**: Start of power phase
- **Drive Mid**: Middle of drive
- **Drive End**: End of drive phase
- **Recovery**: Recovery phase
- **Finish**: End of stroke

### Example Workflow

1. **Run your analysis**:
   ```bash
   python create_complete_kinematics_overlay.py --video your_video.mp4
   ```

2. **Generate combined frames**:
   ```bash
   python simple_combined_frames.py analysis_py3rowcap_20251004_123520
   ```

3. **View results**:
   ```bash
   open analysis_py3rowcap_20251004_123520/combined_frames/stroke_01_combined_frames.png
   ```

Perfect for coaching and technique analysis! 🚣‍♂️

## 📊 Comprehensive Stroke Analysis

For the most detailed analysis, use the `comprehensive_stroke_analysis.py` script that combines video frames with the "Speed & Sequence" plot.

### What It Does

The comprehensive analysis creates a single visualization that includes:
- **6 key video frames** from each stroke (top section)
- **Speed & Sequence plot** showing drive and recovery phases (bottom section)
- **Legs, Back, Arms, and Handle** contribution curves
- **Phase labels** and separation percentages
- **Peak timing analysis** for optimal stroke sequencing

### Usage

```bash
# Generate comprehensive analyses for all strokes
python comprehensive_stroke_analysis.py analysis_py3rowcap_20251004_123520
```

### Output

The script creates a `comprehensive_analyses` folder inside your analysis directory:
```
analysis_py3rowcap_20251004_123520/
├── comprehensive_analyses/
│   ├── stroke_01_comprehensive_analysis.png
│   ├── stroke_02_comprehensive_analysis.png
│   └── ...
├── rowing_analysis_data_*.csv
├── pose_data_*.json
└── complete_kinematics_overlay_*.mp4
```

### Speed & Sequence Plot Features

The bottom section shows the ideal rowing sequence:
- **Drive Phase**: Legs → Back → Arms (left side)
- **Recovery Phase**: Arms → Back → Legs (right side)
- **Colored lines**: Green (Legs), Blue (Back), Magenta (Arms), Black dotted (Handle)
- **Force curve mapping**: Uses overlay-accurate mapping by default (matches PM5 overlay)
- **Peak labels**: L, B, A markers show timing of maximum contribution
- **Separation percentages**: Quantify how well movements are sequenced
- **Metrics table row**: Finish/Catch/Sequence values rendered under the plot (new)

This provides the most complete analysis for coaching and technique improvement! 🚣‍♂️

## 🎬 3D Biomechanical Visualizations

For advanced 3D analysis of rowing biomechanics, the system now includes interactive 3D visualizations that show stroke patterns in three-dimensional space.

### What It Does

The 3D visualization system creates:
- **Individual stroke animations** - Animated 3D plots showing each stroke's biomechanical path
- **Combined stroke analysis** - All strokes overlaid in a single 3D plot for comparison
- **Interactive axis dashboard** - Choose which angles to display on X, Y, and Z axes

### Usage

#### Option 1: Generate All 3D Visualizations (Recommended)
```bash
# This creates ALL visualizations in one command:
# - Individual stroke animations
# - Combined stroke analysis  
# - Interactive axis dashboard
python comprehensive_stroke_analysis.py analysis_py3rowcap_YYYYMMDD_HHMMSS
```

#### Option 2: Use Standalone 3D Tool
```bash
# Generate individual stroke animations
python create_3d_stroke_animation.py analysis_py3rowcap_YYYYMMDD_HHMMSS

# Generate combined analysis (all strokes in one plot)
python create_3d_stroke_animation.py analysis_py3rowcap_YYYYMMDD_HHMMSS --combined

# Generate interactive dashboard with axis selection
python create_3d_stroke_animation.py analysis_py3rowcap_YYYYMMDD_HHMMSS --interactive

# Generate animation for specific stroke only
python create_3d_stroke_animation.py analysis_py3rowcap_YYYYMMDD_HHMMSS --stroke 2
```

### Output

The 3D visualizations create a `3d_animations` folder:
```
analysis_py3rowcap_YYYYMMDD_HHMMSS/
├── 3d_animations/
│   ├── stroke_01_3d_animation.html      # Individual stroke animations
│   ├── stroke_02_3d_animation.html
│   ├── ...
│   ├── combined_stroke_analysis.html    # All strokes combined
│   └── interactive_axis_dashboard.html  # Interactive axis selector
├── comprehensive_analyses/
├── rowing_analysis_data_*.csv
└── pose_data_*.json
```

### 3D Visualization Features

#### Individual Stroke Animations
- **Animated 3D path** showing the stroke's biomechanical trajectory
- **Force color coding** - Color represents force (Newtons) at each point
- **Play/Pause controls** and time slider
- **Fixed axis ranges** for consistent viewing
- **Trail effect** showing the path taken so far

#### Combined Stroke Analysis
- **All strokes overlaid** in a single 3D plot
- **Different colors** for each stroke
- **Force color coding** across all strokes
- **Legend** showing stroke numbers
- **Fixed axis ranges** for easy comparison

#### Interactive Axis Dashboard
- **Radio button controls** to select which angle goes on each axis
- **Available angles**: Leg, Back, Arm (averaged and individual left/right)
- **Real-time updates** when changing axis selections
- **3D scatter plot** with force color coding
- **Hover information** showing exact values

### Axis Options

The interactive dashboard allows you to choose from:
- **Leg Angle (avg)** - Average of left and right leg angles
- **Back Angle** - Torso lean relative to vertical
- **Arm Angle (avg)** - Average of left and right arm angles
- **Left/Right Leg Angle** - Individual leg measurements
- **Left/Right Arm Angle** - Individual arm measurements

### Example Workflow

1. **Run comprehensive analysis** (includes 3D visualizations):
   ```bash
   python comprehensive_stroke_analysis.py analysis_py3rowcap_20251004_123520
   ```

2. **Open the interactive dashboard**:
   ```bash
   open analysis_py3rowcap_20251004_123520/3d_animations/interactive_axis_dashboard.html
   ```

3. **Experiment with different axis combinations**:
   - Try "Left Leg Angle" vs "Right Leg Angle" vs "Back Angle" to see asymmetry
   - Use "Leg Angle" vs "Back Angle" vs "Arm Angle" for classic sequence analysis
   - Compare individual vs averaged measurements

### Performance Notes

- **Processing time**: 3D visualizations add ~30-60 seconds to analysis time
- **File sizes**: HTML files are typically 1-5 MB each
- **Browser compatibility**: Works best in Chrome, Firefox, Safari
- **Interactive performance**: Smooth on modern computers, may be slower on older devices

Perfect for advanced biomechanical analysis and coaching! 🚣‍♂️

## Final Support

For issues or questions:
- Check the troubleshooting section above
- Verify all prerequisites are installed
- Ensure PM5 is properly connected and in workout mode
- Check that video capture completed successfully

The system provides comprehensive biomechanical analysis with perfect synchronization between video, force data, and body angles for detailed rowing technique analysis.
