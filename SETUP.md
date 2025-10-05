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
