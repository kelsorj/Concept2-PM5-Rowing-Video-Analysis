# Rowing Data Visualization Dashboard

A real-time rowing data visualization dashboard that displays Concept2 PM5 rowing machine data with synchronized computer clock timing for video analysis.

## Features

- **Real-time Computer Clock**: Shows current computer time for video synchronization
- **Integrated BLE Capture**: Start/stop rowing data capture directly from the web interface
- **Live Rowing Metrics**: Displays elapsed time, distance, stroke rate, heart rate, pace, and power
- **Power Curve Visualization**: Shows instantaneous power, average power, and peak power over time
- **Multiple Charts**: Power curves, stroke rate, distance progress, and heart rate over time
- **Capture Status**: Real-time status indicators for BLE connection and data recording
- **Responsive Design**: Works on desktop and mobile devices

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies in virtual environment
python3 -m venv rowing_env
source rowing_env/bin/activate  # On Windows: rowing_env\Scripts\activate
pip install bleak git+https://github.com/droogmic/Py3Row.git

# Install libusb (required for USB access on macOS)
brew install libusb

# Note: USB access requires sudo permissions on macOS
```

### 2. Start the Data Server

```bash
npm run server
```

This starts the Express server on port 3001, serving both the API and React app.

### 3. Start the React Dashboard

In a new terminal window:

```bash
npm run server
```

This starts the Express server on port 3001, serving both the API and React app.

### 4. Open Dashboard

Navigate to `http://localhost:3001` in your browser.

## Usage

### Quick Start (USB Method)
1. **Connect PM5**: Use USB cable from PM5 to computer (square USB-B port on PM5)
2. **Open Dashboard**: Navigate to `http://localhost:3001` in your browser
3. **Run Capture Script**: In a separate terminal, run:
   ```bash
   ./start_rowing_capture.sh
   ```
4. **Start rowing** on your Concept2 PM5 with "Just Row" mode
5. **Monitor real-time data** as you row (including force curves!)
6. **Stop the script** with Ctrl+C when finished

### Manual Usage (if needed)
If you prefer running scripts manually:
```bash
source rowing_env/bin/activate
python3 enhanced_usb_c2.py
```

## Integrated USB Control

The dashboard now includes direct control over the rowing data capture:

- **Start Capture Button**: Launches the Python USB capture script
- **Stop Capture Button**: Gracefully stops data collection
- **Status Indicators**:
  - 🔴 **Recording**: Actively capturing data via USB
  - ⏸️ **Ready**: Ready to start capture
  - ⏳ **Starting/Stopping**: Transition states
  - ❌ **Error**: Connection or capture issues

**USB Connection Required**: Make sure your PM5 is connected via USB cable to your computer.

## Power Data Explanation

### Current Limitations
The original BLE script only captured **average power** which updates slowly. Power values appeared static because they represented workout averages, not instantaneous power.

### Enhanced USB Solution
The USB connection provides complete access to PM5 data:
- **Basic Metrics**: Time, distance, stroke rate, pace, heart rate ✅
- **Real-time Power**: Current power output (watts) ✅
- **Force Curves**: Actual force curve data from each stroke ✅
- **High Frequency**: Updates at 5Hz (every 200ms) ✅

### Data Sources
- **USB Direct Connection**: Full access to PM5 internal data ✅
- **Py3Row Library**: Professional rowing machine communication
- **Force Plot Data**: Raw force measurements from each stroke

### ✅ **USB Advantages over BLE**

**USB provides:**
- **Complete power data** including real-time watts
- **Force curves** - the actual power curve you need!
- **Higher update rate** (5Hz vs 1-2Hz BLE)
- **More reliable connection** (no Bluetooth interference)
- **All rowing metrics** without firmware limitations

**What you'll get:**
- Instantaneous power during each stroke
- Peak power calculations from force curves
- Complete rowing analytics
- Video synchronization with computer clock

## API Endpoints

- `GET /api/latest-data` - Get latest rowing data from CSV files
- `GET /api/time` - Get current server time
- `POST /api/start-capture` - Start USB data capture (launches Python script)
- `POST /api/stop-capture` - Stop USB data capture (gracefully terminates)
- `GET /api/capture-status` - Get current capture status

## Troubleshooting

### Python Dependencies Not Found
- Ensure you activated the virtual environment: `source rowing_env/bin/activate`
- Install dependencies: `pip install bleak git+https://github.com/droogmic/Py3Row.git`
- Check that the virtual environment Python is being used

### No PM5 Found
- **USB Connection**: Ensure PM5 is connected via USB cable (square USB-B port on PM5)
- **Power**: Make sure PM5 is powered on and awake
- **USB Permissions**: On macOS, USB access requires sudo permissions
- **Test Connection**: Try running: `sudo ./rowing_env/bin/python3 enhanced_usb_c2.py`
- **Alternative**: Use the helper script: `./start_usb_capture.sh`

### USB Permission Issues (macOS)
- **Sudo Required**: macOS requires administrator privileges for USB device access
- **Password Prompt**: You may be prompted for your sudo password
- **Helper Script**: Use `./start_usb_capture.sh` for easier sudo access
- **Manual Method**: `sudo -E ./rowing_env/bin/python3 enhanced_usb_c2.py`

### Slow Power Updates
Use the `enhanced_ble_c2.py` script which captures power curve data from characteristics 0x0035/0x0036.

### Connection Issues
- Try running the BLE script with `sudo` (though not recommended)
- Ensure BLE permissions are granted on macOS
- Check that the PM5 is in range and powered on

## File Structure

```
/Users/kelsorj/My Drive/Code/rowingIA/
├── ble_c2.py              # Original BLE script
├── highres_ble_c2.py      # High-res version
├── enhanced_ble_c2.py     # Enhanced with power curves
├── server.js              # Express server for data
├── package.json           # Node.js dependencies
├── public/                # Static React files
├── src/                   # React source code
│   ├── App.js            # Main dashboard component
│   ├── App.css           # Dashboard styling
│   └── index.js          # React entry point
└── *.csv                 # Generated data files
```

## Dependencies

- **Python**: bleak, asyncio, csv, datetime
- **Node.js**: express, cors, papaparse, react, recharts

## License

MIT License - feel free to use and modify for your rowing analysis needs.
