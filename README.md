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
pip install bleak
```

### 2. Start the Data Server

First, run the enhanced BLE data collection script to capture rowing data:

```bash
python3 enhanced_ble_c2.py
```

This will:
- Connect to your Concept2 PM5 via Bluetooth
- Capture detailed rowing data including power curves
- Generate CSV files: `pm5_enhanced_parsed_*.csv`, `pm5_enhanced_raw_*.csv`, `pm5_power_curve_*.csv`

### 3. Start the React Dashboard

In a new terminal window:

```bash
npm run server
```

This starts the Express server on port 3001, serving both the API and React app.

### 4. Open Dashboard

Navigate to `http://localhost:3001` in your browser.

## Usage

### Quick Start
1. **Open Dashboard**: Navigate to `http://localhost:3001` in your browser
2. **Click "Start Capture"**: This launches the BLE data collection automatically
3. **Start rowing** on your Concept2 PM5 with "Just Row" mode
4. **Record video** while rowing - use the computer clock in the dashboard to sync timing
5. **Monitor real-time data** as you row
6. **Click "Stop Capture"** when finished

### Manual Usage (if needed)
If you prefer running scripts manually:
```bash
python3 enhanced_ble_c2.py
```

## Integrated BLE Control

The dashboard now includes direct control over the rowing data capture:

- **Start Capture Button**: Launches the Python BLE capture script
- **Stop Capture Button**: Gracefully stops data collection
- **Status Indicators**:
  - 🔴 **Recording**: Actively capturing data
  - ⏸️ **Ready**: Ready to start capture
  - ⏳ **Starting/Stopping**: Transition states
  - ❌ **Error**: Connection or capture issues

## Power Data Explanation

### Current Limitations
The original BLE script only captured **average power** which updates slowly. Power values appeared static because they represented workout averages, not instantaneous power.

### Enhanced Solution
The integrated system captures:
- **Basic Metrics**: Time, distance, stroke rate, pace, heart rate ✅
- **Average Power**: Traditional workout average power ✅
- **Instantaneous Power**: Real-time power curves (PM5 firmware dependent) ⚠️
- **Peak Power**: Maximum power per stroke (PM5 firmware dependent) ⚠️

### Data Sources
- **Characteristics 0x0031-0x0033**: Basic rowing metrics (speed, distance, heart rate, etc.) ✅
- **Characteristics 0x0035-0x0036**: Detailed power curve data (not supported by all PM5 models) ⚠️

### ⚠️ **Power Data Limitations**

**Important**: Not all Concept2 PM5 models/firmware versions support power curve data via BLE. If you see "N/A" for instantaneous and peak power, your PM5 doesn't provide this data through Bluetooth.

**Symptoms of missing power data:**
- Instantaneous Power: N/A
- Peak Power: N/A
- Average Power: May be 0 or low values
- No data from characteristics 0035/0036

### 🛠️ **Solutions for Power Curves**

1. **Use Concept2 ErgData App** (Recommended)
   - Connect PM5 to ErgData app on phone/tablet
   - ErgData can access power curve data not available via BLE
   - Export CSV files with complete power data

2. **Check PM5 Firmware**
   - Ensure your PM5 has the latest firmware
   - Older firmware versions may not support power curves

3. **Use Different PM5 Model**
   - Some PM5 models have better BLE power data support
   - Check Concept2 website for model comparisons

4. **Calculate Estimated Power**
   - Use speed, stroke rate, and drag factor to estimate power
   - Less accurate than direct measurements

## API Endpoints

- `GET /api/latest-data` - Get latest rowing data from CSV files
- `GET /api/time` - Get current server time
- `POST /api/start-capture` - Start BLE data capture (launches Python script)
- `POST /api/stop-capture` - Stop BLE data capture (gracefully terminates)
- `GET /api/capture-status` - Get current capture status

## Troubleshooting

### Python Dependencies Not Found
- Ensure you activated the virtual environment: `source rowing_env/bin/activate`
- Install bleak: `pip install bleak`
- Check that the virtual environment Python is being used

### No PM5 Found
- Ensure your PM5 has Bluetooth enabled
- Open the PM5's Bluetooth/Connect menu
- Make sure no other device is connected to the PM5
- Try running the BLE script manually: `./rowing_env/bin/python3 enhanced_ble_c2.py`

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
