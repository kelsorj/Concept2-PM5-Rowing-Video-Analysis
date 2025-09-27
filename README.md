# Rowing Data Visualization Dashboard

A real-time rowing data visualization dashboard that displays Concept2 PM5 rowing machine data with synchronized computer clock timing for video analysis.

## Features

- **Real-time Computer Clock**: Shows current computer time for video synchronization
- **Live Rowing Metrics**: Displays elapsed time, distance, stroke rate, heart rate, pace, and power
- **Power Curve Visualization**: Shows instantaneous power, average power, and peak power over time
- **Multiple Charts**: Power curves, stroke rate, distance progress, and heart rate over time
- **Responsive Design**: Works on desktop and mobile devices

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
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

This starts both the Express server (port 3001) and React app (port 3000).

### 4. Open Dashboard

Navigate to `http://localhost:3000` in your browser.

## Usage

1. **Start rowing** on your Concept2 PM5 with "Just Row" mode
2. **Record video** while rowing - use the computer clock in the dashboard to sync timing
3. **Monitor real-time data** as you row
4. **Analyze power curves** to see your stroke power distribution

## Power Data Explanation

### Current Limitations
The original BLE script only captured **average power** which updates slowly. Power values appeared static because they represented workout averages, not instantaneous power.

### Enhanced Solution
The new `enhanced_ble_c2.py` script captures:
- **Instantaneous Power**: Real-time power during each stroke phase
- **Peak Power**: Maximum power achieved in each stroke
- **Average Power**: Traditional workout average
- **Stroke-by-stroke Data**: Detailed metrics for each rowing stroke

### Data Sources
- **Characteristics 0x0031-0x0033**: Basic rowing metrics (speed, distance, heart rate, etc.)
- **Characteristics 0x0035-0x0036**: Detailed power curve and stroke data

## Troubleshooting

### No PM5 Found
- Ensure your PM5 has Bluetooth enabled
- Open the PM5's Bluetooth/Connect menu
- Make sure no other device is connected to the PM5

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
