# Rowing Data Dashboard

Real-time rowing data dashboard with force curve visualization for Concept2 PM5 rowing machines.

## Features

- 🚀 **Real-time Data Capture**: USB connection to Concept2 PM5 via Py3Row
- 📊 **Force Curve Visualization**: Complete stroke force profiles with aggregation
- ⏱️ **Video Synchronization**: Computer clock for video timing sync
- 📈 **Power Analytics**: Real-time power, pace, and stroke rate monitoring
- 🎯 **Stroke Analysis**: Complete drive and recovery phase visualization

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

### 3. Start Rowing Data Capture
```bash
sudo ./rowing_env/bin/python3 py3row_usb_capture.py
```

### 4. Open Dashboard
Navigate to `http://localhost:3001`

## PM2 Commands

```bash
# Start development server
npm run dev

# Start production server  
npm run prod

# Stop server
npm run stop

# Restart server
npm run restart

# View logs
npm run logs

# Check status
npm run status

# Delete from PM2
npm run delete
```

## Data Capture

The system captures comprehensive rowing data including:

- **Force Curves**: Complete stroke force profiles (drive + recovery)
- **Power Data**: Instantaneous and average power
- **Stroke Metrics**: Rate, distance, pace, calories
- **Timing**: Precise timestamps for video synchronization

## File Structure

```
rowingIA/
├── server.js                 # Express server
├── ecosystem.config.js       # PM2 configuration
├── py3row_usb_capture.py     # USB data capture script
├── src/                      # React frontend
│   ├── App.js               # Main dashboard component
│   └── App.css              # Dashboard styling
├── Py3Row/                  # Py3Row library for USB communication
├── logs/                    # PM2 log files
└── build/                   # React production build
```

## Requirements

- **Node.js** 16+ 
- **Python 3.13** with virtual environment
- **Py3Row** library for USB communication
- **Concept2 PM5** rowing machine
- **USB cable** (square USB-B port on PM5)

## Development

The dashboard automatically detects new CSV data files and displays:
- Real-time force curves
- Power over time graphs  
- Complete rowing metrics
- Video synchronization clock

## Troubleshooting

### USB Connection Issues
- Ensure PM5 is powered on and connected via USB
- Run capture script with `sudo` for USB permissions
- Check that no other apps are using the PM5

### Dashboard Not Updating
- Verify the server is running: `npm run status`
- Check logs: `npm run logs`
- Ensure CSV files are being generated in the project directory

### Force Curves Not Showing
- The system aggregates force data across the complete stroke
- Force curves only appear when a complete stroke is captured
- Check console output for "Complete stroke captured" messages

## License

MIT License - see LICENSE file for details.