const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const Papa = require('papaparse');
const { spawn } = require('child_process');

const app = express();
const PORT = 3001;

let bleProcess = null;
let isCapturing = false;
let lastDataTime = null;

app.use(cors());
app.use(express.json());

// Serve static files from the build directory
app.use(express.static(path.join(__dirname, 'build')));

// API endpoint to get the latest CSV data
app.get('/api/latest-data', (req, res) => {
  try {
    // Find the most recent CSV file
    const files = fs.readdirSync(__dirname)
      .filter(file => (file.startsWith('pm5_enhanced_parsed_') || file.startsWith('pm5_py3row_parsed_')) && file.endsWith('.csv'))
      .sort()
      .reverse();
    
    if (files.length === 0) {
      return res.json({
        data: [],
        message: 'No data files found',
        isConnected: false,
        lastDataTime: lastDataTime
      });
    }

    const latestFile = files[0];
    const filePath = path.join(__dirname, latestFile);
    const csvContent = fs.readFileSync(filePath, 'utf8');

    Papa.parse(csvContent, {
      header: true,
      complete: (results) => {
        // Update last data time if we have data
        const recentData = results.data.slice(-10); // Check last 10 rows
        for (let i = recentData.length - 1; i >= 0; i--) {
          if (recentData[i].timestamp_iso) {
            lastDataTime = new Date(recentData[i].timestamp_iso);
            break;
          }
        }

        // Check if data is recent (within last 10 seconds)
        const now = new Date();
        const isConnected = lastDataTime && (now - lastDataTime) < 10000; // 10 seconds

        res.json({
          data: results.data.slice(-50), // Return last 50 rows
          filename: latestFile,
          count: results.data.length,
          isConnected: isConnected,
          lastDataTime: lastDataTime
        });
      },
      error: (error) => {
        res.status(500).json({ error: error.message });
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to get current time
app.get('/api/time', (req, res) => {
  res.json({
    timestamp: new Date().toISOString(),
    unix: Date.now()
  });
});

// API endpoint to start BLE capture
app.post('/api/start-capture', (req, res) => {
  if (isCapturing) {
    return res.json({ success: false, message: 'Already capturing' });
  }

  try {
    console.log('Starting USB capture...');
    console.log('Note: Run this command in a separate terminal:');
    console.log('sudo ./rowing_env/bin/python3 enhanced_usb_c2.py');
    console.log('Then refresh the dashboard to see live data.');

    // Don't actually start the process - just provide instructions
    // The user runs the USB script manually with sudo
    return res.json({
      success: false,
      message: 'Please run the USB capture script manually with sudo in a separate terminal, then refresh the dashboard.'
    });

    isCapturing = true;

    // Handle process output
    bleProcess.stdout.on('data', (data) => {
      console.log('BLE stdout:', data.toString());
    });

    bleProcess.stderr.on('data', (data) => {
      console.log('BLE stderr:', data.toString());
    });

    bleProcess.on('close', (code) => {
      console.log(`BLE process exited with code ${code}`);
      isCapturing = false;
      bleProcess = null;
    });

    res.json({ success: true, message: 'USB capture started' });
  } catch (error) {
    console.error('Error starting USB capture:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

// API endpoint to stop USB capture
app.post('/api/stop-capture', (req, res) => {
  if (!isCapturing || !bleProcess) {
    return res.json({ success: false, message: 'Not currently capturing' });
  }

  try {
    console.log('Stopping USB capture...');
    bleProcess.kill('SIGINT'); // Send Ctrl+C
    isCapturing = false;
    res.json({ success: true, message: 'USB capture stopped' });
  } catch (error) {
    console.error('Error stopping USB capture:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

// API endpoint to get capture status
app.get('/api/capture-status', (req, res) => {
  res.json({
    isCapturing,
    processRunning: bleProcess !== null
  });
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('API endpoints:');
  console.log(`  GET /api/latest-data - Get latest rowing data`);
  console.log(`  GET /api/time - Get current server time`);
  console.log(`  POST /api/start-capture - Start USB data capture`);
  console.log(`  POST /api/stop-capture - Stop USB data capture`);
  console.log(`  GET /api/capture-status - Get capture status`);
});

// Cleanup on exit
process.on('SIGINT', () => {
  console.log('Server shutting down...');
  if (bleProcess) {
    console.log('Killing BLE process...');
    bleProcess.kill('SIGINT');
  }
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('Server shutting down...');
  if (bleProcess) {
    console.log('Killing BLE process...');
    bleProcess.kill('SIGINT');
  }
  process.exit(0);
});
