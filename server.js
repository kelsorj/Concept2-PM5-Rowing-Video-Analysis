const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const Papa = require('papaparse');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// Serve static files from the build directory
app.use(express.static(path.join(__dirname, 'build')));

// API endpoint to get the latest CSV data
app.get('/api/latest-data', (req, res) => {
  try {
    // Find the most recent CSV file
    const files = fs.readdirSync(__dirname)
      .filter(file => file.startsWith('pm5_ble_parsed_') && file.endsWith('.csv'))
      .sort()
      .reverse();
    
    if (files.length === 0) {
      return res.json({ data: [], message: 'No data files found' });
    }
    
    const latestFile = files[0];
    const filePath = path.join(__dirname, latestFile);
    const csvContent = fs.readFileSync(filePath, 'utf8');
    
    Papa.parse(csvContent, {
      header: true,
      complete: (results) => {
        res.json({ 
          data: results.data,
          filename: latestFile,
          count: results.data.length
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

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('API endpoints:');
  console.log(`  GET /api/latest-data - Get latest rowing data`);
  console.log(`  GET /api/time - Get current server time`);
});
