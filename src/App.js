import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import './App.css';

function App() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [rowingData, setRowingData] = useState([]);
  const [latestMetrics, setLatestMetrics] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureStatus, setCaptureStatus] = useState('idle');

  // Sample data for testing
  const sampleData = [
    { elapsed_s: 10.0, distance_m: 35.0, spm: 24, hr_bpm: 140, avg_power_w: 180, speed_m_s: 3.5, pace_cur_s_per_500m: 143.0 },
    { elapsed_s: 20.0, distance_m: 70.0, spm: 26, hr_bpm: 150, avg_power_w: 200, speed_m_s: 3.8, pace_cur_s_per_500m: 132.0 },
    { elapsed_s: 30.0, distance_m: 105.0, spm: 24, hr_bpm: 155, avg_power_w: 190, speed_m_s: 3.6, pace_cur_s_per_500m: 139.0 },
  ];

  // Update clock every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch rowing data every 2 seconds
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:3001/api/latest-data');
        const result = await response.json();

        if (result.data && result.data.length > 0) {
          // Filter out empty rows and convert numeric fields
          const validData = result.data
            .filter(row => row.elapsed_s && row.elapsed_s !== '')
            .map(row => ({
              ...row,
              elapsed_s: parseFloat(row.elapsed_s) || 0,
              distance_m: parseFloat(row.distance_m) || 0,
              spm: parseInt(row.spm) || 0,
              hr_bpm: parseInt(row.hr_bpm) || 0,
              avg_power_w: parseInt(row.avg_power_w) || 0,
              speed_m_s: parseFloat(row.speed_m_s) || 0,
              pace_cur_s_per_500m: parseFloat(row.pace_cur_s_per_500m) || 0
            }))
            .slice(-100); // Keep last 100 data points for performance

          setRowingData(validData);

          if (validData.length > 0) {
            setLatestMetrics(validData[validData.length - 1]);
            setIsConnected(true);
          }
        } else {
          // Use sample data if no real data
          setRowingData(sampleData);
          setLatestMetrics(sampleData[sampleData.length - 1]);
          setIsConnected(false);
        }
      } catch (error) {
        // Use sample data on connection error
        setRowingData(sampleData);
        setLatestMetrics(sampleData[sampleData.length - 1]);
        setIsConnected(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  // Check capture status every 3 seconds
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch('http://localhost:3001/api/capture-status');
        const result = await response.json();
        setIsCapturing(result.isCapturing);
        setCaptureStatus(result.isCapturing ? 'capturing' : 'idle');
      } catch (error) {
        console.error('Error checking capture status:', error);
        setCaptureStatus('error');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  // Function to start BLE capture
  const startCapture = async () => {
    try {
      setCaptureStatus('starting');
      const response = await fetch('http://localhost:3001/api/start-capture', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const result = await response.json();
      if (result.success) {
        setCaptureStatus('capturing');
        setIsCapturing(true);
      } else {
        setCaptureStatus('error');
        alert('Failed to start capture: ' + result.message);
      }
    } catch (error) {
      console.error('Error starting capture:', error);
      setCaptureStatus('error');
      alert('Error starting capture: ' + error.message);
    }
  };

  // Function to stop BLE capture
  const stopCapture = async () => {
    try {
      setCaptureStatus('stopping');
      const response = await fetch('http://localhost:3001/api/stop-capture', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const result = await response.json();
      if (result.success) {
        setCaptureStatus('idle');
        setIsCapturing(false);
      } else {
        setCaptureStatus('error');
        alert('Failed to stop capture: ' + result.message);
      }
    } catch (error) {
      console.error('Error stopping capture:', error);
      setCaptureStatus('error');
      alert('Error stopping capture: ' + error.message);
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
  };

  const formatPace = (pace) => {
    if (!pace || pace === 0) return '--:--';
    const mins = Math.floor(pace / 60);
    const secs = (pace % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Rowing Dashboard</h1>
        <div className="clock-container">
          <div className="clock">
            <div className="clock-label">Computer Time</div>
            <div className="clock-time">{formatTime(currentTime)}</div>
          </div>
          <div className="connection-status">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>

          <div className="capture-controls">
            <button
              className={`capture-button start ${captureStatus === 'capturing' ? 'disabled' : ''}`}
              onClick={startCapture}
              disabled={captureStatus === 'capturing' || captureStatus === 'starting'}
            >
              {captureStatus === 'starting' ? 'Starting...' : 'Start Capture'}
            </button>
            <button
              className={`capture-button stop ${captureStatus !== 'capturing' ? 'disabled' : ''}`}
              onClick={stopCapture}
              disabled={captureStatus !== 'capturing'}
            >
              {captureStatus === 'stopping' ? 'Stopping...' : 'Stop Capture'}
            </button>
            <div className="capture-status">
              <span className={`status-text ${captureStatus}`}>
                {captureStatus === 'capturing' && '🔴 Recording'}
                {captureStatus === 'idle' && '⏸️ Ready'}
                {captureStatus === 'starting' && '⏳ Starting...'}
                {captureStatus === 'stopping' && '⏳ Stopping...'}
                {captureStatus === 'error' && '❌ Error'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="dashboard">
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Elapsed Time</div>
            <div className="metric-value">
              {latestMetrics.elapsed_s ? formatDuration(latestMetrics.elapsed_s) : '--:--'}
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Distance</div>
            <div className="metric-value">
              {latestMetrics.distance_m ? `${latestMetrics.distance_m.toFixed(1)}m` : '--'}
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Stroke Rate</div>
            <div className="metric-value">
              {latestMetrics.spm ? `${latestMetrics.spm} spm` : '--'}
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Heart Rate</div>
            <div className="metric-value">
              {latestMetrics.hr_bpm ? `${latestMetrics.hr_bpm} bpm` : '--'}
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Current Pace</div>
            <div className="metric-value">
              {formatPace(latestMetrics.pace_cur_s_per_500m)}
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Avg Power</div>
            <div className="metric-value">
              {latestMetrics.avg_power_w ? `${latestMetrics.avg_power_w}W` : '--'}
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Instant Power</div>
            <div className="metric-value">
              {latestMetrics.instantaneous_power_w ? `${latestMetrics.instantaneous_power_w}W` : '--'}
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Peak Power</div>
            <div className="metric-value">
              {latestMetrics.peak_power_w ? `${latestMetrics.peak_power_w}W` : '--'}
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Stroke Count</div>
            <div className="metric-value">
              {latestMetrics.stroke_count || '--'}
            </div>
          </div>
        </div>

        <div className="charts-container">
          <div className="chart-card">
            <h3>Power Curves</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rowingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis
                  dataKey="elapsed_s"
                  stroke="#888"
                  tickFormatter={(value) => formatDuration(value)}
                />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2a2a2a',
                    border: '1px solid #444',
                    color: '#fff'
                  }}
                  formatter={(value, name) => {
                    if (name === 'instantaneous_power_w') return [`${value}W`, 'Instant Power'];
                    if (name === 'avg_power_w') return [`${value}W`, 'Average Power'];
                    if (name === 'peak_power_w') return [`${value}W`, 'Peak Power'];
                    return [value, name];
                  }}
                  labelFormatter={(value) => `Time: ${formatDuration(value)}`}
                />
                <Line
                  type="monotone"
                  dataKey="instantaneous_power_w"
                  stroke="#ff6b6b"
                  strokeWidth={1}
                  dot={false}
                  name="instantaneous_power_w"
                />
                <Line
                  type="monotone"
                  dataKey="avg_power_w"
                  stroke="#00ff88"
                  strokeWidth={2}
                  dot={false}
                  name="avg_power_w"
                />
                <Line
                  type="monotone"
                  dataKey="peak_power_w"
                  stroke="#4ecdc4"
                  strokeWidth={1}
                  dot={false}
                  name="peak_power_w"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Stroke Rate Over Time</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rowingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis 
                  dataKey="elapsed_s" 
                  stroke="#888"
                  tickFormatter={(value) => formatDuration(value)}
                />
                <YAxis stroke="#888" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#2a2a2a', 
                    border: '1px solid #444',
                    color: '#fff'
                  }}
                  formatter={(value) => [`${value} spm`, 'Stroke Rate']}
                  labelFormatter={(value) => `Time: ${formatDuration(value)}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="spm" 
                  stroke="#ff6b6b" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Distance Progress</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rowingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis 
                  dataKey="elapsed_s" 
                  stroke="#888"
                  tickFormatter={(value) => formatDuration(value)}
                />
                <YAxis stroke="#888" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#2a2a2a', 
                    border: '1px solid #444',
                    color: '#fff'
                  }}
                  formatter={(value) => [`${value.toFixed(1)}m`, 'Distance']}
                  labelFormatter={(value) => `Time: ${formatDuration(value)}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="distance_m" 
                  stroke="#4ecdc4" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <h3>Heart Rate Over Time</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rowingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis 
                  dataKey="elapsed_s" 
                  stroke="#888"
                  tickFormatter={(value) => formatDuration(value)}
                />
                <YAxis stroke="#888" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#2a2a2a', 
                    border: '1px solid #444',
                    color: '#fff'
                  }}
                  formatter={(value) => [`${value} bpm`, 'Heart Rate']}
                  labelFormatter={(value) => `Time: ${formatDuration(value)}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="hr_bpm" 
                  stroke="#ff9f43" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
