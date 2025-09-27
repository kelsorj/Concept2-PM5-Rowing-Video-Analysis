import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import './App.css';

function App() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [rowingData, setRowingData] = useState([]);
  const [latestMetrics, setLatestMetrics] = useState({});
  const [isConnected, setIsConnected] = useState(false);

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
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        setIsConnected(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

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
        </div>

        <div className="charts-container">
          <div className="chart-card">
            <h3>Power Over Time</h3>
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
                  formatter={(value) => [`${value}W`, 'Power']}
                  labelFormatter={(value) => `Time: ${formatDuration(value)}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="avg_power_w" 
                  stroke="#00ff88" 
                  strokeWidth={2}
                  dot={false}
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
