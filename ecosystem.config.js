module.exports = {
  apps: [
    {
      name: 'rowing-dashboard',
      script: 'server.js',
      cwd: '/Users/kelsorj/My Drive/Code/rowingIA',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'development',
        PORT: 3001
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 3001
      },
      log_file: './logs/rowing-dashboard.log',
      out_file: './logs/rowing-dashboard-out.log',
      error_file: './logs/rowing-dashboard-error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true
    }
  ]
};
