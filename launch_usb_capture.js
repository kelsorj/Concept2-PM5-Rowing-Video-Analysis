#!/usr/bin/env node
// Script to launch USB rowing data capture with proper sudo handling

const { spawn } = require('child_process');
const path = require('path');

console.log('Launching USB rowing data capture...');
console.log('This will request sudo permissions for USB device access.');
console.log('');

// Launch the shell script which handles sudo
const usbScript = spawn('./start_usb_capture.sh', [], {
  cwd: __dirname,
  stdio: 'inherit', // Pass through stdin/stdout/stderr so user can interact
  shell: true
});

usbScript.on('close', (code) => {
  console.log(`\nUSB capture process exited with code ${code}`);
  process.exit(code);
});

usbScript.on('error', (error) => {
  console.error('Failed to launch USB capture:', error);
  process.exit(1);
});
