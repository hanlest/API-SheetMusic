/** PM2: pm2 start ecosystem.config.cjs  |  pm2 logs API-SheetMusic */
const fs = require('fs')
const path = require('path')

const root = __dirname
const muscriptorExe = path.join(root, '.venv', 'Scripts', 'muscriptor.exe')

let device = 'cuda'
const deviceFile = path.join(root, '.pm2-device')
if (fs.existsSync(deviceFile)) {
  device = fs.readFileSync(deviceFile, 'utf8').trim() || device
}

const host = process.env.MUSCRIPTOR_HOST || '127.0.0.1'
const port = process.env.MUSCRIPTOR_PORT || '8222'

module.exports = {
  apps: [
    {
      name: 'API-SheetMusic',
      cwd: root,
      script: muscriptorExe,
      args: `serve --host ${host} --port ${port} --device ${device}`,
      interpreter: 'none',
      exec_mode: 'fork',
      instances: 1,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '120s',
      max_memory_restart: '12G',
      windowsHide: true,
      time: true,
      env: {
        HF_TOKEN: process.env.HF_TOKEN || '',
        UV_LINK_MODE: 'copy',
      },
    },
  ],
}
