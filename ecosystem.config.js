module.exports = {
  apps : [
    {
      name: "ai-mobile-api",
      script: "mobile_api.py",
      interpreter: "/home/popos/projek-anakmagang/ai-recognition/.venv/bin/python", // Uses the python environment on the host
      autorestart: true,
      watch: false,
      env: {
        FLASK_ENV: "production"
      },
      error_file: "logs/api-error.log",
      out_file: "logs/api-out.log",
      time: true
    },
    {
      name: "ai-recognition-worker",
      script: "recognition.py",
      interpreter: "/home/popos/projek-anakmagang/ai-recognition/.venv/bin/python",
      kill_timeout: 3000,
      wait_ready: false,
      autorestart: true,
      watch: false,
      min_uptime: '10s',
      max_restarts: 5,
      restart_delay: 2000,
      // Model loading and enrollment synchronization temporarily exceed 1 GB.
      // Keep a ceiling for genuine leaks without restarting a healthy GPU worker.
      max_memory_restart: '2G',
      error_file: "logs/worker-error.log",
      out_file: "logs/worker-out.log",
      time: true
    }
  ]
};
