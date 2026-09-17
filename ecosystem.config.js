module.exports = {
  apps: [
    {
      name: "ai-detection-api",
      script: "api_server.py",
      interpreter: "./.venv/bin/python",
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      env: {
        NODE_ENV: "production",
      },
      error_file: "logs/api-error.log",
      out_file: "logs/api-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss"
    },
    {
      name: "ai-detection-supervisor",
      script: "supervisor.py",
      interpreter: "./.venv/bin/python",
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
      },
      error_file: "logs/supervisor-error.log",
      out_file: "logs/supervisor-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss"
    }
  ]
};
