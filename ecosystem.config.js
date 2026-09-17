module.exports = {
  apps : [
    {
      name: "ai-mobile-api",
      script: "mobile_api.py",
      interpreter: "python3", // Uses the python environment on the host
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
      interpreter: "python3",
      autorestart: true,
      watch: false,
      max_memory_restart: '1G', // Prevent memory leaks
      error_file: "logs/worker-error.log",
      out_file: "logs/worker-out.log",
      time: true
    }
  ]
};
