module.exports = {
  apps: [
    {
      name: "media_control_plane",
      script: "server.py",
      cwd: __dirname,
      interpreter: `${__dirname}/.venv/bin/python`,
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: "development",
      },
      env_production: {
        NODE_ENV: "production",
      }
    }
  ]
};
