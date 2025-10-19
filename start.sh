#!/usr/bin/env bash
# start.sh
export PORT=${PORT:-3000}
# use 0.0.0.0 so Replit exposes it
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
