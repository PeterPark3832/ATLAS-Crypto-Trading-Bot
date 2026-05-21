#!/bin/bash
cd /root/atlas_spot
nohup uvicorn atlas_spot_web_dashboard:app --host 0.0.0.0 --port 8080 > logs/dashboard.log 2>&1 &
echo "ATLAS Spot Dashboard started on :8080"
