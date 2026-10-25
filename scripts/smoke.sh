#!/bin/sh
# usage: ./scripts/smoke.sh https://your-service.onrender.com YOUR_KEY
curl -s "$1/health"; echo
curl -s -X POST "$1/predict" -H "X-API-Key: $2" -H "Content-Type: application/json" \
  -d '{"log_text":"travis_fold:start:script\n$ make\nError: boom\ntravis_fold:end:script\nThe command \"make\" exited with 1.","top_k":3}'; echo
