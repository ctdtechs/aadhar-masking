#!/bin/bash
source /data/aadhaarmask/masking/bin/activate
cd /data/aadhaarmask
exec gunicorn wsgi:app \
  --bind 0.0.0.0:8000 \
  --workers 2 \
  --timeout 300 \
  --access-logfile /data/aadhaarmask/logs/gunicorn-access.log \
  --error-logfile /data/aadhaarmask/logs/gunicorn-error.log
