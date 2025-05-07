#!/bin/bash

source /data/aadhaarmask/masking/bin/activate

if ! pgrep -f "gunicorn.*pwsgi:app" > /dev/null
then
    echo "$(date): Gunicorn not running. Restarting..." >> /data/aadhaarmask/restart.log
    cd /data/aadhaarmask
    nohup gunicorn -w 1 -b 0.0.0.0:8080 pwsgi:app --timeout 300 > nohup.out 2>&1 &
else
    echo "$(date): Gunicorn is running."  >> /data/aadhaarmask/restart.log
fi
