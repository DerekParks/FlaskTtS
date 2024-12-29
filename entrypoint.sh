#!/bin/bash

set -e
python3 flasktts/run.py &
huey_consumer.py flasktts.tasks.tasks.huey -w 1 &
wait
