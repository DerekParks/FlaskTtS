#!/bin/bash

set -e
python3 flasktts/run.py &
huey_consumer.py flasktts.tasks.tasks.huey -w 1 &
# Exit if any child process dies so docker restart policy brings us back up
wait -n
exit $?
