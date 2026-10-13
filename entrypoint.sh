#!/bin/bash

set -e
python3 flasktts/run.py &
python3 -m flasktts.worker &
# Exit if any child process dies so docker restart policy brings us back up
wait -n
exit $?
