#!/bin/bash

ports=$@
length=$(($# - 1))

python_dir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

declare -a PID_LIST=()

echo "Current process $$"


for p in $ports
do
    $($python_dir/bash_scripts/start_server.sh $p) &> /dev/null & pid=$!
    PID_LIST+=" $pid";
done

trap "kill $PID_LIST" SIGINT
echo "Parallel processes have started $PID_LIST";
wait $PID_LIST
echo
echo "All processes have completed";

# CHECKPOINT_PATH='8_512' ./bash_scripts/start_servers.sh 8000 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012