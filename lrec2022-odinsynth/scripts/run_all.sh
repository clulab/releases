#!/bin/bash

save_path=${@:$#} # last parameter 
ports=${*%${!#}} # all parameters except the last
shard_size=10

echo $save_path
echo $ports

# Read the number of clusters
number_of_clusters=$(find "/data/nlp/corpora/odinsynth/data/TACRED/odinsynth_tacred4/" -type f -print | wc -l)
echo $number_of_clusters

# Number of ports to use
length=$(($# - 1))
echo $length


# Create directories
for i in $(seq 0 $(($number_of_clusters / $shard_size)));
do
    if [ ! -d /home/mlzboy/b2c2/shared/db ]; then
        $(mkdir $save_path/s$i)
    fi
done

# How many iterations we have to make
total_iterations=$(($(($number_of_clusters / $shard_size)) / $length + 1))
echo $(($number_of_clusters / $shard_size))
echo $total_iterations
echo $ports

# How many clusters were set to run
total=0
shard=0
initial_sleep=0
for i in $(seq 0 $total_iterations);
do
    echo "Doing $i"
    declare -a PID_LIST=()
    for j in $ports; do {
        # Don't start any new command if we are already done
        if [ "$total" -lt "$number_of_clusters" ]; then
            echo "Process \"$PORT\" started";
            $(sbt -J-Xmx32g 'runMain org.clulab.odinsynth.evaluation.tacred.TacredRuleGeneration dynamic_rule_generation.conf' -Dodinsynth.evaluation.tacred.skipClusters=$total -Dodinsynth.evaluation.tacred.takeClusters=$shard_size -Dodinsynth.evaluation.tacred.ruleBasepath=$save_path/s$shard -Dodinsynth.evaluation.tacred.endpoint="http://localhost:$j") &> /dev/null & pid=$!
            if [ "$initial_sleep" -eq 0]; then
                sleep $((5 + RANDOM % 11))
            fi
            # $(sleep 2) & pid=$!
            PID_LIST+=" $pid";
            # Increment the total with the shard_size to use as offset for the next
            total=$(($total + shard_size))
            shard=$(($shard + 1))
            echo $j
        fi
    } done
    # Chill all child processes on SIGINT
    trap "kill $PID_LIST" SIGINT
    echo "Parallel processes have started $PID_LIST";
    wait $PID_LIST
    echo
    echo "All processes have completed";
done
echo

# Running example
# ./run_all.sh 8000 8001 8002 8003 8004 8005 8006 8007 "save_path/"
# ./run_all.sh 8000 8001 8002 8003 8004 8005 8006 8007 "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/dynamic_new_model_smalltrained/"
# python run_all.py -p 8000 8002 8003 8004 8005 8006 8007 8008  -sp "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models_with_reward/2_256/shards/" -ap '/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models_with_reward/2_256/aggregated/' -wr