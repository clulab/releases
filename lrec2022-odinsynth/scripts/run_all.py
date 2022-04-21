import glob
import os
import signal
import threading
import subprocess
import datetime
from collections import defaultdict
from multiprocessing import Pool
from argparse import ArgumentParser

def process_data(data):
    (commands, sleep) = data
    print(f"sleep {sleep} {threading.get_ident()}")
    # Sleep to avoid lock of sbt. Necessary to sleep only in the beginning
    os.system(f"sleep {sleep}")
    for c in commands:
        os.system(c)
        # print(f'{threading.get_ident()} - {c}')
    pass

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Entry point of the application. This script attempts to create rules for all the clusters")

    parser.add_argument('-ss', '--shard-size', type=int, required=False, default=20, 
        help="How many clusters to attempt per run. If it is too small, the sbt will be invoked repeteadly (slow). If it is too big, some clusters may fail (not implemented error).")
    parser.add_argument('-cp', '--cluster-path', type=str, required=False, default="/data/nlp/corpora/odinsynth/data/TACRED/odinsynth_tacred101/", 
        help="Where to find the clusters that we are attempting to solve")
    parser.add_argument('-sp', '--save-path', type=str, required=True, 
        help="Path to a folder where we will save the results. We will create a folder for each shard.")
    parser.add_argument('-p', '--ports', nargs='+', type=int, required=True,
        help="What ports to use")
    parser.add_argument('-sc', '--skip-cluster', type=int, required=False, default=0,
        help="How many clusters to skip. Useful when we do not want to generate for all the clusters immediately.")
    parser.add_argument('-noc', '--number-of-clusters', type=int, required=False, default=None,
        help="Total number of clusters to generate for. When None, use all the clusters")
    parser.add_argument('-ap', '--aggregation-path', type=str, required=False, 
        help="If present, we aggregate each shard in that folder at the end")
    parser.add_argument('-wr', '--with-reward', action='store_true', help="Add reward to the score or not")
    parser.add_argument('--start-servers', action='store_true', help="Whether to start the servers or not")
    parser.add_argument('-wc', '--which-checkpoint', type=str, required=False, default=None, 
        help="Which checkpoint to use. Can be from {'2_128', '2_256', '4_256', '4_512', '8_512'} or a full path (e.g. '/home/rvacareanu/projects/odinsynth_models/8_512/best.ckpt')")

    args = parser.parse_args()
    result = vars(args)
    print(result)

    save_path        = result['save_path']
    shard_size       = result['shard_size']
    cluster_path     = result['cluster_path']
    aggregation_path = result['aggregation_path']
    with_reward      = "true" if(result['with_reward']) else "false" # str of boolean is "True" or "False", but we have to pass "true" or "false" (lowercased). Explicitly store a string

    process = None

    path = cluster_path + '/*/*'
    if result['number_of_clusters'] is None:
        number_of_clusters = len([x for x in glob.glob(path) if os.path.isfile(x)])
    else:
        number_of_clusters = result['number_of_clusters'] + result['skip_cluster']
    print(number_of_clusters)


    # Attempt to create directories for each
    for i in range(0, int(number_of_clusters/result['shard_size']) + 1):
        if not os.path.exists(f'{save_path}/s{i}'):
            os.mkdir(f'{save_path}/s{i}')

    # Start the server if specified
    if result.get('start_servers', False):
        if 'which_checkpoint' not in result:
            print("The checkpoint should be specified if the servers are to be started")
            exit()
        checkpoint_path = result['which_checkpoint']
        ports_string = ' '.join([str(p) for p in result['ports']])
        process = subprocess.Popen(f"CHECKPOINT_PATH='{checkpoint_path}' /home/rvacareanu/projects/odinsynth/python/bash_scripts/start_servers.sh {ports_string}", shell=True, start_new_session=True)
        # Sleep 10s
        print("Started servers. Sleep 10s")
        os.system("sleep 10")
    
    # Create a list of commands for each port, then run them in parallel
    commands = defaultdict(list)
    total_iterations = int(number_of_clusters / (result['shard_size'] * len(result['ports']))) + 1
    shards_to_skip = result['skip_cluster']
    shard = int(result['skip_cluster']/shard_size)
    for i in range(0, total_iterations):
        for port in result['ports']:
            if shards_to_skip < number_of_clusters:
                command = f"sbt -J-Xmx32g 'runMain org.clulab.odinsynth.evaluation.tacred.TacredRuleGeneration dynamic_rule_generation.conf' -Dodinsynth.evaluation.tacred.clusterPath='{cluster_path}' -Dodinsynth.evaluation.tacred.skipClusters={shards_to_skip} -Dodinsynth.evaluation.tacred.takeClusters={shard_size} -Dodinsynth.evaluation.tacred.ruleBasepath={save_path}/s{shard} -Dodinsynth.evaluation.tacred.endpoint='http://localhost:{port}' -Dodinsynth.useReward={with_reward}"
                commands[port].append(command)
                shards_to_skip += shard_size
                shard += 1
    start_time = datetime.datetime.now()
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    pool = Pool(len(result['ports']))
    commands = list(commands.values())
    sleeps = []
    for port in range(0, len(result["ports"])):
        sleeps.append(15 * port)
    pool.map(process_data, list(zip(commands, sleeps)))

    # Copy all results in an aggregation folder
    if aggregation_path is not None:
        import glob
        from shutil import copyfile

        shards = glob.glob(f"{save_path}/*/*")
        versions = []
        for f in [s for s in shards if "version.txt" in s]:
            with open(f) as fin:
                versions.append(fin.readlines()[0].strip())

        # We should use the same scorer version
        assert(len(set(versions)) == 1)
        shards = [s for s in list(set(shards).difference(versions)) if "all_solutions" not in s]
        output_names = [aggregation_path + '/' + s.split('/')[-1] for s in shards]

        for original_file, moved_file in zip(shards, output_names):
            copyfile(original_file, moved_file)
        

        # Run the system once again to generate the all_solutions.tsv and all_solutions_all_trials.tsv
        port = result['ports'][0]
        command = f"sbt -J-Xmx32g 'runMain org.clulab.odinsynth.evaluation.tacred.TacredRuleGeneration dynamic_rule_generation.conf' -Dodinsynth.evaluation.tacred.skipClusters=0 -Dodinsynth.evaluation.tacred.takeClusters={len(shards) + 1000} -Dodinsynth.evaluation.tacred.ruleBasepath={aggregation_path} -Dodinsynth.evaluation.tacred.endpoint='http://localhost:{port}'"
        os.system(command)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Stop the server if it was started
    if result.get('start_servers', False):
        # The process object is set if start_servers is True
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    