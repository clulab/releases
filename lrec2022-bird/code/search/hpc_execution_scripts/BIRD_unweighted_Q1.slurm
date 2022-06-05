#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=bird_unweighted_Q1
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x.out
### REQUIRED. Specify the PI group for this job
#SBATCH --account=your_pi_group
### Optional. Request email when job begins and ends
#SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
#SBATCH --mail-user=your_email
### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --nodes=1
#SBATCH --ntasks=20
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=100gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1 

# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------

### Load required modules/libraries if needed
module load python/3.8/3.8.2
module load cuda11/11.0
module load cuda11-dnn/8.0.2
module load cuda11-sdk/20.7
source ~/bird/bird_virenv/bin/activate
### change to your code directory
cd ~/bird/src/main/scala/code/search
### Run your work
python3 -u BIRD.py "bert-base-cased" "<-nsubj<-author->nmod_of->" "unweighted"
deactivate
