#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=zz_thinkinnaturallanguage

### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out

### REQUIRED. Specify the PI group for this job
#SBATCH --account=msurdeanu

### Optional. Request email when job begins and ends
### SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
### SBATCH --mail-user=zhengzhongliang@email.arizona.edu

### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard

### REQUIRED. Set the number of cores that will be used for this job.
### SBATCH --nodes=1
### SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=8gb

### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 4_T5Small_Chaining_FormalResults.py 11 1 2 3nn 70k 5 3 c f r
/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 4_T5Small_Chaining_FormalResults.py 11 1 3 3nn 70k 5 3 c f r






