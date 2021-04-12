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
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=8gb

### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1

### echo ". /home/u15/zhengzhongliang/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc

### conda activate t5

### which python

/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 2_TrainAndSave.py 3nn f 10k 5 3
/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 2_TrainAndSave.py 3nn r 10k 5 3
/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 2_TrainAndSave.py 3nn c 10k 5 3

/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 2_TrainAndSave.py 3nn f 30k 5 3
/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 2_TrainAndSave.py 3nn r 30k 5 3
/home/u15/zhengzhongliang/anaconda3/envs/t5/bin/python 2_TrainAndSave.py 3nn c 30k 5 3
