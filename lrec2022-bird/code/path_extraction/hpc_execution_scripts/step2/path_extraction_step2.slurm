#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=pe_s2
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
#SBATCH --ntasks=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=200gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=100:00:00
 
 
# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------

export _JAVA_OPTIONS="-Xmx100G"

### Load required modules/libraries if needed
module load maven
module load python/3.8/3.8.2
### change to your code directory
cd ~/bird
### Run your work
echo "Path Extraction step 2"
### Provide the path for "bird_virenv"
mvn scala:run  -DmainClass=Path_Extraction_Step2 "-DaddArgs=XXX/bird_virenv/bin/activate|bert-base-cased"
echo "End of path extraction step 2."
