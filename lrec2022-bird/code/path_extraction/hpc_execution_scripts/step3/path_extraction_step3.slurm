#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=pe_s3_partpartno
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
#SBATCH --mem=50gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=240:00:00
 
 
# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------

export _JAVA_OPTIONS="-Xmx40G"

### Load required modules/libraries if needed
module load maven
### change to your script’s directory
cd ~/bird
### Run your work
echo "Path extraction step 3"
mvn scala:run  -DmainClass=Path_Extraction_Step3  "-DaddArgs=partno"
echo "End of path extraction step 3 part partno."
