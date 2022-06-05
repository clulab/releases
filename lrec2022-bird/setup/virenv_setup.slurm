#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=virenv_setup
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
#SBATCH --mem=10gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=01:00:00

 
# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------

### If there is a previously created virtual environment folder, delete it:
rm -f -r ~/bird/bird_virenv

### Create the virtual environment:
module load python/3.8/3.8.2
module load cuda11/11.0
module load cuda11-dnn/8.0.2
module load cuda11-sdk/20.7
python3 -m venv ~/bird/bird_virenv

### Activate the virtual environment:
source ~/bird/bird_virenv/bin/activate

### Install the required packages:
echo "Upgrading pip..."
pip install --upgrade pip

echo
echo "Installing numpy..."
pip install numpy==1.18.5

echo
echo "Installing scikit-learn..."
pip install scikit-learn==0.23.2

echo
echo "Installing pytorch..."
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

echo
echo "Installing huggingface transformers..."
pip install transformers==3.5.1

echo
echo "List of all of the currently installed packages:"
pip list

### Deactivate the virtual environment:
deactivate

