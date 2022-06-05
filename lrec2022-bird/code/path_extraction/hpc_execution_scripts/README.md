In each step: 

- In each slurm file, replace "your_pi_group" with your PI group. 

- In each slurm file, replace "your_email" with your email address to receive notifications about the start and the completion of the execution of the scripts. If you don't want to receive notifications, you can comment out that line. 

- If there are shell scripts called "run_path_extraction_stepXXX" in the corresponding subdirectory, give execute persmissions to them (using chmod command) and then run the shell scripts. Do not manually run the slurm files in that subdirectory. The shell scripts run them.

- If there are no such shell scripts in the corresponding subdirectory, run the slurm files in that subdirectory manually. On the University of Arizona Puma cluster, you can run them using the following command: *sbatch slurm_file_name*
