Before running the source code, you must set up the needed directory structure as well as creating a Python virtual environment. Here, we provide the Slurm script that we used to create the virtual environment. Follow the following steps to set up the directory structure as well as the virtual environment:

1. The "bird" directory contains the maven and scala files required to compile the source code. Move this directory to your home directory (or anywhere that you would like). The *virenv_setup* script assumes that the "bird" directory is located at your home directory. If you want to put it somewhere else, you need to modify the *virenv_setup* script accordingly. 

2. Place the provided "code" directory (located in the parent directory of this directory) in *bird/src/main/scala/*.

3. On line 12 of *virenv_setup.slurm*, replace "your_pi_group" with your PI group. 

4. On line 16 of *virenv_setup.slurm*, replace "your_email" with your email address to receive notifications about the start and the completion of the execution of the script. If you don't want to receive notifications, you can comment out this line. 

5. Now run the *virenv_setup.slurm* script to create the virtual environment directory required to execute BIRD. The directory will be called "bird_virenv" and it will be located inside the "bird" directory. On the University of Arizona Puma HPC cluster, you can run the script using the following command: *sbatch virenv_setup.slurm*

Note: If you don't want to create the Python virtual environment by yourself, you can download it (1.2GB) from [here](https://drive.google.com/file/d/1eWLFBIdmRA8p2ugydsefLzUqidN0tjKC/view?usp=sharing). It contains the prebuilt "bird_virenv" directory. Put the "bird" directory in your home directory and the "bird_virenv" directory inside "bird".
