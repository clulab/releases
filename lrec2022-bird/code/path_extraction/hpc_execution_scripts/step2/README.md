On line 42 of the slurm file, you need to provide the absolute path of the virtual environment directory "bird_virenv". In other words, you need to replace XXX in the following address with the correct absolute path on your system. By default, "bird_virenv" is created inside the "bird" directory, and "bird" is located in your home directory. 

XXX/bird_virenv/bin/activate

As an example, this could be the replaced address on a certain computer system:

/home/u13/john/bird/bird_virenv/bin/activate
