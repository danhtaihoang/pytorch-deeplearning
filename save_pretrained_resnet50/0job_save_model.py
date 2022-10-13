##!/usr/bin/env python
#import numpy as np
#import pandas as pd
import os,sys,time

##===========================================================
## write jobs:
file_name = "save_model.py"
#step = int(input("select step (1 to write, 2 to submit): "))

a = ["#!/bin/bash\n"
     "#PBS -l ncpus=4\n",
     "#PBS -l ngpus=0\n",
     "#PBS -l mem=2GB\n",
     "#PBS -l jobfs=0GB\n",
     "#PBS -l walltime=1:00:00\n",
     "#PBS -q normal\n",
     #"#PBS -q gpuvolta\n",
    # "#PBS -P me11\n",
     "#PBS -P xd23\n",
    # "#PBS -q biodev\n"
    # "#PBS -P dl76\n"
     "#PBS -l storage=scratch/dl76\n",
     "#PBS -l wd\n",
     "module load python3/3.9.2\n"]

##-----------------
f = open("job.sh" ,"w")
f.writelines(a)
f.write("python3 %s >$PBS_JOBID.log" %(file_name))
f.close()

command_line = "qsub job.sh"
print(command_line)
os.system(command_line)