#!/bin/bash
# This line tells the shell how to execute this script, and is unrelated
# to SLURM.
#sbatch run_task_HPC_background.sh (to run the script)
#scancel jobId to stop the job


# at the beginning of the script, lines beginning with "#SBATCH" are read by
# SLURM and used to set queueing options. You can comment out a SBATCH
# directive with a second leading #, eg:
##SBATCH --nodes=1

# we need 1 node, will launch a maximum of one task. The task uses 2 CPU cores
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# we expect the job to finish within 1 hours. If it takes longer than 1
# hours, SLURM can kill it:
#SBATCH --time=05:00:00

# we expect the job to use no more than 10GB of memory:
#SBATCH --mem=100GB

# we want the job to be named "myMatlabTest" rather than something generated
# from the script name. This will affect the name of the job as reported
# by squeue:
#SBATCH --job-name=ic_train

# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.

# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=slurm_run_%j.out

# once the first non-comment, non-SBATCH-directive line is encountered, SLURM
# stops looking for SBATCH directives. The remainder of the script is executed
# as a normal Unix shell script

# first we ensure a clean running environment:

python -u sample.py --encoder_path models-attention/encoder-10-3000.pkl --decoder_path models-attention/decoder-10-3000.pkl --image ../image_captioning/data/val2014
