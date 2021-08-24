#!/bin/bash
#   last-update: 2019-03-21 16:00
#SBATCH -J 3dscene
#SBATCH -N 1 # Zahl der nodes, we have one only
#SBATCH --gres=gpu:v100:1     # number of GPU cards (max=8)
#SBATCH --mem-per-cpu=5500    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks (max 40)
#SBATCH --cpus-per-task 10     # max 10/GPU CPU-threads needed (physcores*2)
#SBATCH --time 167:59:00 # set 0h59min walltime 167:59:00 max
## outcommended (does not work at the moment, ToDo):

exec 2>&1      # send errors into stdout stream
#env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG
#
# we have 8 GPUs, ToDo: add cublas-test as an example
# replace next 6 lines by your commands:
#Commandline Arguments:
#If no arguments supplied, it will use the default parameters written in this script
#First argument: name of the python main file, located inside the programROOT
#Second argument: path to the programROOT, starting from /scratch/tmp/schatter/Code. No need to add this to the argument, just rest of the path after this
#can supply either no argument, only first or both arguments

#Default Parameters
programROOT=/nfs1/kprabhu/3dReconstruction1
pythonMain=executor.py

echo $programROOT
echo $pythonMain

#If parameters were supplied
if [ $# -gt 0 ]; then
    if [ $# == 1 ]; then
        pythonMain=$1
    fi
    if [ $# == 2 ]; then
        pythonMain=$1
        programROOT=/nfs1/kprabhu/3dReconstruction1/$2
    fi
fi

#Create full path of the program
pyFullPath=$programROOT/$pythonMain

# . /usr/local/bin/slurmProlog.sh  # output slurm settings, debugging

#Activate conda environment
source /nfs1/kprabhu/anaconda3/etc/profile.d/conda.sh
conda activate 3dscene

srun python $pyFullPath

# .  /usr/local/bin/slurmEpilog.sh   # cleanup
