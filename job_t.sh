#!/bin/bash
# Project to run under 
#SBATCH -A SNIC2021-22-752
# Name of the job (makes easier to find in the status lists) 
#SBATCH -J Threads
#SBATCH --constraint=skylake
# name of the output file
#SBATCH --output=runf.out.%j
# name of the error file
#SBATCH --error=runf.err.%j
#SBATCH --exclusive
# asking for 14 cores 
#SBATCH -c 14 
# the job can use up to 30 minutes to run
#SBATCH --time=00:30:00

# load any modules you need. In this example, GCC compilers with OpenMPI 
# this should be changed according to which modules/programs are needed
module purge
module load foss
module load CMake
bash test_n.sh

# run the program - start parallel programs with srun
#srun ./my_parallel_prograbatch
