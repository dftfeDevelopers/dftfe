#!/bin/bash
#SBATCH -A MAT187_crusher
#SBATCH -J complextest
#SBATCH -t 00:50:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --gpus-per-node=6
#SBATCH --gpu-bind=closest

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
srun -n 6 -c 8 ./dftfe parameterFileMg2x_1.prm > outputMg2x_1

srun -n 6 -c 8 ./dftfe parameterFileMg2x_2.prm > outputMg2x_2

srun -n 6 -c 8 ./dftfe parameterFileMg2x_3.prm > outputMg2x_3

srun -n 6 -c 8 ./dftfe parameterFileMg2x_4.prm > outputMg2x_4

srun -n 6 -c 8 ./dftfe parameterFileMg2x_5.prm > outputMg2x_5

srun -n 6 -c 8 ./dftfe parameterFileMg2x_6.prm > outputMg2x_6

srun -n 6 -c 8 ./dftfe parameterFileBe.prm > outputBe

