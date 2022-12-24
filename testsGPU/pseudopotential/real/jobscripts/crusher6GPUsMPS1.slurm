#!/bin/bash
#SBATCH -A MAT187_crusher
#SBATCH -J realmps1
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
srun -n 6 -c 8 ./dftfe parameterFileN2_1.prm > outputN2_1
srun -n 6 -c 8 ./dftfe parameterFileN2_2.prm > outputN2_2
srun -n 6 -c 8 ./dftfe parameterFileN2_3.prm > outputN2_3
srun -n 6 -c 8 ./dftfe parameterFileN2_4.prm > outputN2_4
srun -n 6 -c 8 ./dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun -n 6 -c 8 ./dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun -n 6 -c 8 ./dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun -n 6 -c 8 ./dftfe parameterFileMg2x_11.prm > outputMg2x_11
srun -n 6 -c 8 ./dftfe parameterFileMg2x_14.prm > outputMg2x_14
