#!/bin/bash
#SBATCH -A m3979_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name complextests
#SBATCH -t 00:30:00
#SBATCH -n 6
#SBATCH --ntasks-per-node=3
#SBATCH -c 32
#SBATCH --gpus-per-node=3
#SBATCH --gpu-bind=map_gpu:0*1,1*1,2*1


export SLURM_CPU_BIND="cores"
#module load PrgEnv-gnu
#module load cpe-cuda
#module load cudatoolkit/21.9_11.4
#module unload cray-libsci/21.08.1.2
#module load cmake/3.22.0
#module load nccl/2.11.4
export OMP_NUM_THREADS=1

srun ./dftfe parameterFileMg2x_1.prm > outputMg2x_1

srun ./dftfe parameterFileMg2x_2.prm > outputMg2x_2

srun ./dftfe parameterFileMg2x_3.prm > outputMg2x_3

srun ./dftfe parameterFileMg2x_4.prm > outputMg2x_4

srun ./dftfe parameterFileMg2x_5.prm > outputMg2x_5

srun ./dftfe parameterFileMg2x_6.prm > outputMg2x_6
