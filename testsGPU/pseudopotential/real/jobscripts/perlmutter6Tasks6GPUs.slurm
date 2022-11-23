#!/bin/bash
#SBATCH -A m3979_g
#SBATCH -C gpu
#SBATCH -q early_science
#SBATCH --job-name realtests
#SBATCH -t 00:40:00
#SBATCH -n 6
#SBATCH --ntasks-per-node=3
#SBATCH -c 32
#SBATCH --gpus-per-node=3
#SBATCH --gpu-bind=map_gpu:0*1,1*1,2*1,3*0


export SLURM_CPU_BIND="cores"
#module load PrgEnv-gnu
#module load cpe-cuda
#module load cudatoolkit/21.9_11.4
#module unload cray-libsci/21.08.1.2
#module load cmake/3.22.0
#module load nccl/2.11.4
export OMP_NUM_THREADS=1

srun ./dftfe parameterFileN2_1.prm > outputN2_1
srun ./dftfe parameterFileN2_2.prm > outputN2_2
srun ./dftfe parameterFileN2_3.prm > outputN2_3
srun ./dftfe parameterFileN2_4.prm > outputN2_4
srun ./dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun ./dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun ./dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun ./dftfe parameterFileMg2x_11.prm > outputMg2x_11
srun ./dftfe parameterFileMg2x_14.prm > outputMg2x_14
