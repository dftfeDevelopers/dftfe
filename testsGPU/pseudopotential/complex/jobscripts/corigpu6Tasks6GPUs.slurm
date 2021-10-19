#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=10
#SBATCH --constraint=gpu
#SBATCH -G 6
#SBATCH -A m1759
#SBATCH -q special
#SBATCH --exclusive
#SBATCH --mail-user=dsambit@umich.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -J testGPU1
#SBATCH -o testGPU1.o%j

# setup and run
echo "Job $SLURM_JOB_NAME, ID $SLURM_JOB_ID"
cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1

srun ./dftfe parameterFileMg2x_1.prm > outputMg2x_1

srun ./dftfe parameterFileMg2x_2.prm > outputMg2x_2

srun ./dftfe parameterFileMg2x_3.prm > outputMg2x_3

srun ./dftfe parameterFileMg2x_4.prm > outputMg2x_4

srun ./dftfe parameterFileMg2x_5.prm > outputMg2x_5
