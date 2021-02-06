#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=00:10:00
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

srun ./dftfe parameterFileN2_1.prm > outputN2_1
srun ./dftfe parameterFileN2_2.prm > outputN2_2
srun ./dftfe parameterFileN2_3.prm > outputN2_3
srun ./dftfe parameterFileN2_4.prm > outputN2_4
srun ./dftfe parameterFileN2CPU.prm > outputN2_CPU
srun ./dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun ./dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun ./dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun ./dftfe parameterFileMg2x_11.prm > outputMg2x_11
