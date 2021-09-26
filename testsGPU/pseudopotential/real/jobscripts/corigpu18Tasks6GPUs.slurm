#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=3
#SBATCH --constraint=gpu
#SBATCH -G 6
#SBATCH -A m1759
#SBATCH -q special
#SBATCH --exclusive
#SBATCH --mail-user=dsambit@umich.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -J testGPU2
#SBATCH -o testGPU2.o%j

# setup and run
echo "Job $SLURM_JOB_NAME, ID $SLURM_JOB_ID"
cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1

srun ./dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun ./dftfe parameterFileMg2x_1_spincpu.prm > outputMg2x_1_spin_cpu
srun ./dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
srun ./dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun ./dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun ./dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun ./dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun ./dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun ./dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun ./dftfe parameterFileMg2x_12.prm > outputMg2x_12
srun ./dftfe parameterFileMg2x_13.prm > outputMg2x_13
