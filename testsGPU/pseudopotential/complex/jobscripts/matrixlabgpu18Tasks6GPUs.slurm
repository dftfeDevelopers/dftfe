#!/bin/sh
#SBATCH --job-name=MDTest # Job name
#SBATCH --ntasks-per-node=18 # Number of tasks per node
#SBATCH --nodes=1
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH -o gpurun_nmd3.out
#SBATCH --gres=gpu:6
#SBATCH --partition=debug
#SBATCH --dependency=singleton
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"​
module load openmpigcc9.2slurm20
module load gcc9.2
module load cuda11gcc9.2
module load mkl/2021.2.0 ​

​

srun -n 18 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/complex/dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun -n 18 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/complex/dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
srun -n 18 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/complex/dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun -n 18 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/complex/dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun -n 18 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/complex/dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun -n 18 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/complex/dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun -n 18 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/complex/dftfe parameterFileMg2x_6.prm > outputMg2x_6
