#!/bin/sh
#SBATCH --job-name=MDtest # Job name
#SBATCH --ntasks-per-node=32 # Number of tasks per node
#SBATCH --nodes=1
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --partition=debug

​
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
​
module load openmpigcc9.2slurm20
module load gcc9.2
module load cuda11gcc9.2
module load mkl/2021.2.0 

srun -n 32 --mpi=pmi2 /home/kartickr/dft_fe_development/MDRestartfix/build/release/real/dftfe parameterFile_MD.prm > ex3_MD.output


