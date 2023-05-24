#!/bin/sh
#SBATCH --job-name=GPUcTestComplex          # Job name
#SBATCH --ntasks-per-node=18                # Number of tasks per node
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH -o gpu_ctest_complex.out
#SBATCH --partition=debug

echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load gcc9.2 cmake3.19 cuda11gcc9.2 mkl/2021.2.0 openmpicuda11

export UCX_LOG_LEVEL=ERROR

srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileBe.prm     > outputBe

