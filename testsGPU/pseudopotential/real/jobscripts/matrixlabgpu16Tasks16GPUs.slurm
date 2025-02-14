#!/bin/bash
#SBATCH --job-name=GPUcTestReal             # Job name
#SBATCH --ntasks-per-node=8                # Number of tasks per node
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH -o gpu_ctest_real.out
#SBATCH --partition=debug

echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"


module load cuda/12.6 openmpi/gcc13cuda126 modulefiles/compiler-rt/2025.0.4 modulefiles/tbb/2022.0 modulefiles/mkl/2025.0


###needs to change this DFTFE_PATH
export DFTFE_PATH=/home/kartickr/dftfe_development/DFTFE_MinvHX/build/release/real/
export UCX_LOG_LEVEL=ERROR
export OMP_NUM_THREADS=1
export DFTFE_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export ELPA_DEFAULT_omp_threads=1
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFile_ReS2.prm > outputReS2

