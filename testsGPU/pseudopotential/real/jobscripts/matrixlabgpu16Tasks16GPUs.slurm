#!/bin/sh
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

. /shared/spack/share/spack/setup-env.sh    ###--- For bash Shell 
spack load gcc@11.3.0 cuda@11.7.0 boost@1.76.0 intel-oneapi-mkl@2022 cmake%gcc@11.3.0 gmake%gcc@11.3.0 ninja openmpi%gcc@11.3.0 numactl@2.0.14%gcc@11.3.0 p4est ^openmpi

###needs to change this DFTFE_PATH
export DFTFE_PATH=/home/nikhilk/dftfe/publicGithubDevelop/build/release/real/
export UCX_LOG_LEVEL=ERROR
export OMP_NUM_THREADS=1
export DFTFE_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export ELPA_DEFAULT_omp_threads=1
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFile_ReS2.prm > outputReS2

