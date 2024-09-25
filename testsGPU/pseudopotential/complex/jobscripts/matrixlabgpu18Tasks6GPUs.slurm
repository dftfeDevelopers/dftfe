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

. /shared/spack/share/spack/setup-env.sh    ###--- For bash Shell 
spack load gcc@11.3.0 cuda@11.7.0 boost@1.76.0 intel-oneapi-mkl@2022 cmake%gcc@11.3.0 gmake%gcc@11.3.0 ninja openmpi%gcc@11.3.0 numactl@2.0.14%gcc@11.3.0 p4est ^openmpi

# export DFTFE_PATH=/home/nikhilk/dftfe/chebFiltSP/build/release/complex/
export DFTFE_PATH=/home/nikhilk/dftfe/energyResidual/build/release/complex
export UCX_LOG_LEVEL=ERROR
export OMP_NUM_THREADS=1
export DFTFE_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export ELPA_DEFAULT_omp_threads=1

srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileBe.prm     > outputBe
