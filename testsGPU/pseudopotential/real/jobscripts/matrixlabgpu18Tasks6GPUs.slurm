#!/bin/sh
#SBATCH --job-name=GPUcTestReal             # Job name
#SBATCH --ntasks-per-node=18                # Number of tasks per node
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
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
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe Input_MD_0.prm > output_MD_0
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe Input_MD_1.prm > output_MD_1
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe Input_MD_2.prm > output_MD_2
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_2_spingpu.prm > outputMg2x_2_spin_gpu
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_12.prm > outputMg2x_12
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_13.prm > outputMg2x_13
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_17.prm > outputMg2x_17
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileBe.prm > outputBe
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFile_LLZO.prm > outputLLZO
srun -n $SLURM_NTASKS --mpi=pmi2 $DFTFE_PATH/dftfe parameterFile_ReS2.prm > outputReS2

srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileN2_1.prm > outputN2_1
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileN2_2.prm > outputN2_2
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileN2_3.prm > outputN2_3
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileN2_4.prm > outputN2_4
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileN2_5.prm > outputN2_5
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_11.prm > outputMg2x_11
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_14.prm > outputMg2x_14
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_15.prm > outputMg2x_15
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe parameterFileMg2x_16.prm > outputMg2x_16
srun -n 6 --mpi=pmi2 $DFTFE_PATH/dftfe Pt3Ni_hubbard_spin.prm   > outputPt3Ni_hubbard_spin
