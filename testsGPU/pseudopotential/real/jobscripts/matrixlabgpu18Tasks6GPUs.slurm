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

module load gcc9.2 cmake3.19 cuda11gcc9.2 mkl/2021.2.0 openmpicuda11

export UCX_LOG_LEVEL=ERROR

srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe Input_MD_0.prm > output_MD_0
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe Input_MD_1.prm > output_MD_1
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe Input_MD_2.prm > output_MD_2
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_12.prm > outputMg2x_12
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileMg2x_13.prm > outputMg2x_13
srun -n $SLURM_NTASKS --mpi=pmi2 ./dftfe parameterFileBe.prm > outputBe

srun -n 6 --mpi=pmi2 ./dftfe parameterFileN2_1.prm > outputN2_1
srun -n 6 --mpi=pmi2 ./dftfe parameterFileN2_2.prm > outputN2_2
srun -n 6 --mpi=pmi2 ./dftfe parameterFileN2_3.prm > outputN2_3
srun -n 6 --mpi=pmi2 ./dftfe parameterFileN2_4.prm > outputN2_4
srun -n 6 --mpi=pmi2 ./dftfe parameterFileN2_5.prm > outputN2_5
srun -n 6 --mpi=pmi2 ./dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun -n 6 --mpi=pmi2 ./dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun -n 6 --mpi=pmi2 ./dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun -n 6 --mpi=pmi2 ./dftfe parameterFileMg2x_11.prm > outputMg2x_11
srun -n 6 --mpi=pmi2 ./dftfe parameterFileMg2x_14.prm > outputMg2x_14
srun -n 6 --mpi=pmi2 ./dftfe parameterFileMg2x_15.prm > outputMg2x_15
srun -n 6 --mpi=pmi2 ./dftfe parameterFileMg2x_16.prm > outputMg2x_16

