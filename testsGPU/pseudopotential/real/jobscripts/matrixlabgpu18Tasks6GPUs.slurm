#!/bin/bash
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

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"


###needs to change this DFTFE_PATH
export DFTFE_PATH=/home/kartickr/dftfe_development/DFTFE_MinvHX/build/release/real/
export UCX_LOG_LEVEL=ERROR
export OMP_NUM_THREADS=1
export DFTFE_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export ELPA_DEFAULT_omp_threads=1
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe Input_MD_0.prm > output_MD_0
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe Input_MD_1.prm > output_MD_1
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe Input_MD_2.prm > output_MD_2
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_1.prm > outputMg2x_1
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_2_spingpu.prm > outputMg2x_2_spin_gpu
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_2.prm > outputMg2x_2
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_3.prm > outputMg2x_3
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_4.prm > outputMg2x_4
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_5.prm > outputMg2x_5
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_6.prm > outputMg2x_6
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_7.prm > outputMg2x_7
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_12.prm > outputMg2x_12
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_13.prm > outputMg2x_13
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_17.prm > outputMg2x_17
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileBe.prm > outputBe
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFile_LLZO.prm > outputLLZO
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFile_ReS2.prm > outputReS2
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFile_ReS2FullMassMatrix.prm > outputReS2FullMassMatrix

mpirun -n 6  $DFTFE_PATH/dftfe parameterFileN2_1.prm > outputN2_1
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileN2_2.prm > outputN2_2
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileN2_3.prm > outputN2_3
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileN2_4.prm > outputN2_4
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileN2_5.prm > outputN2_5
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileMg2x_8.prm > outputMg2x_8
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileMg2x_9.prm > outputMg2x_9
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileMg2x_10.prm > outputMg2x_10
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileMg2x_11.prm > outputMg2x_11
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileMg2x_14.prm > outputMg2x_14
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileMg2x_15.prm > outputMg2x_15
mpirun -n 6  $DFTFE_PATH/dftfe Pt3Ni_hubbard_spin.prm   > outputPt3Ni_hubbard_spin
mpirun -n 6  $DFTFE_PATH/dftfe Pt3Ni_hubbard_spin_mixedPrec.prm > outputPt3Ni_hubbard_spin_mixedPrec
