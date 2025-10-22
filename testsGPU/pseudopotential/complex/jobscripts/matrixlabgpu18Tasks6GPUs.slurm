#!/bin/bash
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

module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:$LIBRARY_PATH"


###needs to change this DFTFE_PATH
export DFTFE_PATH=/home/kartickr/dftfe_development/DFTFE_MinvHX/build/release/complex/
export UCX_LOG_LEVEL=ERROR
export OMP_NUM_THREADS=1
export DFTFE_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export ELPA_DEFAULT_omp_threads=1

mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_1.prm > outputMg2x_1
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_2.prm > outputMg2x_2
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_3.prm > outputMg2x_3
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_4.prm > outputMg2x_4
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_5.prm > outputMg2x_5
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_6.prm > outputMg2x_6
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_7.prm > outputMg2x_7
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_8.prm > outputMg2x_8
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_9.prm > outputMg2x_9
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileMg2x_10.prm > outputMg2x_10
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileBe.prm     > outputBe
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileTiAl.prm > outputTiAl_hubbard
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileTiAl_mixedPrec.prm > outputTiAl_hubbard_mixedPrec_mpi6
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileFeCuPt2_scf.prm > outputFeCuPt2_scf
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileFeCuPt2_pdos.prm > outputFeCuPt2_pdos
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileBccFe_scf.prm > outputBccFe_scf
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileBccFe_scfConstraintMag.prm > outputBccFe_scfConstraintMag
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileBccFe_pdos.prm > outputBccFe_pdos
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileBccFe_relax.prm > outputBccFe_relax
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileGaAs.prm > outputGaAs
mpirun -n $SLURM_NTASKS  $DFTFE_PATH/dftfe parameterFileGaAs_BANDS.prm > outputGaAs_bands
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileBccFe_relaxFullMassMatrix.prm > outputBccFe_relaxFullMassMatrix
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileBccFe_scan.prm > outputBccFe_scan
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileBccFe_scan_kerker.prm > outputBccFe_scan_kerker
mpirun -n 6  $DFTFE_PATH/dftfe parameterFileBccFe_scan_useLibxcFalse.prm > outputBccFe_scan_useLibxcFalse