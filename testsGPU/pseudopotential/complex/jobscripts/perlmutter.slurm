#!/global/homes/d/dsambit/perlmutter/bin/rc
#SBATCH -A m2360_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --job-name complextests
#SBATCH -t 1:30:00
#SBATCH -n 6
#SBATCH --ntasks-per-node=3
#SBATCH -c 32
#SBATCH --gpus-per-node=3
#SBATCH --gpu-bind=map_gpu:0*1,1*1,2*1,3*0

module load gcc-native/12.3
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cudatoolkit
module load cray-libsci
module load cmake
module load nccl


SLURM_CPU_BIND='cores'
OMP_NUM_THREADS=1
MPICH_GPU_SUPPORT_ENABLED=1
LD_LIBRARY_PATH = $LD_LIBRARY_PATH:$WD/env2/lib
LD_LIBRARY_PATH = $LD_LIBRARY_PATH:$WD/env2/lib64

srun ./dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun ./dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun ./dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun ./dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun ./dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun ./dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun ./dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun ./dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun ./dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun ./dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun ./dftfe parameterFileBe.prm > outputBe
srun ./dftfe parameterFileTiAl.prm > outputTiAl_hubbard
srun ./dftfe parameterFileTiAl_mixedPrec.prm > outputTiAl_hubbard_mixedPrec_mpi6
srun ./dftfe parameterFileFeCuPt2_scf.prm > outputFeCuPt2_scf
srun ./dftfe parameterFileFeCuPt2_pdos.prm > outputFeCuPt2_pdos
srun ./dftfe parameterFileBccFe_scf.prm > outputBccFe_scf
srun ./dftfe parameterFileBccFe_scfConstraintMag.prm > outputBccFe_scfConstraintMag
srun ./dftfe parameterFileBccFe_pdos.prm > outputBccFe_pdos
srun ./dftfe parameterFileBccFe_relax.prm > outputBccFe_relax
srun ./dftfe parameterFileGaAs.prm > outputGaAs
srun ./dftfe parameterFileGaAs_BANDS.prm > outputGaAs_bands
srun ./dftfe parameterFileBccFe_relaxFullMassMatrix.prm > outputBccFe_relaxFullMassMatrix
srun ./dftfe parameterFileBccFe_scan.prm > outputBccFe_scan
srun ./dftfe parameterFileBccFe_scan_kerker.prm > outputBccFe_scan_kerker
