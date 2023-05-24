#!/bin/bash
#SBATCH -A MAT187_crusher
#SBATCH -J complextest
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus-per-node=6
#SBATCH --gpu-bind=closest

export OMP_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=1
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"

srun -n 6 -c 1 ./dftfe parameterFileMg2x_1.prm > outputMg2x_1

srun -n 6 -c 1 ./dftfe parameterFileMg2x_2.prm > outputMg2x_2

srun -n 6 -c 1 ./dftfe parameterFileMg2x_3.prm > outputMg2x_3

srun -n 6 -c 1 ./dftfe parameterFileMg2x_4.prm > outputMg2x_4

srun -n 6 -c 1 ./dftfe parameterFileMg2x_5.prm > outputMg2x_5

srun -n 6 -c 1 ./dftfe parameterFileMg2x_6.prm > outputMg2x_6

srun -n 6 -c 1 ./dftfe parameterFileBe.prm > outputBe

