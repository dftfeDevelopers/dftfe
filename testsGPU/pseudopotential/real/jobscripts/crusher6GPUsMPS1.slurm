#!/bin/bash
#SBATCH -A MAT187_crusher
#SBATCH -J realmps1
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest

export OMP_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=1
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"

srun -n 6 -c 1 ./dftfe parameterFileN2_1.prm > outputN2_1
srun -n 6 -c 1 ./dftfe parameterFileN2_2.prm > outputN2_2
srun -n 6 -c 1 ./dftfe parameterFileN2_3.prm > outputN2_3
srun -n 6 -c 1 ./dftfe parameterFileN2_4.prm > outputN2_4
srun -n 6 -c 1 ./dftfe parameterFileN2_5.prm > outputN2_5
srun -n 6 -c 1 ./dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun -n 6 -c 1 ./dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun -n 6 -c 1 ./dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun -n 6 -c 1 ./dftfe parameterFileMg2x_11.prm > outputMg2x_11
srun -n 6 -c 1 ./dftfe parameterFileMg2x_14.prm > outputMg2x_14
srun -n 6 -c 1 ./dftfe parameterFileMg2x_15.prm > outputMg2x_15
srun -n 6 -c 1 ./dftfe parameterFileMg2x_16.prm > outputMg2x_16

