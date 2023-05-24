#!/bin/bash
#SBATCH -A MAT187_crusher
#SBATCH -J realmps3
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-gpu=3
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest

export OMP_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=1
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"

srun -n 18 -c 1 ./dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun -n 18 -c 1 ./dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
srun -n 18 -c 1 ./dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun -n 18 -c 1 ./dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun -n 18 -c 1 ./dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun -n 18 -c 1 ./dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun -n 18 -c 1 ./dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun -n 18 -c 1 ./dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun -n 18 -c 1 ./dftfe parameterFileMg2x_12.prm > outputMg2x_12
srun -n 18 -c 1 ./dftfe parameterFileMg2x_13.prm > outputMg2x_13
srun -n 18 -c 1 ./dftfe Input_MD_0.prm > output_MD_0
srun -n 18 -c 1 ./dftfe Input_MD_1.prm > output_MD_1
srun -n 18 -c 1 ./dftfe Input_MD_2.prm > output_MD_2
srun -n 18 -c 1 ./dftfe parameterFileBe.prm > outputBe

