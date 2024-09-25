#!/bin/sh
#SBATCH -A nti115
#SBATCH -J gputests
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 3
#SBATCH --gpus-per-node 6
#SBATCH --ntasks-per-gpu 1
#SBATCH --gpu-bind closest

export OMP_NUM_THREADS=1
export MPICH_VERSION_DISPLAY=1
export MPICH_ENV_DISPLAY=1
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INST/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INST/lib/lib64
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=disabled

export BASE=$WD/src/dftfeDebug/build/release/real

srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe Input_MD_0.prm > output_MD_0
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe Input_MD_1.prm > output_MD_1
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe Input_MD_2.prm > output_MD_2
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_2_spingpu.prm > outputMg2x_2_spin_gpu
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_12.prm > outputMg2x_12
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileMg2x_13.prm > outputMg2x_13
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFileBe.prm > outputBe
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFile_LLZO.prm > outputLLZO
srun -n 18 -c 7 --gpu-bind closest $BASE/dftfe parameterFile_ReS2.prm > outputReS2

