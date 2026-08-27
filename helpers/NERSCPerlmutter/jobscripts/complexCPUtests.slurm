#!/bin/bash
#SBATCH -A m5214
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name complexCPUtests
#SBATCH -t 04:00:00
#SBATCH -N 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

BUILD_DIR=/global/homes/r/rezgar/dftfe/build/releaseCPU/complex

cd $BUILD_DIR

ctest -E "TiAl_hubbard|bccFe_01_a|bccFe_01_b|bccFe_afm" --output-on-failure 2>&1 | tee ctest-complex-perlmutter-CPU.out
