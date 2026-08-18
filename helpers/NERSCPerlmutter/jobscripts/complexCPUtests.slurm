#!/bin/bash
#SBATCH -A m5214
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name complexCPUtests
#SBATCH -t 04:00:00
#SBATCH -N 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DFTFE_EXE=/global/homes/r/rezgar/dftfe/build/releaseCPU/complex/dftfe
TEST_DIR=/global/homes/r/rezgar/dftfe/build/releaseCPU/complex/tests/dft/pseudopotential/complex

cd $TEST_DIR

srun -n 32 $DFTFE_EXE TiAl_hubbard.prm            > output_TiAl_hubbard 2>&1
srun -n 32 $DFTFE_EXE TiAl_hubbard_singlePrec.prm > output_TiAl_hubbard_singlePrec 2>&1
srun -n 8  $DFTFE_EXE bccFe_01_a.prm              > output_bccFe_01_a 2>&1
srun -n 8  $DFTFE_EXE bccFe_01_b.prm              > output_bccFe_01_b 2>&1
srun -n 32 $DFTFE_EXE bccFe_afm.prm               > output_bccFe_afm 2>&1
