#!/bin/bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:20:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q prod
#PBS -A DFTCalc2
#PBS -N DFTFE_SYCL


echo "Number of Nodes Allocated      = $PBS_NUM_NODES"
echo "Number of Tasks Allocated      = $PBS_NUM_PPN"
echo "CPU on node                    = $PBS_NUM_CPUS"
module load boost

export MPIR_CVAR_ENABLE_GPU=0
cd $PBS_O_WORKDIR

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS_PER_NODE=6
NDEPTH=1
NTHREADS=1
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export DFTFE_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1

export CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
export GPU_BIND_SCHEME="--gpu-bind=list:0.0:0.1:1.0:1.1:2.0:2.1:3.0:3.1:4.0:4.1:5.0:5.1"


export BASE=/lus/flare/projects/DFTCalc2/dftfeDependenciesNew/dftfe/install/complex
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_1.prm > outputMg2x_1
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_2.prm > outputMg2x_2
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_3.prm > outputMg2x_3
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_4.prm > outputMg2x_4
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_5.prm > outputMg2x_5
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_6.prm > outputMg2x_6
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_7.prm > outputMg2x_7
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_8.prm > outputMg2x_8
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_9.prm > outputMg2x_9
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_10.prm > outputMg2x_10
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBe.prm > outputBe
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileTiAl_mixedPrec.prm > outputTiAl_hubbard_mixedPrec_mpi6
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileTiAl.prm > outputTiAl_hubbard
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileFeCuPt2_scf.prm > outputFeCuPt2_scf
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileFeCuPt2_pdos.prm > outputFeCuPt2_pdos
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_scf.prm > outputBccFe_scf
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_scfConstraintMag.prm > outputBccFe_scfConstraintMag
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_pdos.prm > outputBccFe_pdos
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_relax.prm > outputBccFe_relax
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileGaAs.prm > outputGaAs
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileGaAs_BANDS.prm > outputGaAs_bands
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_relaxFullMassMatrix.prm > outputBccFe_relaxFullMassMatrix
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_scan.prm > outputBccFe_scan
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_scan_kerker.prm > outputBccFe_scan_kerker
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileBccFe_scan_useLibxcFalse.prm > outputBccFe_scan_useLibxcFalse