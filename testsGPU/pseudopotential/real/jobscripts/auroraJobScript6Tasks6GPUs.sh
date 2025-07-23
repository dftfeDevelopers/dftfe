#!/bin/bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
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


export BASE=/lus/flare/projects/DFTCalc2/dftfeDependenciesNew/dftfe/install/real

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileN2_1.prm > outputN2_1
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileN2_2.prm > outputN2_2
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileN2_3.prm > outputN2_3
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileN2_4.prm > outputN2_4
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileN2_5.prm > outputN2_5
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_8.prm > outputMg2x_8
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_9.prm > outputMg2x_9
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_10.prm > outputMg2x_10
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_11.prm > outputMg2x_11
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_14.prm > outputMg2x_14
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe parameterFileMg2x_15.prm > outputMg2x_15
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe Pt3Ni_hubbard_spin.prm > outputPt3Ni_hubbard_spin
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ${CPU_BIND_SCHEME} ${GPU_BIND_SCHEME} $BASE/dftfe Pt3Ni_hubbard_spin_mixedPrec.prm > outputPt3Ni_hubbard_spin_mixedPrec
