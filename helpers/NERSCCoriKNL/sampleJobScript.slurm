#!/bin/bash -l

#SBATCH -q debug
#SBATCH -N 512
#SBATCH -C knl
#SBATCH -S 4
#SBATCH --switches=2@20:00:00
#SBATCH -t 00:25:00
#SBATCH -A m2360
#SBATCH --mail-user=user@umich.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -J 6x6x6
#SBATCH -o 6x6x6.o%j

# setup and run
echo "Job $SLURM_JOB_NAME, ID $SLURM_JOB_ID"
cd $SLURM_SUBMIT_DIR

module load craype-hugepages2M

export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


sbcast /global/homes/d/user/dftfeProduction/build/release/real/main /tmp/main

srun -n 16384 -c 8 --cpu_bind=cores /tmp/main parameterFile.prm > output
