DFT-FE (Real Space KSDFT implementation)

Installation:

1) Install deal.II (version >=8.4 recommended)
  + Download CMake [http://www.cmake.org/download/]
  + Add CMake to your path (e.g. $ PATH="/path/to/cmake/Contents/bin":"$PATH"), preferably in a shell configuration file 
  + Download and install Deal.II following instructions from from https://www.dealii.org/download.html 
  + If a deal.II binary is downloaded, open it and follow the instructions in the terminal window. Pre-build binaries recommended for OSX. 
  + If deal.II is installed from the source, the MPI and p4est libraries must be installed as prerequisites.
  + If not compiled as part of deal.II, install Petsc and Slepc with both real (For Non-periodic problems) and complex (For Periodic problems) data types.
  + Important: Add deal.II main directory path to DEAL_II_DIR environemnt variable: $export DEAL_II_DIR = /path/to/deal.ii/

2) Install Alglib and Libxc
  + http://www.alglib.net/
  + http://octopus-code.org/wiki/Libxc

2) Clone the repo
  + $ git clone https://rudraa@bitbucket.org/rudraa/dft.git (OR) git clone git@bitbucket.org:rudraa/dft.git
  + $ cd dft 
  + $ git checkout master 

3) Install
  + Ensure DEAL_II_PATH environment variable is set correctly ( check: $echo $DEAL_II_DIR )
  + $ ./setup.sh (You may need to set permissions. e.g.: $ chmod u=+x setup.sh)

Execution: 

1) From folder containing parameterFile.prm:
  + $ ./dftRun -np numProcs parameterFile.prm (You may need to set permissions for dftrun. e.g.: $ chmod u=+x dftrun)
  + Note: numProcs is the number of parallel processes supplied to mpirun. For serial runs numProcs is 1.
