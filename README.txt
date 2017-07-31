DFT-FE (Real Space finite-element based KSDFT implementation)

Installation:

1) Install deal.II (version >=8.4 recommended)

  -- Download CMake [http://www.cmake.org/download/]
  -- Add CMake to your path (e.g. $ PATH="/path/to/cmake/Contents/bin":"$PATH"), preferably in a shell configuration file 
  -- Download and install Deal.II following instructions from from https://www.dealii.org/download.html 
  -- If a deal.II binary is downloaded, open it and follow the instructions in the terminal window. Pre-build binaries recommended for OSX. 
  -- If deal.II is installed from the source, the MPI and p4est libraries must be installed as prerequisites if they are not already installed.
  -- If not compiled as part of deal.II, install Petsc and Slepc with both real (For Non-periodic problems) and complex (For Periodic problems) data types.
  -- Important: Add deal.II directory path to DEAL_II_DIR environment variable: $export DEAL_II_DIR = /path/to/deal.ii/

2) Install Alglib and Libxc

  -- http://www.alglib.net/
      * Download Alglib free C++ edition
      * After downloading go to $HOME/alglib/cpp/src, create shared library by first compiling all cpp files and then linking them to a shared library.
            Eg: To compile using g++ compiler do "g++ -c -fPIC *.cpp" and then to link into a shared library do "g++ *.o -shared -o libAlglib.so"

  -- http://octopus-code.org/wiki/Libxc

3) Clone the repo

  -- $ git clone https://userid@bitbucket.org/rudraa/dft-fe.git (OR) git clone git@bitbucket.org:userid/dft-fe.git
  -- $ cd dft-fe 
  -- $ git checkout master 

4) Setting the paths

  -- set the paths related to alglib, libxc, petsc, slepc both real and complex in CMakeLists.txt in the "dft-fe" folder

5) Install

  -- Ensure DEAL_II_PATH environment variable is set correctly ( check: $echo $DEAL_II_DIR )
  -- $ ./setup.sh (You may need to set permissions. e.g.: $ chmod u=+x setup.sh)

Execution: 

1) Do the following from the folder containing parameterFile.prm:
  -- $ mpirun -np numProcs ./dftRun parameterFile.prm (You may need to set permissions for dftRun. e.g.: $ chmod u=+x dftRun)
  -- Note: numProcs is the number of parallel processes supplied to mpirun. For serial runs numProcs is 1.
