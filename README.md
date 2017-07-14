DFT-FE (Real Space KSDFT implementation)

Installation:

1) Install deal.II (version >=8.5 recommended)
  + Download CMake [http://www.cmake.org/download/]
  + Add CMake to your path (e.g. $ PATH="/path/to/cmake/Contents/bin":"$PATH"), preferably in a shell configuration file 
  + Download and install Deal.II following instructions from from https://www.dealii.org/download.html 
  + If a Deal.II binary is downloaded, open it and follow the instructions in the terminal window 
  + If Deal.II is installed from the source, the MPI and p4est libraries must be installed as prerequisites.

2) Clone the repo
  + $ git clone https://rudraa@bitbucket.org/rudraa/dft.git (OR) git clone git@bitbucket.org:rudraa/dft.git
  + $ cd dft 
  + $ git checkout master 


Simulations:

Run a simulation:
  For debug mode [default mode, very slow]: 
  + $ cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Debug 
  For optimized mode:
  + $ cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Release 
  and 
  + $ make 
  Execution (serial runs): 
  + $ make run 
  Execution (parallel runs): 
  + $ mpirun -np nprocs ./main 
  [here nprocs denotes the number of processors]
  
Visualization:
  Output of the primal fields (if any) is in standard vtk format (parallel:*.pvtu, serial:*.vtu files) which can be visualized with the 
  following open source applications:
  1. VisIt (https://wci.llnl.gov/simulation/computer-codes/visit/downloads)
  2. Paraview (http://www.paraview.org/download/)