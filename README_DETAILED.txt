--------------------------------------------------------
Detailed Installation and execution procedure of dft-fe:
--------------------------------------------------------
As mentioned in README.md, we assume that you have already installed the dependency packages "deal.II", "alglib", "libxc", "petsc", "slepc".
If not follow the steps mentioned in README.md, else follow the steps below:

--------------------------
Follow the steps in order:
--------------------------

*************
Installation:
*************

(1) Loading of relevant modules:
      -- Modules related to cmake, compilers, openmpi has to be loaded.
        For example on flux systems located at umich, you may need to do 
          $ module load cmake
          $ module load intel/17.0.1
          $ module load openmpi/1.10.2/intel/17.0.1

(2) Cloning the repository:
      -- Execute the following commands on your command-line
         $ git clone https://phanim@bitbucket.org/rudraa/dft-fe.git
         $ cd dft-fe
         $ git checkout master

(3) Setting the relevant paths in CMakeLists.txt:          
      -- Open the file CMakeLists.txt in the dft-fe folder and set the include and library paths for alglib, libxc, petsc, slepc as follows:
         SET(ALGLIB_INCLUDE "path to alglib include folder")
         SET(ALGLIB_LIB "path to alglib library libAlglib.so")

         SET(LIBXC_INCLUDE "path to libxc include folder")
         SET(LIBXC_LIB  "path to libxc library libxc.a")

         SET(PETSC_REAL_INCLUDE "path to petsc-double include")
         SET(PETSC_REAL_LIB "path to petsc-double library libpetsc.so")

         SET(SLEPC_REAL_INCLUDE "path to slepc-double include")
         SET(SLEPC_REAL_LIB "path to slepc-double library libslepc.so")

         SET(PETSC_COMPLEX_INCLUDE "path to petsc-complex include")
         SET(PETSC_COMPLEX_LIB "path to petsc-complex library libpetsc.so")
         
         SET(SLEPC_COMPLEX_INCLUDE "path to slepc-complex include")
         SET(SLEPC_COMPLEX_LIB "path to slepc-complex library libslepc.so")

      -- Set DEAL_II_PATH environment variable to path containing deal.II install folder in your command-line as follows:
         $ export DEAL_II_DIR = /path/to/deal.ii/

(4) Installing the dft-fe code:
      -- Do the following from your command line
          $ ./setup.sh (You may need to set permissions. e.g.: $ chmod u+x setup.sh)


**********
Execution:        
**********
   (1) Running the executable for a given material system requires the user to prepare a parameter file "parameterFile.prm" containing 
   user-defined parameters. You can find example of a parameterFile in the subfolders of "/dft-fe/examples".

   (2) Do "$mpirun -np numProcs ./dftRun parameterFile.prm>outputFileName" on the terminal or by using the job submission script. Here numProce denotes
   the number of parallel processors supplied to mpirun. You may need to set permission for dftRun using "chmod u+x dftRun". Note that "dftRun"
   is located in the main folder dft-fe and it is recommended to point to this location while running the "dftRun" executable irrespective of the folder
   your "parameterFile.prm" lies in.
      
   (3) Demo on an example problem: 

     -- There is an "examples" folder in the dft-fe folder containing various examples on sample atomic systems both for all-electron, pseudopotential
        including non-periodic and periodic cases for each type. The following assumes we are in "/examples/allElectron/nonPeriodic/carbonSingleAtom/"

     -- Most important aspect to run the dft-fe code is to prepare the finite-element mesh file and coordinates file containing atomic coordinates. 
        For this example problem, you can find sample finite-element mesh file "mesh.inp" and relevant coordinate file "coordinates.inp" in the folder.

          ** The mesh file to be supplied to dft-fe code must be in "Unstructured Cell Data (UCD)" format (http://www.csit.fsu.edu/~burkardt/data/ucd/ucd.html).
             UCD mesh file name must end with the extension ".inp". Look at the path "/dft-fe/utils", you will find a script "convertCubit2UCD.py" that helps to 
             convert finite-element meshes with extension ".exo" into UCD format by making use of "Cubit" software. The following explains this procedure 
             to generate mesh file in UCD format.

              * After the relevant mesh gets generated in Cubit (or any other mesh generation software), export the mesh as "meshFileName.exo" file.
              * Assuming that the Cubit is installed or cubit module is loaded, type "clarox" on command line to open the Cubit GUI.
              * In the GUI, go to File->Import, a window opens. Then choose "Files of type" to "Genesis/Exodus" and then select the "meshFileName.exo" file.
              * After the "meshFileName.exo" is imported, go to Tools->Play Journal File, a window opens. Then choose "Files of type" to "Python Files"
                and then select the "convertCubit2UCD.py" file. You will now find "mesh.inp" in the same location of your "convertCubit2UCD.py" script.

     -- The geometry of the given atomic system is specified in "coordinates.inp". Each row of coordinates.inp indicates the atomic charge, valence atomic charge,
        X, Y, Z cartesian coordinates of each atom in the sytem. In the case of periodic system, each row of "coordinates.inp" must correspond to atomic charge, 
        valence atomic charge, fractional coordinates associated with each atom.

     -- Now comes the parameter file "parameterFile.prm". This file contains the various input parameters specifying
        the name and location of mesh files, coordinate files, solver tolerances etc.,
           
           Note the following:
            * Every parameter option is preceded by the keyword "set" and followed by "=" symbol.
            * Lines starting with '#' are treated as comment lines and will be ignored by the executable.
            * Every parameter input has a comment above it which explains briefly about the parameter input option.


     -- Once the mesh file, coordinates file are ready, you can run the executable "dftRun" as mentioned in Step(2) and redirect the output to a output file 
        name. Compare the output you obtained with that provided in /examples/allElectron/nonPeriodic/carbonSingleAtom/results folder


   (4) Miscellaneous Instructions:       

     -- In the case of periodic calculation, one has to supply a file containing lattice vectors associated with the given periodic simulation domain. For example,
        see the parameter files in /examples/allElectron/periodic/simpleCubicCarbon or see /examples/pseudopotential/periodic/faceCenteredCubicAluminum. In addition,    
        one has to also point to required k-point quadrature rule in the parameter file. The list of k-point quadrature rules are given in "/dft-fe/data/kPointList".
        Look at the parameter files in the examples folder containing periodic cases. Currently fully periodic with cubic/cuboidal unit-cells are handled.

     -- The dft-fe code executable automatically picks up the single atom radial wave functions to be used as initial guesses from the folders 
        "/dft-fe/data/electronicStructure/allElectron" or "/dft-fe/data/electronicStructure/pseudopotential". Folders inside this path are named as z1, z2, z6 etc., 
        where the number following z indicates the atomic number. Populate whenever necessary with single atom wavefunctions based on the atom-type 
        in your given problem.

     -- dft-fe code currently handles non-local Troullier Martins pseudopotentials. The pseudpotential files are located at 
        "/dft-fe/data/electronicStructure/pseudopotential".  As explained before, folders inside this path are named as z1, z2, z6 etc., 
        where the number following z indicates the atomic number. Each "zX" contains a folder "pseudoAtomData" which contains the files corresponding to radial 
        parts of the pseudowavefunctions and angular momentum dependent potentials along with the file containing local part of the pseudopotential. The file 
        "PseudoAtomData" present inside this folder embeds this information. First line of this file indicates the total number of pseudowavefunctions, 
        each of the subsequent lines indicate the radial Id, azimuthal quantum number and magnetic quantum number of the associated pseudowavefunctions. Next lines 
        indicate the filenames containing the data corresponding to radial parts of the pseudowavefunctions. Subsequently, the name of the local pseudopotential file
        is provided followed with the number of angular momentum dependent potentials. Next lines indicate the radial id, azimuthal quantum number of the associated    
        pseudopotential file. Finally, the filenames containing the data corresponding to angular momentum dependent potentials is provided. 
        Note that the name of local pseudopotential file has to be "locPot.dat"
        

      
            
         


        








