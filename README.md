DFT-FE : Density Functional Theory With Finite-Elements 
=======================================================


About
-----
DFT-FE is a C++ code for materials modeling from first principles using Kohn-Sham density functional theory. The origins of DFT-FE were in the  [Computational Materials Physics Group](http://www-personal.umich.edu/~vikramg) at the University of Michigan, Ann Arbor, with Vikram Gavini, Professor of Mechanical Engineering and Materials Science & Engineering, as the principal investigator broadly overseeing the effort. The current development efforts span across the [Computational Materials Physics Group](http://www-personal.umich.edu/~vikramg) (Prof. Vikram Gavini, Dr. Sambit Das) at the University of Michigan and the [MATRIX lab](http://cds.iisc.ac.in/faculty/phanim/) (Prof. Phani Motamarri) at the Indian Institute of Science.

DFT-FE is based on an adaptive finite-element discretization that handles pseudopotential and all-electron calculations in the same framework, and incorporates scalable and efficient solvers for the solution of the Kohn-Sham equations. Importantly, DFT-FE can handle periodic, semi-periodic and non-periodic boundary conditions and general geometries. DFT-FE can be run on massively parallel many-core CPU and hybrid CPU-GPU architectures (tested up to ~200,000 cores on many-core CPUs and ~24,000 GPUs on hybrid CPU-GPU architectures). DFT-FE is capable of fast and accurate large-scale pseudopotential DFT calculations, reaching 50,000-100,000 electrons. 

Installation instructions
-------------------------

DFT-FE code builds on top of the deal.II library for everything that has to do with finite elements, geometries, meshes, etc., and, through deal.II on p4est for parallel adaptive mesh handling.
The steps to install the necessary dependencies and DFT-FE itself are described in the *Installation* section of the DFT-FE manual (compile doc/manual/manual.tex or download the development version manual [here](https://github.com/dftfeDevelopers/dftfe/blob/manual/manual-develop.pdf)). 

We have created several shell based installation scripts for the development version of DFT-FE (`publicGithubDevelop` branch) on various machines:
  - [OLCF Frontier](https://github.com/dftfeDevelopers/install_DFTFE/tree/frontierScript)
  - [NERSC Perlmutter](https://github.com/dftfeDevelopers/install_DFTFE/tree/perlmutterScript)
  - [ALCF Polaris](https://github.com/dftfeDevelopers/install_DFTFE/tree/polarisScript)
  - [UMICH Greatlakes](https://github.com/dftfeDevelopers/install_DFTFE/tree/greatlakesScript)
    


Running DFT-FE
--------------

Instructions on how to run DFT-FE including demo examples can also be found in the *Running DFT-FE* section of the manual (compile doc/manual/manual.tex or download the development version manual [here](https://github.com/dftfeDevelopers/dftfe/blob/manual/manual-develop.pdf)). Beyond the demo examples in the manual, we also refer to our [benchmarks repository](https://github.com/dftfeDevelopers/dftfe-benchmarks) which contains several accuracy and performance benchmarks on a range of system sizes.


Contributing to DFT-FE
----------------------
Learn more about contributing to DFT-FE's development [here](https://github.com/dftfeDevelopers/dftfe/wiki/Contributing).


More information
----------------

 - See the official [website](https://sites.google.com/umich.edu/dftfe) for information on code capabilities, appropriate referencing of the code, acknowledgements, and news related to DFT-FE.
  
 - See Doxygen generated [documentation](https://dftfedevelopers.github.io/dftfe/).

 - For questions about DFT-FE, installation, bugs, etc., use the [DFT-FE discussion forum](https://groups.google.com/forum/#!forum/dftfe-user-group). 

 - For latest news, updates, and release announcements about DFT-FE please send an email to dft-fe.admin@umich.edu, and we will add you to our announcement mailing list.
 
 - DFT-FE is primarily based on the [deal.II library](http://www.dealii.org/). If you have particular questions about deal.II, use the [deal.II discussion forum](https://www.dealii.org/mail.html).
 
 - If you have specific questions about DFT-FE that are not suitable for the public and archived mailing lists, you can contact the following:
    - Phani Motamarri: phanim@iisc.ac.in
    - Sambit Das: dsambit@umich.edu
    - Vikram Gavini: vikramg@umich.edu 

 - The following people have significantly contributed either in the past or current and advanced DFT-FE's goals: (All the underlying lists are in alphabetical order based on last name)
   - Mentors/Development leads
      - Dr. Sambit Das (University of Michigan Ann Arbor, USA)
      - Prof. Vikram Gavini (University of Michigan Ann Arbor, USA)
      - Prof. Phani Motamarri (Indian Institute of Science, India)

   - Principal developers  
       - Dr. Sambit Das (University of Michigan Ann Arbor, USA)
       - Prof. Phani Motamarri (Indian Institute of Science, India)
    
 - A complete list of the many authors that have contributed to DFT-FE can be found at [authors](authors).    

License
-------

DFT-FE is published under [LGPL v2.1 or newer](https://github.com/dftfeDevelopers/dftfe/blob/publicGithubDevelop/LICENSE).
