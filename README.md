DFT-FE : Density Functional Theory With Finite-Elements 
=======================================================


About
-----

DFT-FE is a C++ code for material modeling from first principles using Kohn-Sham density functional theory.
It is based on adaptive finite-element based methodologies and handles all-electron and pseudopotential calculations in the 
same framework while accomodating arbitrary boundary conditions. dft-fe code builds on top of the deal.II library for everything 
that has to do with finite elements, geometries, meshes, etc., and, through deal.II on p4est for parallel adaptive mesh handling. 



Installation instructions
-------------------------

The steps to install the necessary dependencies and DFT-FE itself are described
in the Installation instructions section of the DFT-FE [manual](/dftfedevelopers/dftfe/src/manualSkeletonWithParametersList/doc/manual/manual.pdf).



Running DFT-FE
--------------

Instructions on how to run and DFT-FE can also be found in the DFT-FE [manual](/dftfedevelopers/dftfe/src/manualSkeletonWithParametersList/doc/manual/manual.pdf). 



Contributing to DFT-FE
----------------------




More information
----------------

For more information see:

 - The official website at (give link)
 
 - The current [manual](/dftfedevelopers/dftfe/src/manualSkeletonWithParametersList/doc/manual/manual.pdf)
 
 - DFT-FE is primarily based on the deal.II library. If you have particular questions about deal.II, contact the [deal.II discussion groups](https://groups.google.com/d/forum/dealii).
 
 - If you have specific questions about DFT-FE that are not suitable for the public bitbucket issue list, you can contact the principal developers:

    - Phani Motamarri: phanim@umich.edu
    - Sambit Das: dsambit@umich.edu



License
-------

DFT-FE is published under [LGPL v2.1 or newer](LICENSE).
