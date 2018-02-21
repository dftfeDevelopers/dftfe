What is dft-fe?
---------------

dft-fe is a C++ code for material modeling from first principles using Kohn-Sham density functional theory.
It is based on adaptive finite-element based methodologies and handles all-electron and pseudopotential calculations in the 
same framework while accomodating arbitrary boundary conditions. dft-fe code builds on top of the deal.II library for everything 
that has to do with finite elements, geometries, meshes, etc., and, through deal.II on p4est for parallel adaptive mesh handling. 

*********************************************
Getting started:
*********************************************
dft-fe requires deal.II compiled with p4est, petsc, slepc, alglib, spglib and libxc to be pre-installed. Refer to install.txt
for more details about installing and demo.txt for executing the code.


