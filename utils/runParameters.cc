// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Phani Motamarri, Sambit Das
//
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <runParameters.h>
#include <fstream>
#include <iostream>

using namespace dealii;

namespace dftfe
{
  namespace internalRunParameters
  {
    void
    declare_parameters(ParameterHandler &prm)
    {
      prm.declare_entry(
        "REPRODUCIBLE OUTPUT",
        "false",
        Patterns::Bool(),
        "[Developer] Limit output to what is reproducible, i.e. don't print timing or absolute paths. This parameter is only used for testing purposes.");


      prm.declare_entry(
        "H REFINED ELECTROSTATICS",
        "false",
        Patterns::Bool(),
        "[Advanced] Compute electrostatic energy on a h refined mesh after each ground-state solve. Default: false.");

      prm.declare_entry(
        "DFT-FE SOLVER MODE",
        "GS",
        Patterns::Selection("GS|MD|NEB"),
        "[Standard] DFT-FE SOLVER MODE: If GS: performs GroundState calculations, ionic and cell relaxation. If MD: performs Molecular Dynamics Simulation. If NEB: performs a NEB calculation");


      prm.declare_entry(
        "VERBOSITY",
        "1",
        Patterns::Integer(-1, 5),
        "[Standard] Parameter to control verbosity of terminal output. Ranges from 1 for low, 2 for medium (prints some more additional information), 3 for high (prints eigenvalues and fractional occupancies at the end of each self-consistent field iteration), and 4 for very high, which is only meant for code development purposes. VERBOSITY=0 is only used for unit testing and shouldn't be used by standard users. VERBOSITY=-1 ensures no outout is printed, which is useful when DFT-FE is used as a calculator inside a larger workflow where multiple parallel DFT-FE jobs might be running, for example when using ASE or generating training data for ML workflows.");

      prm.declare_entry(
        "KEEP SCRATCH FOLDER",
        "false",
        Patterns::Bool(),
        "[Advanced] If set to true this option does not delete the dftfeScratch folder when the dftfe object is destroyed. This is useful for debugging and code development. Default: false.");

      prm.enter_subsection("GPU");
      {
        prm.declare_entry("USE GPU",
                          "false",
                          Patterns::Bool(),
                          "[Standard] Use GPU for compute.");

        prm.declare_entry(
          "AUTO GPU BLOCK SIZES",
          "true",
          Patterns::Bool(),
          "[Advanced] Automatically sets total number of kohn-sham wave functions and eigensolver optimal block sizes for running on GPUs. If manual tuning is desired set this parameter to false and set the block sizes using the input parameters for the block sizes. Default: true.");

        prm.declare_entry(
          "FINE GRAINED GPU TIMINGS",
          "false",
          Patterns::Bool(),
          "[Developer] Print more fine grained GPU timing results. Default: false.");


        prm.declare_entry(
          "SUBSPACE ROT FULL CPU MEM",
          "true",
          Patterns::Bool(),
          "[Developer] Option to use full NxN memory on CPU in subspace rotation and when mixed precision optimization is not being used. This reduces the number of MPI\_Allreduce communication calls. Default: true.");

        prm.declare_entry(
          "USE GPUDIRECT MPI ALL REDUCE",
          "false",
          Patterns::Bool(),
          "[Adavanced] Use GPUDIRECT MPI\_Allreduce. This route will only work if DFT-FE is compiled with NVIDIA NCCL library. Also note that one MPI rank per GPU can be used when using this option. Default: false.");

        prm.declare_entry(
          "USE ELPA GPU KERNEL",
          "false",
          Patterns::Bool(),
          "[Advanced] If DFT-FE is linked to ELPA eigensolver library configured to run on GPUs, this parameter toggles the use of ELPA GPU kernels for dense symmetric matrix diagonalization calls in DFT-FE. ELPA version>=2020.11.001 is required for this feature. Default: false.");

        prm.declare_entry(
          "GPU MEM OPT MODE",
          "true",
          Patterns::Bool(),
          "[Adavanced] Uses algorithms which have lower peak memory on GPUs but with a marginal performance degradation. Recommended when using more than 100k degrees of freedom per GPU. Default: true.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Postprocessing");
      {
        prm.declare_entry(
          "WRITE WFC",
          "false",
          Patterns::Bool(),
          "[Standard] Writes DFT ground state wavefunction solution fields (FEM mesh nodal values) to wfcOutput.vtu file for visualization purposes. The wavefunction solution fields in wfcOutput.vtu are named wfc\_s\_k\_i in case of spin-polarized calculations and wfc\_k\_i otherwise, where s denotes the spin index (0 or 1), k denotes the k point index starting from 0, and i denotes the Kohn-Sham wavefunction index starting from 0. In the case of geometry optimization, the wavefunctions corresponding to the last ground-state solve are written.  Default: false.");

        prm.declare_entry(
          "WRITE DENSITY",
          "false",
          Patterns::Bool(),
          "[Standard] Writes DFT ground state electron-density solution fields (FEM mesh nodal values) to densityOutput.vtu file for visualization purposes. The electron-density solution field in densityOutput.vtu is named density. In case of spin-polarized calculation, two additional solution fields- density\_0 and density\_1 are also written where 0 and 1 denote the spin indices. In the case of geometry optimization, the electron-density corresponding to the last ground-state solve is written. Default: false.");

        prm.declare_entry(
          "WRITE DENSITY OF STATES",
          "false",
          Patterns::Bool(),
          "[Standard] Computes density of states using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'dosData.out' containing two columns with first column indicating the energy in eV and second column indicating the density of states");

        prm.declare_entry(
          "WRITE LOCAL DENSITY OF STATES",
          "false",
          Patterns::Bool(),
          "[Standard] Computes local density of states on each atom using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'ldosData.out' containing NUMATOM+1 columns with first column indicating the energy in eV and all other NUMATOM columns indicating local density of states for each of the NUMATOM atoms.");

        prm.declare_entry(
          "WRITE PROJECTED DENSITY OF STATES",
          "false",
          Patterns::Bool(),
          "[Standard] Computes projected density of states on each atom using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'pdosData\_x' with x denoting atomID. This file contains columns with first column indicating the energy in eV and all other columns indicating projected density of states corresponding to single atom wavefunctions.");

        prm.declare_entry(
          "READ ATOMIC WFC PDOS FROM PSP FILE",
          "false",
          Patterns::Bool(),
          "[Standard] Read atomic wavefunctons from the pseudopotential file for computing projected density of states. When set to false atomic wavefunctions from the internal database are read, which correspond to sg15 ONCV pseudopotentials.");

        prm.declare_entry(
          "WRITE LOCALIZATION LENGTHS",
          "false",
          Patterns::Bool(),
          "[Standard] Computes localization lengths of all wavefunctions which is defined as the deviation around the mean position of a given wavefunction. Outputs a file name 'localizationLengths.out' containing 2 columns with first column indicating the wavefunction index and second column indicating localization length of the corresponding wavefunction.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Parallelization");
      {
        prm.declare_entry(
          "NPKPT",
          "1",
          Patterns::Integer(1),
          "[Standard] Number of groups of MPI tasks across which the work load of the irreducible k-points is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPKPT must be less than or equal to the number of irreducible k-points.");

        prm.declare_entry(
          "NPBAND",
          "1",
          Patterns::Integer(1),
          "[Standard] Number of groups of MPI tasks across which the work load of the bands is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPBAND must be less than or equal to NUMBER OF KOHN-SHAM WAVEFUNCTIONS.");

        prm.declare_entry(
          "MPI ALLREDUCE BLOCK SIZE",
          "100.0",
          Patterns::Double(0),
          "[Advanced] Block message size in MB used to break a single MPI\_Allreduce call on wavefunction vectors data into multiple MPI\_Allreduce calls. This is useful on certain architectures which take advantage of High Bandwidth Memory to improve efficiency of MPI operations. This variable is relevant only if NPBAND>1. Default value is 100.0 MB.");

        prm.declare_entry(
          "BAND PARAL OPT",
          "true",
          Patterns::Bool(),
          "[Standard] Uses a more optimal route for band parallelization but at the cost of extra wavefunctions memory.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Checkpointing and Restart");
      {
        prm.declare_entry(
          "CHK TYPE",
          "0",
          Patterns::Integer(0, 3),
          "[Standard] Checkpoint type, 0 (do not create any checkpoint), 1 (create checkpoint for geometry optimization restart if either ION OPT or CELL OPT is set to true. Currently, checkpointing and restart framework does not work if both ION OPT and CELL OPT are set to true simultaneously- the code will throw an error if attempted.), 2 (create checkpoint for scf restart using the electron-density field. Currently, this option cannot be used if geometry optimization is being performed. The code will throw an error if this option is used in conjunction with geometry optimization.)");

        prm.declare_entry(
          "RESTART FROM CHK",
          "false",
          Patterns::Bool(),
          "[Standard] Boolean parameter specifying if the current job reads from a checkpoint. The nature of the restart corresponds to the CHK TYPE parameter. Hence, the checkpoint being read must have been created using the CHK TYPE parameter before using this option. Further, for CHK TYPE=2 same number of MPI tasks must be used as used to create the checkpoint files. RESTART FROM CHK is always false for CHK TYPE 0.");

        prm.declare_entry(
          "RESTART SP FROM NO SP",
          "false",
          Patterns::Bool(),
          "[Standard] Enables ground-state solve for SPIN POLARIZED case reading the SPIN UNPOLARIZED density from the checkpoint files, and use the START MAGNETIZATION to compute the spin up and spin down densities. This option is only valid for CHK TYPE=2 and RESTART FROM CHK=true. Default false..");

        prm.declare_entry(
          "RESTART MD FROM CHK",
          "false",
          Patterns::Bool(),
          "[Developer] Boolean parameter specifying if the current job reads from a MD checkpoint (in development).");
      }
      prm.leave_subsection();

      prm.enter_subsection("Geometry");
      {
        prm.declare_entry(
          "ATOMIC COORDINATES FILE",
          "",
          Patterns::Anything(),
          "[Standard] Atomic-coordinates input file name. For fully non-periodic domain give Cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic domain give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.");

        prm.declare_entry(
          "ATOMIC DISP COORDINATES FILE",
          "",
          Patterns::Anything(),
          "[Standard] Atomic displacement coordinates input file name. The FEM mesh is deformed using Gaussian functions attached to the atoms. File format (example for two atoms): delx1 dely1 delz1 (row1), delx2 dely2 delz2 (row2). The number of rows must be equal to NATOMS. Units in a.u.");


        prm.declare_entry(
          "NATOMS",
          "0",
          Patterns::Integer(0),
          "[Standard] Total number of atoms. This parameter requires a mandatory non-zero input which is equal to the number of rows in the file passed to ATOMIC COORDINATES FILE.");

        prm.declare_entry(
          "NATOM TYPES",
          "0",
          Patterns::Integer(0),
          "[Standard] Total number of atom types. This parameter requires a mandatory non-zero input which is equal to the number of unique atom types in the file passed to ATOMIC COORDINATES FILE.");

        prm.declare_entry(
          "DOMAIN VECTORS FILE",
          "",
          Patterns::Anything(),
          "[Standard] Domain vectors input file name. Domain vectors are the vectors bounding the three edges of the 3D parallelepiped computational domain. File format: v1x v1y v1z (row1), v2y v2y v2z (row2), v3z v3y v3z (row3). Units: a.u. CAUTION: please ensure that the domain vectors form a right-handed coordinate system i.e. dotProduct(crossProduct(v1,v2),v3)>0. Domain vectors are the typical lattice vectors in a fully periodic calculation.");

        prm.enter_subsection("Optimization");
        {
          prm.declare_entry(
            "ION FORCE",
            "false",
            Patterns::Bool(),
            "[Standard] Boolean parameter specifying if atomic forces are to be computed. Automatically set to true if ION OPT is true.");

          prm.declare_entry(
            "NON SELF CONSISTENT FORCE",
            "false",
            Patterns::Bool(),
            "[Developer] Boolean parameter specifying whether to include the force contributions arising out of non self-consistency in the Kohn-Sham ground-state calculation. Currently non self-consistent force computation is still in experimental phase. The default option is false.");

          prm.declare_entry(
            "ION OPT",
            "false",
            Patterns::Bool(),
            "[Standard] Boolean parameter specifying if atomic forces are to be relaxed.");

          prm.declare_entry(
            "ION OPT SOLVER",
            "CGPRP",
            Patterns::Selection("CGDESCENT|LBFGS|CGPRP"),
            "[Standard] Method for Ion relaxation solver. CGPRP (Nonlinear conjugate gradient with Secant and Polak-Ribiere approach) is the default");

          prm.declare_entry(
            "MAX LINE SEARCH ITER",
            "5",
            Patterns::Integer(1, 100),
            "[Standard] Sets the maximum number of line search iterations in the case of CGPRP. Default is 5.");

          prm.declare_entry(
            "FORCE TOL",
            "1e-4",
            Patterns::Double(0, 1.0),
            "[Standard] Sets the tolerance on the maximum force (in a.u.) on an atom during atomic relaxation, when the atoms are considered to be relaxed.");

          prm.declare_entry(
            "ION RELAX FLAGS FILE",
            "",
            Patterns::Anything(),
            "[Standard] File specifying the permission flags (1-free to move, 0-fixed) and external forces for the 3-coordinate directions and for all atoms. File format (example for two atoms with atom 1 fixed and atom 2 free and 0.01 Ha/Bohr force acting on atom 2): 0 0 0 0.0 0.0 0.0(row1), 1 1 1 0.0 0.0 0.01(row2). External forces are optional.");

          prm.declare_entry(
            "CELL STRESS",
            "false",
            Patterns::Bool(),
            "[Standard] Boolean parameter specifying if cell stress needs to be computed. Automatically set to true if CELL OPT is true.");

          prm.declare_entry(
            "CELL OPT",
            "false",
            Patterns::Bool(),
            "[Standard] Boolean parameter specifying if cell needs to be relaxed to achieve zero stress");

          prm.declare_entry(
            "STRESS TOL",
            "1e-6",
            Patterns::Double(0, 1.0),
            "[Standard] Sets the tolerance of the cell stress (in a.u.) during cell-relaxation.");

          prm.declare_entry(
            "CELL CONSTRAINT TYPE",
            "12",
            Patterns::Integer(1, 13),
            "[Standard] Cell relaxation constraint type, 1 (isotropic shape-fixed volume optimization), 2 (volume-fixed shape optimization), 3 (relax along domain vector component v1x), 4 (relax along domain vector component v2y), 5 (relax along domain vector component v3z), 6 (relax along domain vector components v2y and v3z), 7 (relax along domain vector components v1x and v3z), 8 (relax along domain vector components v1x and v2y), 9 (relax along domain vector components v1x, v2y and v3z), 10 (2D - relax along x and y components), 11(2D- relax only x and y components with inplane area fixed), 12(relax all domain vector components), 13 automatically decides the constraints based on boundary conditions. CAUTION: A majority of these options only make sense in an orthorhombic cell geometry.");

          prm.declare_entry(
            "REUSE WFC",
            "true",
            Patterns::Bool(),
            "[Standard] Reuse previous ground-state wavefunctions during geometry optimization. Default setting is true.");

          prm.declare_entry(
            "REUSE DENSITY",
            "0",
            Patterns::Integer(0, 2),
            "[Standard] Parameter controlling the reuse of ground-state density during geometry optimization. The options are 0 (reinitialize density based on superposition of atomic densities), 1 (reuse ground-state density of previous relaxation step), and 2 (subtract superposition of atomic densities from the previous step's ground-state density and add superposition of atomic densities from the new atomic positions. Option 2 is not enabled for spin-polarized case. Default setting is 0.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Boundary conditions");
      {
        prm.declare_entry(
          "SELF POTENTIAL RADIUS",
          "0.0",
          Patterns::Double(0.0, 50),
          "[Advanced] The radius (in a.u) of the ball around an atom in which self-potential of the associated nuclear charge is solved. For the default value of 0.0, the radius value is automatically determined to accommodate the largest radius possible for the given finite element mesh. The default approach works for most problems.");

        prm.declare_entry(
          "PERIODIC1",
          "false",
          Patterns::Bool(),
          "[Standard] Periodicity along the first domain bounding vector.");

        prm.declare_entry(
          "PERIODIC2",
          "false",
          Patterns::Bool(),
          "[Standard] Periodicity along the second domain bounding vector.");

        prm.declare_entry(
          "PERIODIC3",
          "false",
          Patterns::Bool(),
          "[Standard] Periodicity along the third domain bounding vector.");

        prm.declare_entry(
          "POINT WISE DIRICHLET CONSTRAINT",
          "false",
          Patterns::Bool(),
          "[Developer] Flag to set point wise dirichlet constraints to eliminate null-space associated with the discretized Poisson operator subject to periodic BCs.");

        prm.declare_entry(
          "CONSTRAINTS PARALLEL CHECK",
          "false",
          Patterns::Bool(),
          "[Developer] Check for consistency of constraints in parallel.");

        prm.declare_entry(
          "CONSTRAINTS FROM SERIAL DOFHANDLER",
          "false",
          Patterns::Bool(),
          "[Developer] Create constraints from serial dofHandler.");

        prm.declare_entry(
          "SMEARED NUCLEAR CHARGES",
          "true",
          Patterns::Bool(),
          "[Developer] Nuclear charges are smeared for solving electrostatic fields. Default is true for pseudopotential calculations and false for all-electron calculations.");

        prm.declare_entry(
          "FLOATING NUCLEAR CHARGES",
          "true",
          Patterns::Bool(),
          "[Developer] Nuclear charges are allowed to float independent of the FEM mesh nodal positions. Only allowed for pseudopotential calculations. Internally set to false for all-electron calculations.");
      }
      prm.leave_subsection();


      prm.enter_subsection("Finite element mesh parameters");
      {
        prm.declare_entry(
          "POLYNOMIAL ORDER",
          "6",
          Patterns::Integer(1, 12),
          "[Standard] The degree of the finite-element interpolating polynomial in the Kohn-Sham Hamitonian except the electrostatics. Default value is 6 which is good choice for most pseudopotential calculations. POLYNOMIAL ORDER= 4 or 5 is usually a good choice for all-electron problems.");

        prm.declare_entry(
          "POLYNOMIAL ORDER ELECTROSTATICS",
          "0",
          Patterns::Integer(0, 24),
          "[Standard] The degree of the finite-element interpolating polynomial for the electrostatics part of the Kohn-Sham Hamiltonian. It is automatically set to POLYNOMIAL ORDER if POLYNOMIAL ORDER ELECTROSTATICS set to default value of zero.");

        prm.enter_subsection("Auto mesh generation parameters");
        {
          prm.declare_entry(
            "BASE MESH SIZE",
            "0.0",
            Patterns::Double(0, 20),
            "[Advanced] Mesh size of the base mesh on which refinement is performed. For the default value of 0.0, a heuristically determined base mesh size is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.");

          prm.declare_entry(
            "ATOM BALL RADIUS",
            "0.0",
            Patterns::Double(0, 20),
            "[Standard] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AROUND ATOM and coarse-grained in the region outside the enclosing balls. For the default value of 0.0, a heuristically determined value is used, which is good enough for most cases but can be a bit conservative choice for fully non-periodic and semi-periodic problems as well as all-electron problems. To improve the computational efficiency user may experiment with values of ATOM BALL RADIUS ranging between 3.0 to 6.0 for pseudopotential problems, and ranging between 1.0 to 2.5 for all-electron problems.  Units: a.u.");

          prm.declare_entry(
            "INNER ATOM BALL RADIUS",
            "0.0",
            Patterns::Double(0, 20),
            "[Advanced] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AT ATOM. Standard users do not need to tune this parameter. Units: a.u.");


          prm.declare_entry(
            "MESH SIZE AROUND ATOM",
            "1.0",
            Patterns::Double(0.0001, 10),
            "[Standard] Mesh size in a ball of radius ATOM BALL RADIUS around every atom. For pseudopotential calculations, the value ranges between 0.8 to 2.5 depending on the cutoff energy for the pseudopotential. For all-electron calculations, a value of around 0.5 would be a good starting choice. In most cases, MESH SIZE AROUND ATOM is the only parameter to be tuned to achieve the desired accuracy in energy and forces with respect to the mesh refinement. Units: a.u.");

          prm.declare_entry(
            "MESH SIZE AT ATOM",
            "0.0",
            Patterns::Double(0.0, 10),
            "[Advanced] Mesh size of the finite elements in the immediate vicinity of the atom. For the default value of 0.0, a heuristically determined MESH SIZE AT ATOM is used for all-electron calculations. For pseudopotential calculations, the default value of 0.0, sets the MESH SIZE AT ATOM to be the same value as MESH SIZE AROUND ATOM. Standard users do not need to tune this parameter. Units: a.u.");

          prm.declare_entry(
            "MESH ADAPTION",
            "false",
            Patterns::Bool(),
            "[Developer] Generates adaptive mesh based on a-posteriori mesh adaption strategy using single atom wavefunctions before computing the ground-state. Default: false.");

          prm.declare_entry(
            "AUTO ADAPT BASE MESH SIZE",
            "true",
            Patterns::Bool(),
            "[Developer] Automatically adapt the BASE MESH SIZE such that subdivisions of that during refinement leads closest to the desired MESH SIZE AROUND ATOM. Default: true.");


          prm.declare_entry(
            "TOP FRAC",
            "0.1",
            Patterns::Double(0.0, 1),
            "[Developer] Top fraction of elements to be refined.");

          prm.declare_entry("NUM LEVELS",
                            "10",
                            Patterns::Integer(0, 30),
                            "[Developer] Number of times to be refined.");

          prm.declare_entry(
            "TOLERANCE FOR MESH ADAPTION",
            "1",
            Patterns::Double(0.0, 1),
            "[Developer] Tolerance criteria used for stopping the multi-level mesh adaption done apriori using single atom wavefunctions. This is used as Kinetic energy change between two successive iterations");

          prm.declare_entry(
            "ERROR ESTIMATE WAVEFUNCTIONS",
            "5",
            Patterns::Integer(0),
            "[Developer] Number of wavefunctions to be used for error estimation.");

          prm.declare_entry(
            "GAUSSIAN CONSTANT FORCE GENERATOR",
            "0.75",
            Patterns::Double(0.0),
            "[Developer] Force computation generator gaussian constant. Also used for mesh movement. Gamma(r)= exp(-(r/gaussianConstant);(gaussianOrder)).");

          prm.declare_entry(
            "GAUSSIAN ORDER FORCE GENERATOR",
            "4.0",
            Patterns::Double(0.0),
            "[Developer] Force computation generator gaussian order. Also used for mesh movement. Gamma(r)= exp(-(r/gaussianConstant);(gaussianOrder)).");

          prm.declare_entry(
            "GAUSSIAN ORDER MOVE MESH TO ATOMS",
            "4.0",
            Patterns::Double(0.0),
            "[Developer] Move mesh to atoms gaussian order. Gamma(r)= exp(-(r/gaussianConstant);(gaussianOrder)).");

          prm.declare_entry(
            "USE FLAT TOP GENERATOR",
            "false",
            Patterns::Bool(),
            "[Developer] Use a composite generator flat top and Gaussian generator for mesh movement and configurational force computation.");

          prm.declare_entry(
            "USE MESH SIZES FROM ATOM LOCATIONS FILE",
            "false",
            Patterns::Bool(),
            "[Developer] Use mesh sizes from atom locations file.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Brillouin zone k point sampling options");
      {
        prm.enter_subsection("Monkhorst-Pack (MP) grid generation");
        {
          prm.declare_entry(
            "SAMPLING POINTS 1",
            "1",
            Patterns::Integer(1, 1000),
            "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 1.");

          prm.declare_entry(
            "SAMPLING POINTS 2",
            "1",
            Patterns::Integer(1, 1000),
            "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 2.");

          prm.declare_entry(
            "SAMPLING POINTS 3",
            "1",
            Patterns::Integer(1, 1000),
            "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 3.");

          prm.declare_entry(
            "SAMPLING SHIFT 1",
            "0",
            Patterns::Integer(0, 1),
            "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 1.");

          prm.declare_entry(
            "SAMPLING SHIFT 2",
            "0",
            Patterns::Integer(0, 1),
            "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 2.");

          prm.declare_entry(
            "SAMPLING SHIFT 3",
            "0",
            Patterns::Integer(0, 1),
            "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 3.");
        }
        prm.leave_subsection();

        prm.declare_entry(
          "kPOINT RULE FILE",
          "",
          Patterns::Anything(),
          "[Developer] File providing list of k points on which eigen values are to be computed from converged KS Hamiltonian. The first three columns specify the crystal coordinates of the k points. The fourth column provides weights of the corresponding points, which is currently not used. The eigen values are written on an output file bands.out");

        prm.declare_entry(
          "USE GROUP SYMMETRY",
          "false",
          Patterns::Bool(),
          "[Standard] Flag to control the use of point group symmetries. Currently this feature cannot be used if ION FORCE or CELL STRESS input parameters are set to true.");

        prm.declare_entry(
          "USE TIME REVERSAL SYMMETRY",
          "false",
          Patterns::Bool(),
          "[Standard] Flag to control the use of time reversal symmetry.");
      }
      prm.leave_subsection();

      prm.enter_subsection("DFT functional parameters");
      {
        prm.declare_entry(
          "PSEUDOPOTENTIAL CALCULATION",
          "true",
          Patterns::Bool(),
          "[Standard] Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed. For all-electron DFT calculation set to false.");

        prm.declare_entry(
          "PSEUDO TESTS FLAG",
          "false",
          Patterns::Bool(),
          "[Developer] Boolean parameter specifying the explicit path of pseudopotential upf format files used for ctests");

        prm.declare_entry(
          "PSEUDOPOTENTIAL FILE NAMES LIST",
          "",
          Patterns::Anything(),
          "[Standard] Pseudopotential file. This file contains the list of pseudopotential file names in UPF format corresponding to the atoms involved in the calculations. UPF version 2.0 or greater and norm-conserving pseudopotentials(ONCV and Troullier Martins) in UPF format are only accepted. File format (example for two atoms Mg(z=12), Al(z=13)): 12 filename1.upf(row1), 13 filename2.upf (row2). Important Note: ONCV pseudopotentials data base in UPF format can be downloaded from http://www.quantum-simulation.org/potentials/sg15\_oncv or http://www.pseudo-dojo.org/.  Troullier-Martins pseudopotentials in UPF format can be downloaded from http://www.quantum-espresso.org/pseudopotentials/fhi-pp-from-abinit-web-site.");

        prm.declare_entry(
          "EXCHANGE CORRELATION TYPE",
          "1",
          Patterns::Integer(1, 5),
          "[Standard] Parameter specifying the type of exchange-correlation to be used: 1(LDA: Perdew Zunger Ceperley Alder correlation with Slater Exchange[PRB. 23, 5048 (1981)]), 2(LDA: Perdew-Wang 92 functional with Slater Exchange [PRB. 45, 13244 (1992)]), 3(LDA: Vosko, Wilk \\& Nusair with Slater Exchange[Can. J. Phys. 58, 1200 (1980)]), 4(GGA: Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)], 5(RPBE: B. Hammer, L. B. Hansen, and J. K. Nørskov, Phys. Rev. B 59, 7413 (1999)).");

        prm.declare_entry(
          "SPIN POLARIZATION",
          "0",
          Patterns::Integer(0, 1),
          "[Standard] Spin polarization: 0 for no spin polarization and 1 for collinear spin polarization calculation. Default option is 0.");

        prm.declare_entry(
          "NUMBER OF IMAGES",
          "1",
          Patterns::Integer(1, 50),
          "[Standard] NUMBER OF IMAGES:Default option is 1. When NEB is triggered this controls the total number of images along the MEP including the end points");



        prm.declare_entry(
          "START MAGNETIZATION",
          "0.0",
          Patterns::Double(-0.5, 0.5),
          "[Standard] Starting magnetization to be used for spin-polarized DFT calculations (must be between -0.5 and +0.5). Corresponding magnetization per simulation domain will be (2 x START MAGNETIZATION x Number of electrons) a.u. ");

        prm.declare_entry(
          "PSP CUTOFF IMAGE CHARGES",
          "15.0",
          Patterns::Double(),
          "[Standard] Distance from the domain till which periodic images will be considered for the local part of the pseudopotential. Units in a.u. ");
      }
      prm.leave_subsection();


      prm.enter_subsection("SCF parameters");
      {
        prm.declare_entry(
          "TEMPERATURE",
          "500.0",
          Patterns::Double(1e-5),
          "[Standard] Fermi-Dirac smearing temperature (in Kelvin).");

        prm.declare_entry(
          "MAXIMUM ITERATIONS",
          "200",
          Patterns::Integer(1, 1000),
          "[Standard] Maximum number of iterations to be allowed for SCF convergence");

        prm.declare_entry(
          "TOLERANCE",
          "1e-05",
          Patterns::Double(1e-12, 1.0),
          "[Standard] SCF iterations stopping tolerance in terms of $L_2$ norm of the electron-density difference between two successive iterations. The default tolerance of is set to a tight value of 1e-5 for accurate ionic forces and cell stresses keeping structural optimization and molecular dynamics in mind. A tolerance of 1e-4 would be accurate enough for calculations without structural optimization and dynamics. CAUTION: A tolerance close to 1e-7 or lower can deteriorate the SCF convergence due to the round-off error accumulation.");

        prm.declare_entry(
          "MIXING HISTORY",
          "50",
          Patterns::Integer(1, 1000),
          "[Standard] Number of SCF iteration history to be considered for density mixing schemes. For metallic systems, a mixing history larger than the default value provides better scf convergence.");

        prm.declare_entry(
          "MIXING PARAMETER",
          "0.2",
          Patterns::Double(0.0, 1.0),
          "[Standard] Mixing parameter to be used in density mixing schemes. Default: 0.2.");

        prm.declare_entry(
          "KERKER MIXING PARAMETER",
          "0.05",
          Patterns::Double(0.0, 1000.0),
          "[Standard] Mixing parameter to be used in Kerker mixing scheme which usually represents Thomas Fermi wavevector (k\_{TF}**2).");

        prm.declare_entry(
          "MIXING METHOD",
          "ANDERSON",
          Patterns::Selection("BROYDEN|ANDERSON|ANDERSON_WITH_KERKER"),
          "[Standard] Method for density mixing. ANDERSON is the default option.");


        prm.declare_entry(
          "CONSTRAINT MAGNETIZATION",
          "false",
          Patterns::Bool(),
          "[Standard] Boolean parameter specifying whether to keep the starting magnetization fixed through the SCF iterations. Default is FALSE");

        prm.declare_entry(
          "STARTING WFC",
          "RANDOM",
          Patterns::Selection("ATOMIC|RANDOM"),
          "[Standard] Sets the type of the starting Kohn-Sham wavefunctions guess: Atomic(Superposition of single atom atomic orbitals. Atom types for which atomic orbitals are not available, random wavefunctions are taken. Currently, atomic orbitals data is not available for all atoms.), Random(The starting guess for all wavefunctions are taken to be random). Default: RANDOM.");

        prm.declare_entry(
          "COMPUTE ENERGY EACH ITER",
          "false",
          Patterns::Bool(),
          "[Advanced] Boolean parameter specifying whether to compute the total energy at the end of every SCF. Setting it to false can lead to some computational time savings. Default value is false but is internally set to true if VERBOSITY==5");

        prm.enter_subsection("Eigen-solver parameters");
        {
          prm.declare_entry(
            "NUMBER OF KOHN-SHAM WAVEFUNCTIONS",
            "0",
            Patterns::Integer(0),
            "[Standard] Number of Kohn-Sham wavefunctions to be computed. For spin-polarized calculations, this parameter denotes the number of Kohn-Sham wavefunctions to be computed for each spin. A recommended value for this parameter is to set it to N/2+Nb where N is the number of electrons. Use Nb to be 5-10 percent of N/2 for insulators and for metals use Nb to be 10-20 percent of N/2. If 5-20 percent of N/2 is less than 10 wavefunctions, set Nb to be atleast 10. Default value of 0 automatically sets the number of Kohn-Sham wavefunctions close to 20 percent more than N/2. CAUTION: use more states when using higher electronic temperature.");

          prm.declare_entry(
            "SPECTRUM SPLIT CORE EIGENSTATES",
            "0",
            Patterns::Integer(0),
            "[Advanced] Number of lowest Kohn-Sham eigenstates which should not be included in the Rayleigh-Ritz diagonalization.  In other words, only the eigenvalues and eigenvectors corresponding to the higher eigenstates (Number of Kohn-Sham wavefunctions minus the specified core eigenstates) are computed in the diagonalization of the projected Hamiltonian. This value is usually chosen to be the sum of the number of core eigenstates for each atom type multiplied by number of atoms of that type. This setting is recommended for large systems (greater than 5000 electrons). Default value is 0 i.e., no core eigenstates are excluded from the Rayleigh-Ritz projection step.");

          prm.declare_entry(
            "SPECTRUM SPLIT STARTING SCF ITER",
            "0",
            Patterns::Integer(0),
            "[Advanced] SCF iteration no beyond which spectrum splitting based can be used.");

          prm.declare_entry(
            "CHEBYSHEV POLYNOMIAL DEGREE",
            "0",
            Patterns::Integer(0, 2000),
            "[Advanced] Chebyshev polynomial degree to be employed for the Chebyshev filtering subspace iteration procedure to dampen the unwanted spectrum of the Kohn-Sham Hamiltonian. If set to 0, a default value depending on the upper bound of the eigen-spectrum is used. See Phani Motamarri et.al., J. Comp. Phys. 253, 308-343 (2013).");

          prm.declare_entry(
            "CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF",
            "1.34",
            Patterns::Double(0, 2000),
            "[Advanced] Chebyshev polynomial degree first scf scaling factor. Only activated for pseudopotential calculations.");


          prm.declare_entry(
            "CHEBYSHEV FILTER TOLERANCE",
            "5e-02",
            Patterns::Double(1e-10),
            "[Advanced] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure. Default value is sufficient for most purposes");

          prm.declare_entry(
            "ENABLE HAMILTONIAN TIMES VECTOR OPTIM",
            "true",
            Patterns::Bool(),
            "[Advanced] Turns on optimization for hamiltonian times vector multiplication. Operations involving data movement from global vector to finite-element cell level and vice versa are done by employing different data structures for interior nodes and surfaces nodes of a given cell and this allows reduction of memory access costs");


          prm.declare_entry(
            "ORTHOGONALIZATION TYPE",
            "Auto",
            Patterns::Selection("GS|CGS|Auto"),
            "[Advanced] Parameter specifying the type of orthogonalization to be used: GS(Gram-Schmidt Orthogonalization using SLEPc library) and CGS(Cholesky-Gram-Schmidt Orthogonalization). Auto is the default and recommended option, which chooses GS for all-electron case and CGS for pseudopotential case. On GPUs CGS is the only route currently implemented.");

          prm.declare_entry(
            "CHEBY WFC BLOCK SIZE",
            "400",
            Patterns::Integer(1),
            "[Advanced] Chebyshev filtering procedure involves the matrix-matrix multiplication where one matrix corresponds to the discretized Hamiltonian and the other matrix corresponds to the wavefunction matrix. The matrix-matrix multiplication is accomplished in a loop over the number of blocks of the wavefunction matrix to reduce the memory footprint of the code. This parameter specifies the block size of the wavefunction matrix to be used in the matrix-matrix multiplication. The optimum value is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting CHEBY WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/CHEBY WFC BLOCK SIZE equals an integer value. Default value is 400.");

          prm.declare_entry(
            "WFC BLOCK SIZE",
            "400",
            Patterns::Integer(1),
            "[Advanced]  This parameter specifies the block size of the wavefunction matrix to be used for memory optimization purposes in the orthogonalization, Rayleigh-Ritz, and density computation steps. The optimum block size is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/WFC BLOCK SIZE equals an integer value. Default value is 400.");

          prm.declare_entry(
            "SUBSPACE ROT DOFS BLOCK SIZE",
            "10000",
            Patterns::Integer(1),
            "[Developer] This block size is used for memory optimization purposes in subspace rotation step in Cholesky-Gram-Schmidt orthogonalization and Rayleigh-Ritz steps. Default value is 10000.");

          prm.declare_entry(
            "SCALAPACKPROCS",
            "0",
            Patterns::Integer(0, 300),
            "[Advanced] Uses a processor grid of SCALAPACKPROCS times SCALAPACKPROCS for parallel distribution of the subspace projected matrix in the Rayleigh-Ritz step and the overlap matrix in the Cholesky-Gram-Schmidt step. Default value is 0 for which a thumb rule is used (see http://netlib.org/scalapack/slug/node106.html). If ELPA is used, twice the value obtained from the thumb rule is used as ELPA scales much better than ScaLAPACK.");

          prm.declare_entry(
            "SCALAPACK BLOCK SIZE",
            "0",
            Patterns::Integer(0, 300),
            "[Advanced] ScaLAPACK process grid block size. Also sets the block size for ELPA if linked to ELPA. Default value of zero sets a heuristic block size. Note that if ELPA GPU KERNEL is set to true and ELPA is configured to run on GPUs, the SCALAPACK BLOCK SIZE is set to a power of 2.");

          prm.declare_entry(
            "USE ELPA",
            "true",
            Patterns::Bool(),
            "[Standard] Use ELPA instead of ScaLAPACK for diagonalization of subspace projected Hamiltonian and Cholesky-Gram-Schmidt orthogonalization.  Default setting is true.");

          prm.declare_entry(
            "USE MIXED PREC CGS SR",
            "false",
            Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in subspace rotation step of CGS orthogonalization, if ORTHOGONALIZATION TYPE is set to CGS. Default setting is false.");

          prm.declare_entry(
            "USE MIXED PREC CGS O",
            "false",
            Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in overlap matrix computation step of CGS orthogonalization, if ORTHOGONALIZATION TYPE is set to CGS. Default setting is false.");


          prm.declare_entry(
            "USE MIXED PREC XTHX SPECTRUM SPLIT",
            "false",
            Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in computing subspace projected Kohn-Sham Hamiltonian when SPECTRUM SPLIT CORE EIGENSTATES>0.  Default setting is false.");

          prm.declare_entry(
            "USE MIXED PREC RR_SR",
            "false",
            Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in Rayleigh-Ritz subspace rotation step. Default setting is false.");

          prm.declare_entry(
            "USE MIXED PREC CHEBY",
            "false",
            Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in Chebyshev filtering. Currently this option is only available for real executable and USE ELPA=true for which DFT-FE also has to be linked to ELPA library. Default setting is false.");

          prm.declare_entry(
            "OVERLAP COMPUTE COMMUN CHEBY",
            "true",
            Patterns::Bool(),
            "[Advanced] Overlap communication and computation in Chebyshev filtering. This option can only be activated for USE GPU=true. Default setting is true.");

          prm.declare_entry(
            "OVERLAP COMPUTE COMMUN ORTHO RR",
            "true",
            Patterns::Bool(),
            "[Advanced] Overlap communication and computation in orthogonalization and Rayleigh-Ritz. This option can only be activated for USE GPU=true. Default setting is true.");

          prm.declare_entry(
            "ALGO",
            "NORMAL",
            Patterns::Selection("NORMAL|FAST"),
            "[Standard] In the FAST mode, spectrum splitting technique is used in Rayleigh-Ritz step, and mixed precision arithmetic algorithms are used in Rayleigh-Ritz and Cholesky factorization based orthogonalization step. For spectrum splitting, 85 percent of the total number of wavefunctions are taken to be core states, which holds good for most systems including metallic systems assuming NUMBER OF KOHN-SHAM WAVEFUNCTIONS to be around 10 percent more than N/2. FAST setting is strongly recommended for large-scale (> 10k electrons) system sizes. Both NORMAL and FAST setting use Chebyshev filtered subspace iteration technique. If manual options for mixed precision and spectum splitting are being used, please use NORMAL setting for ALGO. Default setting is NORMAL.");


          prm.declare_entry(
            "REUSE LANCZOS UPPER BOUND",
            "false",
            Patterns::Bool(),
            "[Advanced] Reuse upper bound of unwanted spectrum computed in the first SCF iteration via Lanczos iterations. Default setting is false.");

          prm.declare_entry(
            "ALLOW MULTIPLE PASSES POST FIRST SCF",
            "true",
            Patterns::Bool(),
            "[Advanced] Allow multiple chebyshev filtering passes in the SCF iterations after the first one. Default setting is true.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Poisson problem parameters");
      {
        prm.declare_entry(
          "MAXIMUM ITERATIONS",
          "20000",
          Patterns::Integer(0, 20000),
          "[Advanced] Maximum number of iterations to be allowed for Poisson problem convergence.");

        prm.declare_entry(
          "TOLERANCE",
          "1e-10",
          Patterns::Double(0, 1.0),
          "[Advanced] Absolute tolerance on the residual as stopping criterion for Poisson problem convergence.");
      }
      prm.leave_subsection();


      prm.enter_subsection("Helmholtz problem parameters");
      {
        prm.declare_entry(
          "MAXIMUM ITERATIONS HELMHOLTZ",
          "10000",
          Patterns::Integer(0, 20000),
          "[Advanced] Maximum number of iterations to be allowed for Helmholtz problem convergence.");

        prm.declare_entry(
          "ABSOLUTE TOLERANCE HELMHOLTZ",
          "1e-10",
          Patterns::Double(0, 1.0),
          "[Advanced] Absolute tolerance on the residual as stopping criterion for Helmholtz problem convergence.");
      }
      prm.leave_subsection();


      prm.enter_subsection("Molecular Dynamics");
      {
        prm.declare_entry(
          "ATOMIC MASSES FILE",
          "",
          Patterns::Anything(),
          "[Standard] Input atomic masses file name. File format: atomicNumber1 atomicMass1 (row1), atomicNumber2 atomicMass2 (row2) and so on. Units: a.m.u.");
          prm.declare_entry(
            "EXTRAPOLATE DENSITY",
            "0",
            Patterns::Integer(0, 2),
            "[Standard] Parameter controlling the reuse of ground-state density during molecular dynamics. The options are 0 default setting where superposition of atomic densities is the initial rho, 1 (second order extrapolation of density), and 2 (extrapolation of split density and the atomic densities are added) Option 2 is not enabled for spin-polarized case. Default setting is 0.");



        prm.declare_entry(
          "BOMD",
          "false",
          Patterns::Bool(),
          "[Standard] Perform Born-Oppenheimer NVE molecular dynamics. Input parameters for molecular dynamics have to be modified directly in the code in the file md/molecularDynamics.cc.");

        prm.declare_entry(
          "XL BOMD",
          "false",
          Patterns::Bool(),
          "[Standard] Perform Extended Lagrangian Born-Oppenheimer NVE molecular dynamics. Currently not implemented for spin-polarization case.");

        prm.declare_entry(
          "CHEBY TOL XL BOMD",
          "1e-6",
          Patterns::Double(0.0),
          "[Standard] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure.");

        prm.declare_entry(
          "CHEBY TOL XL BOMD RANK UPDATES FD",
          "1e-7",
          Patterns::Double(0.0),
          "[Standard] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure.");

        prm.declare_entry(
          "CHEBY TOL XL BOMD RESTART",
          "1e-9",
          Patterns::Double(0.0),
          "[Standard] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure.");

        prm.declare_entry(
          "MAX JACOBIAN RATIO FACTOR",
          "1.5",
          Patterns::Double(0.9, 3.0),
          "[Developer] Maximum scaling factor for maximum jacobian ratio of FEM mesh when mesh is deformed.");

        prm.declare_entry(
          "STARTING TEMPERATURE",
          "300.0",
          Patterns::Double(0.0),
          "[Standard] Starting temperature in K for MD simulation.");

        prm.declare_entry(
          "THERMOSTAT TIME CONSTANT",
          "100",
          Patterns::Double(0.0),
          "[Standard] Ratio of Time constant of thermostat and MD timestep. ");

        prm.declare_entry(
          "TEMPERATURE CONTROLLER TYPE",
          "NO_CONTROL",
          Patterns::Selection("NO_CONTROL|RESCALE|NOSE_HOVER_CHAINS|CSVR"),
          "[Standard] Method of controlling temperature in the MD run. NO_CONTROL is the default option.");

        prm.declare_entry("TIME STEP",
                          "0.5",
                          Patterns::Double(0.0),
                          "[Standard] Time step in femtoseconds.");

        prm.declare_entry("NUMBER OF STEPS",
                          "1000",
                          Patterns::Integer(0, 200000),
                          "[Standard] Number of time steps.");
        prm.declare_entry("TRACKING ATOMIC NO",
                          "0",
                          Patterns::Integer(0, 200000),
                          "[Standard] The atom Number to track.");

        prm.declare_entry("MAX WALL TIME",
                          "2592000.0",
                          Patterns::Double(0.0),
                          "[Standard] Maximum Wall Time in seconds");

        prm.declare_entry(
          "DIRAC DELTA KERNEL SCALING CONSTANT XL BOMD",
          "0.1",
          Patterns::Double(0.0),
          "[Developer] Dirac delta scaling kernel constant for XL BOMD.");

        prm.declare_entry(
          "KERNEL RANK XL BOMD",
          "0",
          Patterns::Integer(0, 10),
          "[Standard] Maximum rank for low rank kernel update in XL BOMD.");

        prm.declare_entry("NUMBER DISSIPATION TERMS XL BOMD",
                          "8",
                          Patterns::Integer(1, 8),
                          "[Standard] Number of dissipation terms in XL BOMD.");

        prm.declare_entry(
          "NUMBER PASSES RR SKIPPED XL BOMD",
          "0",
          Patterns::Integer(0),
          "[Standard] Number of starting chebsyev filtering passes without Rayleigh Ritz in XL BOMD.");

        prm.declare_entry("USE ATOMIC RHO XL BOMD",
                          "true",
                          Patterns::Bool(),
                          "[Standard] Use atomic rho xl bomd.");

        prm.declare_entry(
          "DENSITY MATRIX PERTURBATION RANK UPDATES XL BOMD",
          "false",
          Patterns::Bool(),
          "[Standard] Use density matrix perturbation theory for rank updates.");

        prm.declare_entry(
          "XL BOMD KERNEL RANK UPDATE FD PARAMETER",
          "1e-2",
          Patterns::Double(0.0),
          "[Standard] Finite difference perturbation parameter.");
      }
      prm.leave_subsection();
    }
  } // namespace internalRunParameters



  void
  runParameters::parse_parameters(const std::string &parameter_file)
  {
    ParameterHandler prm;
    internalRunParameters::declare_parameters(prm);
    prm.parse_input(parameter_file);

    verbosity  = prm.get_integer("VERBOSITY");
    solvermode = prm.get("DFT-FE SOLVER MODE");
  }

} // namespace dftfe
