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
#include <dftParameters.h>
#include <fstream>
#include <iostream>

using namespace dealii;

namespace dftfe
{
  namespace internalDftParameters
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
        "VERBOSITY",
        "1",
        Patterns::Integer(-1, 5),
        "[Standard] Parameter to control verbosity of terminal output. Ranges from 1 for low, 2 for medium (prints some more additional information), 3 for high (prints eigenvalues and fractional occupancies at the end of each self-consistent field iteration), and 4 for very high, which is only meant for code development purposes. VERBOSITY=0 is only used for unit testing and shouldn't be used by standard users. VERBOSITY=-1 ensures no outout is printed, which is useful when DFT-FE is used as a calculator inside a larger workflow where multiple parallel DFT-FE jobs might be running, for example when using ASE or generating training data for ML workflows.");

      prm.declare_entry(
        "KEEP SCRATCH FOLDER",
        "false",
        Patterns::Bool(),
        "[Advanced] If set to true this option does not delete the dftfeScratch folder when the dftfe object is destroyed. This is useful for debugging and code development. Default: false.");

      prm.declare_entry(
        "SOLVER MODE",
        "GS",
        Patterns::Selection("GS|MD|NEB|OPT"),
        "[Standard] DFT-FE SOLVER MODE: If GS: performs GroundState calculations, ionic and cell relaxation. If MD: performs Molecular Dynamics Simulation. If NEB: performs a NEB calculation. If OPT: performs an ion and/or cell optimization calculation.");

      prm.declare_entry(
        "RESTART",
        "false",
        Patterns::Bool(),
        "[Standard] If set to true RESTART triggers restart checks and modifies the input files for coordinates, domain vectors. Default: false.");

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
          Patterns::Integer(0, 2),
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
            "OPTIMIZATION MODE",
            "ION",
            Patterns::Selection("ION|CELL|IONCELL"),
            "[Standard] Specifies whether the ionic coordinates and/or the lattice vectors are relaxed.");

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
            "ION OPT SOLVER",
            "CGPRP",
            Patterns::Selection("BFGS|LBFGS|CGPRP"),
            "[Standard] Method for Ion relaxation solver. CGPRP (Nonlinear conjugate gradient with Secant and Polak-Ribiere approach) is the default");

          prm.declare_entry(
            "CELL OPT SOLVER",
            "CGPRP",
            Patterns::Selection("BFGS|LBFGS|CGPRP"),
            "[Standard] Method for Cell relaxation solver. CGPRP (Nonlinear conjugate gradient with Secant and Polak-Ribiere approach) is the default");

          prm.declare_entry(
            "MAXIMUM OPTIMIZATION STEPS",
            "300",
            Patterns::Integer(1, 1000),
            "[Standard] Sets the maximum number of optimization steps to be performed.");

          prm.declare_entry(
            "MAXIMUM STAGGERED CYCLES",
            "300",
            Patterns::Integer(1, 1000),
            "[Standard] Sets the maximum number of staggered ion/cell optimization cycles to be performed.");

          prm.declare_entry(
            "MAXIMUM UPDATE STEP",
            "0.5",
            Patterns::Double(0, 5.0),
            "[Standard] Sets the maximum allowed step size (in a.u.) during ion/cell relaxation.");

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
            "1",
            Patterns::Integer(0, 2),
            "[Standard] Parameter controlling the reuse of ground-state density during geometry optimization. The options are 0 (reinitialize density based on superposition of atomic densities), 1 (reuse ground-state density of previous relaxation step), and 2 (subtract superposition of atomic densities from the previous step's ground-state density and add superposition of atomic densities from the new atomic positions. Option 2 is not enabled for spin-polarized case. Default setting is 0.");

          prm.declare_entry(
            "BFGS STEP METHOD",
            "QN",
            Patterns::Selection("QN|RFO"),
            "[Standard] Method for computing update step in BFGS. Quasi-Newton step (default) or Rational Function Step as described in JPC 1985, 89:52-57.");

          prm.declare_entry(
            "USE PRECONDITIONER",
            "false",
            Patterns::Bool(),
            "[Standard] Boolean parameter specifying if the preconditioner described by JCP 144, 164109 (2016) is to be used.");

          prm.declare_entry(
            "LBFGS HISTORY",
            "5",
            Patterns::Integer(1, 20),
            "[Standard] Number of previous steps to considered for the LBFGS update.");
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
        prm.enter_subsection("Dispersion Correction");
        {
          prm.declare_entry(
            "DISPERSION CORRECTION TYPE",
            "0",
            Patterns::Integer(0, 2),
            "[Standard] The dispersion correction type to be included post scf convergence: 0 for none, 1 for DFT-D3[JCP 132, 154104 (2010)][JCC 32, 1456 (2011)], 2 for DFT-D4 [JCP 147, 034112 (2017)][JCP 150, 154122 (2019)][PCCP 22, 8499-8512 (2020)].");
          prm.declare_entry(
            "D3 DAMPING TYPE",
            "3",
            Patterns::Integer(0, 4),
            "[Standard] The damping used for DFTD3, 0 for zero damping, 1 for BJ damping, 2 for D3M variant, 3 for BJM variant (default) and 4 for the OP variant.");
          prm.declare_entry(
            "D3 ATM",
            "false",
            Patterns::Bool(),
            "[Standard] Boolean parameter specifying whether or not the triple dipole correction in DFTD3 is to be included (ignored if DAMPING PARAMETERS FILE is specified).");
          prm.declare_entry(
            "D4 MBD",
            "false",
            Patterns::Bool(),
            "[Standard] Boolean parameter specifying whether or not the MBD correction in DFTD4 is to be included (ignored if DAMPING PARAMETERS FILE is specified).");
          prm.declare_entry(
            "DAMPING PARAMETERS FILE",
            "",
            Patterns::Anything(),
            "[Advanced] Name of the file containing custom damping parameters, for ZERO damping 6 parameters are expected (s6, s8, s9, sr6, sr8, alpha), for BJ anf BJM damping 6 parameters are expected (s6, s8, s9, a1, a2, alpha), for ZEROM damping 7 parameters are expected (s6, s8, s9, sr6, sr8, alpha, beta) and for optimized power damping 7 parameters are expected (s6, s8, s9, a1, a2, alpha, beta).");
          prm.declare_entry(
            "TWO BODY CUTOFF",
            "94.8683298050514",
            Patterns::Double(0.0),
            "[Advanced] Cutoff in a.u. for computing 2 body interactions terms in D3 correction");
          prm.declare_entry(
            "THREE BODY CUTOFF",
            "40.0",
            Patterns::Double(0.0),
            "[Advanced] Cutoff in a.u. for computing 3 body interactions terms in D3 correction");
          prm.declare_entry(
            "CN CUTOFF",
            "40.0",
            Patterns::Double(0.0),
            "[Advanced] Cutoff in a.u. for computing coordination number in D3 correction");
        }
        prm.leave_subsection();
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
          "0.0",
          Patterns::Double(-1e-12, 1.0),
          "[Standard] Mixing parameter to be used in density mixing schemes. For default value of 0.0, it is heuristically set for different mixing schemes (0.2 for Anderson and Broyden, and 0.5 for Kerker and LRJI.");

        prm.declare_entry(
          "KERKER MIXING PARAMETER",
          "0.05",
          Patterns::Double(0.0, 1000.0),
          "[Standard] Mixing parameter to be used in Kerker mixing scheme which usually represents Thomas Fermi wavevector (k\_{TF}**2).");

        prm.declare_entry(
          "MIXING METHOD",
          "ANDERSON",
          Patterns::Selection(
            "BROYDEN|ANDERSON|ANDERSON_WITH_KERKER|LOW_RANK_JACINV_PRECOND"),
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


        prm.enter_subsection("LOW RANK JACINV PRECOND");
        {
          prm.declare_entry(
            "METHOD SUB TYPE",
            "ADAPTIVE",
            Patterns::Selection("ADAPTIVE|ACCUMULATED_ADAPTIVE"),
            "[Advanced] Method subtype for LOW_RANK_JACINV_PRECOND.");

          prm.declare_entry(
            "STARTING NORM LARGE DAMPING",
            "2.0",
            Patterns::Double(0.0, 10.0),
            "[Advanced] L2 norm electron density difference below which damping parameter is set to SCF parameters::MIXING PARAMETER, otherwise set to 0.1.");


          prm.declare_entry(
            "ADAPTIVE RANK REL TOL",
            "0.3",
            Patterns::Double(0.0, 1.0),
            "[Standard] Tolerance criteria for rank updates. 0.4 is a more efficient choice when using ACCUMULATED_ADAPTIVE method.");

          prm.declare_entry(
            "ADAPTIVE RANK REL TOL REACCUM FACTOR",
            "2.0",
            Patterns::Double(0.0, 100.0),
            "[Advanced] For METHOD SUB TYPE=ACCUMULATED_ADAPTIVE.");

          prm.declare_entry(
            "POISSON SOLVER ABS TOL",
            "1e-6",
            Patterns::Double(0.0),
            "[Advanced] Absolute poisson solver tolerance for electrostatic potential response computation.");

          prm.declare_entry(
            "USE SINGLE PREC DENSITY RESPONSE",
            "false",
            Patterns::Bool(),
            "[Advanced] Turns on single precision optimization in density response computation.");

          prm.declare_entry(
            "ESTIMATE JAC CONDITION NO",
            "false",
            Patterns::Bool(),
            "[Advanced] Estimate condition number of the Jacobian at the final SCF iteration step using a low rank approximation with ADAPTIVE RANK REL TOL=1.0e-5.");
        }
        prm.leave_subsection();

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
            "0.0",
            Patterns::Double(-1.0e-12),
            "[Advanced] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure. For default value of 0.0, we heuristically set the value between 1e-3 and 5e-2 depending on the MIXING METHOD used.");

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
          "BOMD",
          "false",
          Patterns::Bool(),
          "[Standard] Perform Born-Oppenheimer NVE molecular dynamics. Input parameters for molecular dynamics have to be modified directly in the code in the file md/molecularDynamics.cc.");

        prm.declare_entry(
          "EXTRAPOLATE DENSITY",
          "0",
          Patterns::Integer(0, 2),
          "[Standard] Parameter controlling the reuse of ground-state density during molecular dynamics. The options are 0 default setting where superposition of atomic densities is the initial rho, 1 (second order extrapolation of density), and 2 (extrapolation of split density and the atomic densities are added) Option 2 is not enabled for spin-polarized case. Default setting is 0.");

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
      }
      prm.leave_subsection();
    }
  } // namespace internalDftParameters

  dftParameters::dftParameters()
  {
    finiteElementPolynomialOrder               = 1;
    finiteElementPolynomialOrderElectrostatics = 1;
    n_refinement_steps                         = 1;
    numberEigenValues                          = 1;
    xc_id                                      = 1;
    spinPolarized                              = 0;
    nkx                                        = 1;
    nky                                        = 1;
    nkz                                        = 1;
    offsetFlagX                                = 0;
    offsetFlagY                                = 0;
    offsetFlagZ                                = 0;
    chebyshevOrder                             = 1;
    numPass                                    = 1;
    numSCFIterations                           = 1;
    maxLinearSolverIterations                  = 1;
    mixingHistory                              = 1;
    npool                                      = 1;
    maxLinearSolverIterationsHelmholtz         = 1;

    radiusAtomBall                      = 0.0;
    mixingParameter                     = 0.5;
    absLinearSolverTolerance            = 1e-10;
    selfConsistentSolverTolerance       = 1e-10;
    TVal                                = 500;
    start_magnetization                 = 0.0;
    absLinearSolverToleranceHelmholtz   = 1e-10;
    chebyshevTolerance                  = 1e-02;
    mixingMethod                        = "";
    optimizationMode                    = "";
    ionOptSolver                        = "";
    cellOptSolver                       = "";

    isPseudopotential           = false;
    periodicX                   = false;
    periodicY                   = false;
    periodicZ                   = false;
    useSymm                     = false;
    timeReversal                = false;
    pseudoTestsFlag             = false;
    constraintMagnetization     = false;
    writeDosFile                = false;
    writeLdosFile               = false;
    writePdosFile               = false;
    writeLocalizationLengths    = false;
    std::string coordinatesFile = "";
    domainBoundingVectorsFile   = "";
    kPointDataFile              = "";
    ionRelaxFlagsFile           = "";
    orthogType                  = "";
    algoType                    = "";
    pseudoPotentialFile         = "";

    std::string coordinatesGaussianDispFile = "";

    outerAtomBallRadius            = 2.5;
    innerAtomBallRadius            = 0.0;
    meshSizeOuterDomain            = 10.0;
    meshSizeInnerBall              = 1.0;
    meshSizeOuterBall              = 1.0;
    numLevels                      = 1;
    numberWaveFunctionsForEstimate = 5;
    topfrac                        = 0.1;
    kerkerParameter                = 0.05;

    isIonOpt               = false;
    isCellOpt              = false;
    isIonForce             = false;
    isCellStress           = false;
    isBOMD                 = false;
    nonSelfConsistentForce = false;
    forceRelaxTol          = 1e-4; // Hartree/Bohr
    stressRelaxTol         = 1e-6; // Hartree/Bohr^3
    toleranceKinetic       = 1e-03;
    cellConstraintType     = 12; // all cell components to be relaxed

    verbosity                 = 0;
    keepScratchFolder         = false;
    chkType                   = 0;
    restartSpinFromNoSpin     = false;
    restartFromChk            = false;
    reproducible_output       = false;
    electrostaticsHRefinement = false;
    meshAdaption              = false;
    pinnedNodeForPBC          = true;
    HXOptimFlag               = false;

    startingWFCType                                = "";
    writeWfcSolutionFields                         = false;
    writeDensitySolutionFields                     = false;
    wfcBlockSize                                   = 400;
    chebyWfcBlockSize                              = 400;
    subspaceRotDofsBlockSize                       = 2000;
    nbandGrps                                      = 1;
    computeEnergyEverySCF                          = true;
    scalapackParalProcs                            = 0;
    scalapackBlockSize                             = 50;
    natoms                                         = 0;
    natomTypes                                     = 0;
    numCoreWfcRR                                   = 0;
    reuseWfcGeoOpt                                 = false;
    reuseDensityGeoOpt                             = 0;
    mpiAllReduceMessageBlockSizeMB                 = 2.0;
    useMixedPrecCGS_SR                             = false;
    useMixedPrecCGS_O                              = false;
    useMixedPrecXTHXSpectrumSplit                  = false;
    useMixedPrecSubspaceRotRR                      = false;
    spectrumSplitStartingScfIter                   = 1;
    useELPA                                        = false;
    constraintsParallelCheck                       = true;
    createConstraintsFromSerialDofhandler          = true;
    bandParalOpt                                   = true;
    autoAdaptBaseMeshSize                          = true;
    readWfcForPdosPspFile                          = false;
    useGPU                                         = false;
    gpuFineGrainedTimings                          = false;
    allowFullCPUMemSubspaceRot                     = true;
    useMixedPrecCheby                              = false;
    overlapComputeCommunCheby                      = false;
    overlapComputeCommunOrthoRR                    = false;
    autoGPUBlockSizes                              = true;
    maxJacobianRatioFactorForMD                    = 1.5;
    reuseDensityMD                                 = 0;
    timeStepBOMD                                   = 0.5;
    numberStepsBOMD                                = 1000;
    gaussianConstantForce                          = 0.75;
    gaussianOrderForce                             = 4.0;
    gaussianOrderMoveMeshToAtoms                   = 4.0;
    useFlatTopGenerator                            = false;
    diracDeltaKernelScalingConstant                = 0.1;
    useMeshSizesFromAtomsFile                      = false;
    chebyshevFilterPolyDegreeFirstScfScalingFactor = 1.34;
    useDensityMatrixPerturbationRankUpdates        = false;
    smearedNuclearCharges                          = false;
    floatingNuclearCharges                         = false;
    nonLinearCoreCorrection                        = false;
    maxLineSearchIterCGPRP                         = 5;
    atomicMassesFile                               = "";
    useGPUDirectAllReduce                          = false;
    pspCutoffImageCharges                          = 15.0;
    reuseLanczosUpperBoundFromFirstCall            = false;
    allowMultipleFilteringPassesAfterFirstScf      = true;
    useELPAGPUKernel                               = false;
    xcFamilyType                                   = "";
    gpuMemOptMode                                  = false;
    // New Paramters for moleculardyynamics class
    startingTempBOMD           = 300;
    thermostatTimeConstantBOMD = 100;
    MaxWallTime                = 2592000.0;
    tempControllerTypeBOMD     = "";
    MDTrack                    = 0;


    // New paramter for selecting mode and NEB parameters
    TotalImages = 1;


    dc_dispersioncorrectiontype = 0;
    dc_d3dampingtype            = 2;
    dc_d3ATM                    = false;
    dc_d4MBD                    = false;
    dc_dampingParameterFilename = "";
    dc_d3cutoff2                = 94.8683298050514;
    dc_d3cutoff3                = 40.0;
    dc_d3cutoffCN               = 40.0;

    /** parameters for LRJI preconditioner **/
    startingNormLRJILargeDamping  = 2.0;
    adaptiveRankRelTolLRJI        = 0.3;
    std::string methodSubTypeLRJI = "";
    factorAdapAccumClearLRJI      = 2.0;
    absPoissonSolverToleranceLRJI = 1.0e-6;
    singlePrecLRJI                = false;
    estimateJacCondNoFinalSCFIter = false;
    /*****************************************/
    bfgsStepMethod     = "QN";
    usePreconditioner  = false;
    lbfgsNumPastSteps  = 5;
    maxOptIter         = 300;
    maxStaggeredCycles = 100;
    maxUpdateStep      = 0.5;
  }


  void
  dftParameters::parse_parameters(const std::string &parameter_file,
                                  const MPI_Comm &   mpi_comm_parent,
                                  const bool         printParams)
  {
    ParameterHandler prm;
    internalDftParameters::declare_parameters(prm);
    prm.parse_input(parameter_file);

    verbosity                 = prm.get_integer("VERBOSITY");
    reproducible_output       = prm.get_bool("REPRODUCIBLE OUTPUT");
    keepScratchFolder         = prm.get_bool("KEEP SCRATCH FOLDER");
    electrostaticsHRefinement = prm.get_bool("H REFINED ELECTROSTATICS");

    prm.enter_subsection("GPU");
    {
      useGPU                     = prm.get_bool("USE GPU");
      gpuFineGrainedTimings      = prm.get_bool("FINE GRAINED GPU TIMINGS");
      allowFullCPUMemSubspaceRot = prm.get_bool("SUBSPACE ROT FULL CPU MEM");
      autoGPUBlockSizes          = prm.get_bool("AUTO GPU BLOCK SIZES");
      useGPUDirectAllReduce      = prm.get_bool("USE GPUDIRECT MPI ALL REDUCE");
      useELPAGPUKernel           = prm.get_bool("USE ELPA GPU KERNEL");
      gpuMemOptMode              = prm.get_bool("GPU MEM OPT MODE");
    }
    prm.leave_subsection();

    prm.enter_subsection("Postprocessing");
    {
      writeWfcSolutionFields     = prm.get_bool("WRITE WFC");
      writeDensitySolutionFields = prm.get_bool("WRITE DENSITY");
      writeDosFile               = prm.get_bool("WRITE DENSITY OF STATES");
      writeLdosFile            = prm.get_bool("WRITE LOCAL DENSITY OF STATES");
      writeLocalizationLengths = prm.get_bool("WRITE LOCALIZATION LENGTHS");
      readWfcForPdosPspFile =
        prm.get_bool("READ ATOMIC WFC PDOS FROM PSP FILE");
      writeLocalizationLengths = prm.get_bool("WRITE LOCALIZATION LENGTHS");
    }
    prm.leave_subsection();

    prm.enter_subsection("Parallelization");
    {
      npool        = prm.get_integer("NPKPT");
      nbandGrps    = prm.get_integer("NPBAND");
      bandParalOpt = prm.get_bool("BAND PARAL OPT");
      mpiAllReduceMessageBlockSizeMB =
        prm.get_double("MPI ALLREDUCE BLOCK SIZE");
    }
    prm.leave_subsection();

    prm.enter_subsection("Checkpointing and Restart");
    {
      chkType               = prm.get_integer("CHK TYPE");
      restartFromChk        = prm.get_bool("RESTART FROM CHK") && chkType != 0;
      restartSpinFromNoSpin = prm.get_bool("RESTART SP FROM NO SP");
    }
    prm.leave_subsection();

    prm.enter_subsection("Geometry");
    {
      natoms                      = prm.get_integer("NATOMS");
      natomTypes                  = prm.get_integer("NATOM TYPES");
      coordinatesFile             = prm.get("ATOMIC COORDINATES FILE");
      coordinatesGaussianDispFile = prm.get("ATOMIC DISP COORDINATES FILE");
      domainBoundingVectorsFile   = prm.get("DOMAIN VECTORS FILE");
      prm.enter_subsection("Optimization");
      {
        optimizationMode = prm.get("OPTIMIZATION MODE");
        isIonOpt = optimizationMode == "ION" || optimizationMode == "IONCELL";
        ionOptSolver           = prm.get("ION OPT SOLVER");
        cellOptSolver          = prm.get("CELL OPT SOLVER");
        maxLineSearchIterCGPRP = prm.get_integer("MAX LINE SEARCH ITER");
        nonSelfConsistentForce = prm.get_bool("NON SELF CONSISTENT FORCE");
        isIonForce             = isIonOpt || prm.get_bool("ION FORCE");
        forceRelaxTol          = prm.get_double("FORCE TOL");
        ionRelaxFlagsFile      = prm.get("ION RELAX FLAGS FILE");
        isCellOpt = optimizationMode == "CELL" || optimizationMode == "IONCELL";
        isCellStress       = isCellOpt || prm.get_bool("CELL STRESS");
        stressRelaxTol     = prm.get_double("STRESS TOL");
        cellConstraintType = prm.get_integer("CELL CONSTRAINT TYPE");
        reuseWfcGeoOpt     = prm.get_bool("REUSE WFC");
        reuseDensityGeoOpt = prm.get_integer("REUSE DENSITY");
        bfgsStepMethod     = prm.get("BFGS STEP METHOD");
        usePreconditioner  = prm.get_bool("USE PRECONDITIONER");
        lbfgsNumPastSteps  = prm.get_integer("LBFGS HISTORY");
        maxOptIter         = prm.get_integer("MAXIMUM OPTIMIZATION STEPS");
        maxStaggeredCycles = prm.get_integer("MAXIMUM STAGGERED CYCLES");
        maxUpdateStep      = prm.get_double("MAXIMUM UPDATE STEP");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
      radiusAtomBall           = prm.get_double("SELF POTENTIAL RADIUS");
      periodicX                = prm.get_bool("PERIODIC1");
      periodicY                = prm.get_bool("PERIODIC2");
      periodicZ                = prm.get_bool("PERIODIC3");
      constraintsParallelCheck = prm.get_bool("CONSTRAINTS PARALLEL CHECK");
      createConstraintsFromSerialDofhandler =
        prm.get_bool("CONSTRAINTS FROM SERIAL DOFHANDLER");
      pinnedNodeForPBC       = prm.get_bool("POINT WISE DIRICHLET CONSTRAINT");
      smearedNuclearCharges  = prm.get_bool("SMEARED NUCLEAR CHARGES");
      floatingNuclearCharges = prm.get_bool("FLOATING NUCLEAR CHARGES");
    }
    prm.leave_subsection();

    prm.enter_subsection("Finite element mesh parameters");
    {
      finiteElementPolynomialOrder = prm.get_integer("POLYNOMIAL ORDER");
      finiteElementPolynomialOrderElectrostatics =
        prm.get_integer("POLYNOMIAL ORDER ELECTROSTATICS") == 0 ?
          prm.get_integer("POLYNOMIAL ORDER") :
          prm.get_integer("POLYNOMIAL ORDER ELECTROSTATICS");
      prm.enter_subsection("Auto mesh generation parameters");
      {
        outerAtomBallRadius   = prm.get_double("ATOM BALL RADIUS");
        innerAtomBallRadius   = prm.get_double("INNER ATOM BALL RADIUS");
        meshSizeOuterDomain   = prm.get_double("BASE MESH SIZE");
        meshSizeInnerBall     = prm.get_double("MESH SIZE AT ATOM");
        meshSizeOuterBall     = prm.get_double("MESH SIZE AROUND ATOM");
        meshAdaption          = prm.get_bool("MESH ADAPTION");
        autoAdaptBaseMeshSize = prm.get_bool("AUTO ADAPT BASE MESH SIZE");
        topfrac               = prm.get_double("TOP FRAC");
        numLevels             = prm.get_double("NUM LEVELS");
        numberWaveFunctionsForEstimate =
          prm.get_integer("ERROR ESTIMATE WAVEFUNCTIONS");
        toleranceKinetic = prm.get_double("TOLERANCE FOR MESH ADAPTION");
        gaussianConstantForce =
          prm.get_double("GAUSSIAN CONSTANT FORCE GENERATOR");
        gaussianOrderForce = prm.get_double("GAUSSIAN ORDER FORCE GENERATOR");
        gaussianOrderMoveMeshToAtoms =
          prm.get_double("GAUSSIAN ORDER MOVE MESH TO ATOMS");
        useFlatTopGenerator = prm.get_bool("USE FLAT TOP GENERATOR");
        useMeshSizesFromAtomsFile =
          prm.get_bool("USE MESH SIZES FROM ATOM LOCATIONS FILE");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Brillouin zone k point sampling options");
    {
      prm.enter_subsection("Monkhorst-Pack (MP) grid generation");
      {
        nkx         = prm.get_integer("SAMPLING POINTS 1");
        nky         = prm.get_integer("SAMPLING POINTS 2");
        nkz         = prm.get_integer("SAMPLING POINTS 3");
        offsetFlagX = prm.get_integer("SAMPLING SHIFT 1");
        offsetFlagY = prm.get_integer("SAMPLING SHIFT 2");
        offsetFlagZ = prm.get_integer("SAMPLING SHIFT 3");
      }
      prm.leave_subsection();

      useSymm        = prm.get_bool("USE GROUP SYMMETRY");
      timeReversal   = prm.get_bool("USE TIME REVERSAL SYMMETRY");
      kPointDataFile = prm.get("kPOINT RULE FILE");
    }
    prm.leave_subsection();

    prm.enter_subsection("DFT functional parameters");
    {
      prm.enter_subsection("Dispersion Correction");
      {
        dc_dispersioncorrectiontype =
          prm.get_integer("DISPERSION CORRECTION TYPE");
        dc_d3dampingtype            = prm.get_integer("D3 DAMPING TYPE");
        dc_d3ATM                    = prm.get_bool("D3 ATM");
        dc_d4MBD                    = prm.get_bool("D4 MBD");
        dc_dampingParameterFilename = prm.get("DAMPING PARAMETERS FILE");
        dc_d3cutoff2                = prm.get_double("TWO BODY CUTOFF");
        dc_d3cutoff3                = prm.get_double("THREE BODY CUTOFF");
        dc_d3cutoffCN               = prm.get_double("CN CUTOFF");
      }
      prm.leave_subsection();
      isPseudopotential     = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
      pseudoTestsFlag       = prm.get_bool("PSEUDO TESTS FLAG");
      pseudoPotentialFile   = prm.get("PSEUDOPOTENTIAL FILE NAMES LIST");
      xc_id                 = prm.get_integer("EXCHANGE CORRELATION TYPE");
      spinPolarized         = prm.get_integer("SPIN POLARIZATION");
      start_magnetization   = prm.get_double("START MAGNETIZATION");
      pspCutoffImageCharges = prm.get_double("PSP CUTOFF IMAGE CHARGES");
      TotalImages           = prm.get_integer("NUMBER OF IMAGES");
    }
    prm.leave_subsection();

    prm.enter_subsection("SCF parameters");
    {
      TVal                          = prm.get_double("TEMPERATURE");
      numSCFIterations              = prm.get_integer("MAXIMUM ITERATIONS");
      selfConsistentSolverTolerance = prm.get_double("TOLERANCE");
      mixingHistory                 = prm.get_integer("MIXING HISTORY");
      mixingParameter               = prm.get_double("MIXING PARAMETER");
      kerkerParameter               = prm.get_double("KERKER MIXING PARAMETER");
      mixingMethod                  = prm.get("MIXING METHOD");
      constraintMagnetization       = prm.get_bool("CONSTRAINT MAGNETIZATION");
      startingWFCType               = prm.get("STARTING WFC");
      computeEnergyEverySCF         = prm.get_bool("COMPUTE ENERGY EACH ITER");

      prm.enter_subsection("LOW RANK JACINV PRECOND");
      {
        methodSubTypeLRJI = prm.get("METHOD SUB TYPE");
        startingNormLRJILargeDamping =
          prm.get_double("STARTING NORM LARGE DAMPING");
        adaptiveRankRelTolLRJI = prm.get_double("ADAPTIVE RANK REL TOL");
        factorAdapAccumClearLRJI =
          prm.get_double("ADAPTIVE RANK REL TOL REACCUM FACTOR");
        absPoissonSolverToleranceLRJI =
          prm.get_double("POISSON SOLVER ABS TOL");
        singlePrecLRJI = prm.get_bool("USE SINGLE PREC DENSITY RESPONSE");
        estimateJacCondNoFinalSCFIter =
          prm.get_bool("ESTIMATE JAC CONDITION NO");
      }
      prm.leave_subsection();

      prm.enter_subsection("Eigen-solver parameters");
      {
        numberEigenValues =
          prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
        numCoreWfcRR = prm.get_integer("SPECTRUM SPLIT CORE EIGENSTATES");
        spectrumSplitStartingScfIter =
          prm.get_integer("SPECTRUM SPLIT STARTING SCF ITER");
        chebyshevOrder = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");
        useELPA        = prm.get_bool("USE ELPA");
        HXOptimFlag    = prm.get_bool("ENABLE HAMILTONIAN TIMES VECTOR OPTIM");
        orthogType     = prm.get("ORTHOGONALIZATION TYPE");
        chebyshevTolerance = prm.get_double("CHEBYSHEV FILTER TOLERANCE");
        wfcBlockSize       = prm.get_integer("WFC BLOCK SIZE");
        chebyWfcBlockSize  = prm.get_integer("CHEBY WFC BLOCK SIZE");
        subspaceRotDofsBlockSize =
          prm.get_integer("SUBSPACE ROT DOFS BLOCK SIZE");
        scalapackParalProcs = prm.get_integer("SCALAPACKPROCS");
        scalapackBlockSize  = prm.get_integer("SCALAPACK BLOCK SIZE");
        useMixedPrecCGS_SR  = prm.get_bool("USE MIXED PREC CGS SR");
        useMixedPrecCGS_O   = prm.get_bool("USE MIXED PREC CGS O");
        useMixedPrecXTHXSpectrumSplit =
          prm.get_bool("USE MIXED PREC XTHX SPECTRUM SPLIT");
        useMixedPrecSubspaceRotRR = prm.get_bool("USE MIXED PREC RR_SR");
        useMixedPrecCheby         = prm.get_bool("USE MIXED PREC CHEBY");
        overlapComputeCommunCheby =
          prm.get_bool("OVERLAP COMPUTE COMMUN CHEBY");
        overlapComputeCommunOrthoRR =
          prm.get_bool("OVERLAP COMPUTE COMMUN ORTHO RR");
        algoType                                       = prm.get("ALGO");
        chebyshevFilterPolyDegreeFirstScfScalingFactor = prm.get_double(
          "CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF");
        reuseLanczosUpperBoundFromFirstCall =
          prm.get_bool("REUSE LANCZOS UPPER BOUND");
        ;
        allowMultipleFilteringPassesAfterFirstScf =
          prm.get_bool("ALLOW MULTIPLE PASSES POST FIRST SCF");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();


    prm.enter_subsection("Poisson problem parameters");
    {
      maxLinearSolverIterations = prm.get_integer("MAXIMUM ITERATIONS");
      absLinearSolverTolerance  = prm.get_double("TOLERANCE");
    }
    prm.leave_subsection();

    prm.enter_subsection("Helmholtz problem parameters");
    {
      maxLinearSolverIterationsHelmholtz =
        prm.get_integer("MAXIMUM ITERATIONS HELMHOLTZ");
      absLinearSolverToleranceHelmholtz =
        prm.get_double("ABSOLUTE TOLERANCE HELMHOLTZ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Molecular Dynamics");
    {
      atomicMassesFile            = prm.get("ATOMIC MASSES FILE");
      reuseDensityMD              = prm.get_integer("EXTRAPOLATE DENSITY");
      isBOMD                      = prm.get_bool("BOMD");
      maxJacobianRatioFactorForMD = prm.get_double("MAX JACOBIAN RATIO FACTOR");
      timeStepBOMD                = prm.get_double("TIME STEP");
      numberStepsBOMD             = prm.get_integer("NUMBER OF STEPS");
      MDTrack                     = prm.get_integer("TRACKING ATOMIC NO");
      startingTempBOMD            = prm.get_double("STARTING TEMPERATURE");
      thermostatTimeConstantBOMD  = prm.get_double("THERMOSTAT TIME CONSTANT");
      MaxWallTime                 = prm.get_double("MAX WALL TIME");



      tempControllerTypeBOMD = prm.get("TEMPERATURE CONTROLLER TYPE");
    }
    prm.leave_subsection();

    if ((restartFromChk == true) && (chkType == 1))
      {
        if (periodicX || periodicY || periodicZ)
          coordinatesFile = floatingNuclearCharges ?
                              "atomsFracCoordCurrent.chk" :
                              "atomsFracCoordAutomesh.chk";
        else
          coordinatesFile = floatingNuclearCharges ?
                              "atomsCartCoordCurrent.chk" :
                              "atomsCartCoordAutomesh.chk";

        domainBoundingVectorsFile = "domainBoundingVectorsCurrent.chk";

        if (!floatingNuclearCharges)
          coordinatesGaussianDispFile = "atomsGaussianDispCoord.chk";
      }

    check_parameters(mpi_comm_parent);

    const bool printParametersToFile = false;
    if (printParametersToFile &&
        Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
      {
        prm.print_parameters(std::cout, ParameterHandler::OutputStyle::LaTeX);
        exit(0);
      }

    if (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0 &&
        verbosity >= 1 && printParams)
      {
        prm.print_parameters(std::cout, ParameterHandler::ShortText);
      }

    //
    setAutoParameters(mpi_comm_parent);
    setXCFamilyType();
  }



  void
  dftParameters::check_parameters(const MPI_Comm &mpi_comm_parent) const
  {
    AssertThrow(
      !((periodicX || periodicY || periodicZ) &&
        (writeLdosFile || writePdosFile)),
      ExcMessage(
        "DFT-FE Error: LOCAL DENSITY OF STATES and PROJECTED DENSITY OF STATES are currently not implemented in the case of periodic and semi-periodic boundary conditions."));

    if (floatingNuclearCharges)
      AssertThrow(
        smearedNuclearCharges,
        ExcMessage(
          "DFT-FE Error: FLOATING NUCLEAR CHARGES can only be used if SMEARED NUCLEAR CHARGES is set to true."));

#ifdef USE_COMPLEX
    if (isIonForce || isCellStress)
      AssertThrow(
        !useSymm,
        ExcMessage(
          "DFT-FE Error: USE GROUP SYMMETRY must be set to false if either ION FORCE or CELL STRESS is set to true. This functionality will be added in a future release"));
#else
    AssertThrow(
      nkx == 1 && nky == 1 && nkz == 1 && offsetFlagX == 0 &&
        offsetFlagY == 0 && offsetFlagZ == 0,
      ExcMessage(
        "DFT-FE Error: Real executable cannot be used for non-zero k point."));
#endif
    AssertThrow(
      !(chkType == 2 && (isIonOpt || isCellOpt)),
      ExcMessage(
        "DFT-FE Error: CHK TYPE=2 cannot be used if geometry optimization is being performed."));

    AssertThrow(
      !(chkType == 1 && (isIonOpt && isCellOpt)),
      ExcMessage(
        "DFT-FE Error: CHK TYPE=1 cannot be used if both ION OPT and CELL OPT are set to true."));

    if (numberEigenValues != 0)
      AssertThrow(
        nbandGrps <= numberEigenValues,
        ExcMessage(
          "DFT-FE Error: NPBAND is greater than NUMBER OF KOHN-SHAM WAVEFUNCTIONS."));

    if (nonSelfConsistentForce)
      AssertThrow(
        false,
        ExcMessage(
          "DFT-FE Error: Implementation of this feature is not completed yet."));

    if (spinPolarized == 1 && mixingMethod == "ANDERSON_WITH_KERKER")
      AssertThrow(
        false,
        ExcMessage(
          "DFT-FE Error: Implementation of this feature is not completed yet."));
    if (spinPolarized == 1 && (reuseDensityMD >= 1 || reuseDensityGeoOpt == 2))
      AssertThrow(
        false,
        ExcMessage(
          "DFT-FE Error: Implementation of this feature is not completed yet."));

    AssertThrow(!coordinatesFile.empty(),
                ExcMessage("DFT-FE Error: ATOMIC COORDINATES FILE not given."));

    AssertThrow(!domainBoundingVectorsFile.empty(),
                ExcMessage("DFT-FE Error: DOMAIN VECTORS FILE not given."));

    if (isPseudopotential)
      AssertThrow(
        !pseudoPotentialFile.empty(),
        ExcMessage("DFT-FE Error: PSEUDOPOTENTIAL FILE NAMES LIST not given."));

    if (spinPolarized == 0)
      AssertThrow(
        !constraintMagnetization,
        ExcMessage(
          "DFT-FE Error: This is a SPIN UNPOLARIZED calculation. Can't have CONSTRAINT MAGNETIZATION ON."));

    if (spinPolarized == 1 && !constraintMagnetization)
      AssertThrow(
        std::abs(std::abs(start_magnetization) - 0.5) > 1e-6,
        ExcMessage(
          "DFT-FE Error: START MAGNETIZATION =+-0.5 only applicable in case of CONSTRAINT MAGNETIZATION set to ON."));

    if (verbosity >= 1 &&
        Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
      if (constraintMagnetization)
        std::cout
          << " WARNING: CONSTRAINT MAGNETIZATION is ON. A fixed occupation will be used no matter what temperature is provided at input"
          << std::endl;

    if (spinPolarized == 1 && mixingMethod == "LOW_RANK_JACINV_PRECOND")
      AssertThrow(
        !constraintMagnetization,
        ExcMessage(
          "DFT-FE Error: CONSTRAINT MAGNETIZATION for LRJI Preconditioner is not yet supported."));

    AssertThrow(
      natoms != 0,
      ExcMessage(
        "DFT-FE Error: Number of atoms not specified or given a value of zero, which is not allowed."));

    AssertThrow(
      natomTypes != 0,
      ExcMessage(
        "DFT-FE Error: Number of atom types not specified or given a value of zero, which is not allowed."));

    if (meshAdaption)
      AssertThrow(
        !(isIonOpt && isCellOpt),
        ExcMessage(
          "DFT-FE Error: Currently Atomic relaxation does not work with automatic mesh adaption scheme."));

    if (nbandGrps > 1)
      AssertThrow(
        wfcBlockSize == chebyWfcBlockSize,
        ExcMessage(
          "DFT-FE Error: WFC BLOCK SIZE and CHEBY WFC BLOCK SIZE must be same for band parallelization."));
  }


  void
  dftParameters::setAutoParameters(const MPI_Comm &mpi_comm_parent)
  {
    //
    // Automated choice of mesh related parameters
    //

    if (isBOMD)
      isIonForce = true;

    if (!isPseudopotential)
      {
        if (!reproducible_output)
          smearedNuclearCharges = false;
        floatingNuclearCharges = false;
      }

    if (meshSizeOuterDomain < 1.0e-6)
      if (periodicX || periodicY || periodicZ)
        meshSizeOuterDomain = 4.0;
      else
        meshSizeOuterDomain = 13.0;

    if (meshSizeInnerBall < 1.0e-6)
      if (isPseudopotential)
        meshSizeInnerBall = 10.0 * meshSizeOuterBall;
      else
        meshSizeInnerBall = 0.1 * meshSizeOuterBall;

    if (outerAtomBallRadius < 1.0e-6)
      {
        if (isPseudopotential)
          {
            if (!floatingNuclearCharges)
              outerAtomBallRadius = 2.5;
            else
              {
                if (!(periodicX || periodicY || periodicZ))
                  outerAtomBallRadius = 6.0;
                else
                  outerAtomBallRadius = 10.0;
              }
          }
        else
          outerAtomBallRadius = 2.0;
      }

    if (!(periodicX || periodicY || periodicZ) && !reproducible_output)
      {
        constraintsParallelCheck              = false;
        createConstraintsFromSerialDofhandler = false;
      }
    else if (reproducible_output)
      createConstraintsFromSerialDofhandler = true;

    if (reproducible_output)
      {
        gaussianOrderMoveMeshToAtoms = 4.0;
      }

    //
    // Automated choice of eigensolver parameters
    //
    if (isPseudopotential && orthogType == "Auto")
      {
        if (verbosity >= 1 &&
            Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=CGS for pseudopotential calculations "
            << std::endl;
        orthogType = "CGS";
      }
    else if (!isPseudopotential && orthogType == "Auto" && !useGPU)
      {
#ifdef USE_PETSC;
        if (verbosity >= 1 &&
            Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=GS for all-electron calculations as DFT-FE is linked to dealii with Petsc and Slepc"
            << std::endl;

        orthogType = "GS";
#else
        if (verbosity >= 1 &&
            Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=CGS for all-electron calculations as DFT-FE is not linked to dealii with Petsc and Slepc "
            << std::endl;

        orthogType = "CGS";
#endif
      }
    else if (orthogType == "GS" && !useGPU)
      {
#ifndef USE_PETSC;
        AssertThrow(
          orthogType != "GS",
          ExcMessage(
            "DFT-FE Error: Please use ORTHOGONALIZATION TYPE to be CGS/Auto as GS option is only available if DFT-FE is linked to dealii with Petsc and Slepc."));
#endif
      }
    else if (!isPseudopotential && orthogType == "Auto" && useGPU)
      {
        if (verbosity >= 1 &&
            Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=CGS for all-electron calculations on GPUs "
            << std::endl;
        orthogType = "CGS";
      }
    else if (orthogType == "GS" && useGPU)
      {
        AssertThrow(
          false,
          ExcMessage(
            "DFT-FE Error: GS is not implemented on GPUs. Use Auto option."));
      }


    if (algoType == "FAST")
      {
        useMixedPrecCGS_O                   = true;
        useMixedPrecCGS_SR                  = true;
        useMixedPrecXTHXSpectrumSplit       = true;
        useMixedPrecCheby                   = true;
        reuseLanczosUpperBoundFromFirstCall = true;
      }
#ifdef USE_COMPLEX
    HXOptimFlag = false;
#endif


#ifdef DFTFE_WITH_GPU
    if (!isPseudopotential && useGPU)
      {
        overlapComputeCommunCheby = false;
      }
#endif


#ifndef DFTFE_WITH_GPU
    useGPU           = false;
    useELPAGPUKernel = false;
#endif

    if (scalapackBlockSize == 0)
      {
        if (useELPAGPUKernel)
          scalapackBlockSize = 16;
        else
          scalapackBlockSize = 32;
      }

#ifndef DFTFE_WITH_NCCL
    useGPUDirectAllReduce = false;
#endif

    if (useMixedPrecCheby)
      AssertThrow(
        useELPA,
        ExcMessage(
          "DFT-FE Error: USE ELPA must be set to true for USE MIXED PREC CHEBY."));

    if (verbosity >= 5)
      computeEnergyEverySCF = true;

    if (std::fabs(chebyshevTolerance - 0.0) < 1.0e-12)
      {
        if (mixingMethod == "LOW_RANK_JACINV_PRECOND")
          chebyshevTolerance = 2.0e-3;
        else if (mixingMethod == "ANDERSON_WITH_KERKER")
          chebyshevTolerance = 1.0e-2;
        else
          chebyshevTolerance = 5.0e-2;
      }

    if (std::fabs(mixingParameter - 0.0) < 1.0e-12)
      {
        if (mixingMethod == "LOW_RANK_JACINV_PRECOND")
          mixingParameter = 0.5;
        else if (mixingMethod == "ANDERSON_WITH_KERKER")
          mixingParameter = 0.5;
        else
          mixingParameter = 0.2;
      }
  }

  void
  dftParameters::setXCFamilyType()
  {
    if (xc_id == 1)
      {
        xcFamilyType = "LDA";
      }
    else if (xc_id == 2)
      {
        xcFamilyType = "LDA";
      }
    else if (xc_id == 3)
      {
        xcFamilyType = "LDA";
      }
    else if (xc_id == 4)
      {
        xcFamilyType = "GGA";
      }
    else if (xc_id == 5)
      {
        xcFamilyType = "GGA";
      }
  }

} // namespace dftfe
