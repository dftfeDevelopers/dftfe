// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
#include <dftParameters.h>
#include <iostream>
#include <fstream>
#include <deal.II/base/data_out_base.h>

using namespace dealii;

namespace dftfe {

namespace dftParameters
{

  unsigned int finiteElementPolynomialOrder=1,n_refinement_steps=1,numberEigenValues=1,xc_id=1, spinPolarized=0, nkx=1,nky=1,nkz=1, offsetFlagX=0,offsetFlagY=0,offsetFlagZ=0;
  unsigned int chebyshevOrder=1,numPass=1, numSCFIterations=1,maxLinearSolverIterations=1, mixingHistory=1, npool=1;

  double radiusAtomBall=0.0, mixingParameter=0.5;
  double lowerEndWantedSpectrum=0.0,relLinearSolverTolerance=1e-10,selfConsistentSolverTolerance=1e-10,TVal=500, start_magnetization=0.0;
  double chebyshevTolerance = 1e-02;

  bool isPseudopotential=false,periodicX=false,periodicY=false,periodicZ=false, useSymm=false, timeReversal=false,pseudoTestsFlag=false;
  std::string meshFileName="",coordinatesFile="",domainBoundingVectorsFile="",kPointDataFile="", ionRelaxFlagsFile="",orthogType="",pseudoPotentialFile="";

  double outerAtomBallRadius=2.0, meshSizeOuterDomain=10.0;
  double meshSizeInnerBall=1.0, meshSizeOuterBall=1.0;

  bool isIonOpt=false, isCellOpt=false, isIonForce=false, isCellStress=false;
  bool nonSelfConsistentForce=false;
  double forceRelaxTol  = 1e-4;//Hartree/Bohr
  double stressRelaxTol = 1e-6;//Hartree/Bohr^3
  unsigned int cellConstraintType=12;// all cell components to be relaxed

  unsigned int verbosity=0; unsigned int chkType=0;
  bool restartFromChk=false;
  bool reproducible_output=false;

  unsigned int chebyshevBlockSize=512;
  std::string startingWFCType="";
  bool useBatchGEMM=false;
  bool writeWfcSolutionFields=false;
  bool writeDensitySolutionFields=false;
  unsigned int orthoRRWaveFuncBlockSize=200;
  unsigned int subspaceRotDofsBlockSize=2000;
  bool enableSwitchToGS=true;
  unsigned int nbandGrps=1;
  bool computeEnergyEverySCF=true;
  unsigned int scalapackParalProcs=0;
  unsigned int natoms=0;
  unsigned int natomTypes=0;
  double lowerBoundUnwantedFracUpper=0;

  void declare_parameters(ParameterHandler &prm)
  {
    prm.declare_entry("REPRODUCIBLE OUTPUT", "false",
                      Patterns::Bool(),
                      "[Developer] Limit output to that which is reprodicible, i.e. don't print timing or absolute paths. This parameter is only used for testing purposes.");

    prm.declare_entry("VERBOSITY", "1",
                      Patterns::Integer(0,4),
                      "[Standard] Parameter to control verbosity of terminal output. Ranging from 1 for low, 2 for medium (prints eigenvalues and fractional occupancies at the end of each ground-state solve), 3 for high (prints eigenvalues and fractional occupancies at the end of each self-consistent field iteration), and 4 for very high, which is only meant for code development purposes. VERBOSITY=0 is only used for unit testing and shouldn't be used by standard users.");

    prm.declare_entry("WRITE WFC", "false",
                      Patterns::Bool(),
                      "[Standard] Writes KSDFT ground state (last ground state solve in case of geometry optimization) wavefunction solution fields (FEM mesh nodal values) to wfcOutput.vtu file for visualization purposes. The wavefunction solution fields in wfcOutput.vtu are named wfc_s_k_i in case of spin-polarized calculations and wfc_k_i otherwise, where s denotes the spin index (0 or 1), k denotes the k point index starting from 0, and i denotes the Kohn-Sham wavefunction index starting from 0. Default: false.");

    prm.declare_entry("WRITE DENSITY", "false",
                      Patterns::Bool(),
                      "[Standard] Writes KSDFT ground state (last ground state solve in case of geometry optimization) electron-density solution fields (FEM mesh nodal values) to densityOutput.vtu file for visualization purposes. The electron-density solution field in densityOutput.vtu is named density. In case of spin-polarized calculation, two additional solution fields- density_0 and density_1 are also written where 0 and 1 denote the spin indices. Default: false.");

    prm.enter_subsection ("Parallelization");
    {
	prm.declare_entry("NPKPT", "1",
			  Patterns::Integer(1),
			  "[Standard] Number of pools of MPI processors across which the work load of the irreducible k-points is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPKPT must be less than or equal to the number of irreducible k-points.");

	prm.declare_entry("NPBAND", "1",
			   Patterns::Integer(1),
			   "[Standard] Number of pools of MPI processors across which the work load of the bands is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPBAND must be less than or equal to NUMBER OF KOHN-SHAM WAVEFUNCTIONS.");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Checkpointing and Restart");
    {
	prm.declare_entry("CHK TYPE", "0",
			   Patterns::Integer(0,2),
			   "[Standard] Checkpoint type, 0(dont create any checkpoint), 1(create checkpoint for geometry optimization restart if ION OPT or CELL OPT is set to true. Currently, checkpointing and restart framework doesn't work if both ION OPT and CELL OPT are set to true- the code will throw an error if attempted.), 2(create checkpoint for scf restart. Currently, this option cannot be used if geometry optimization is being performed. The code will throw an error if this option is used in conjunction with geometry optimization.)");

	prm.declare_entry("RESTART FROM CHK", "false",
			   Patterns::Bool(),
			   "[Standard] Boolean parameter specifying if the current job reads from a checkpoint. The nature of the restart corresponds to the CHK TYPE parameter. Hence, the checkpoint being read must have been created using the same value of the CHK TYPE parameter. RESTART FROM CHK is always false for CHK TYPE 0.");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Geometry");
    {
        prm.declare_entry("NATOMS", "0",
                        Patterns::Integer(0),
                       "[Standard] Total number of atoms. This parameter requires a mandatory non-zero input which is equal to the number of rows in the file passed to ATOMIC COORDINATES FILE.");

        prm.declare_entry("NATOM TYPES", "0",
                        Patterns::Integer(0),
                       "[Standard] Total number of atom types. This parameter requires a mandatory non-zero input which is equal to the number of unique atom types in the file passed to ATOMIC COORDINATES FILE.");

	prm.declare_entry("ATOMIC COORDINATES FILE", "",
			  Patterns::Anything(),
			  "[Standard] Atomic-coordinates input file name. For fully non-periodic domain give cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.");

	prm.declare_entry("DOMAIN VECTORS FILE", "",
			  Patterns::Anything(),
			  "[Standard] Domain vectors input file name. Domain vectors describe the edges of the 3D parallelepiped computational domain. File format: v1x v1y v1z (row1), v2x v2y v2z (row2), v3x v3y v3z (row3). Units: a.u. CAUTION: please ensure that the domain vectors form a right-handed coordinate system i.e. dotProduct(crossProduct(v1,v2),v3)>0. Domain vectors are the typical lattice vectors in a fully periodic calculation.");

	prm.enter_subsection ("Optimization");
	{

	    prm.declare_entry("ION FORCE", "false",
			      Patterns::Bool(),
			      "[Standard] Boolean parameter specifying if atomic forces are to be computed. Automatically set to true if ION OPT is true.");

	    prm.declare_entry("NON SELF CONSISTENT FORCE", "false",
			      Patterns::Bool(),
			      "[Developer] Boolean parameter specfiying whether to add the force terms arising due to the non self-consistency error. Currently non self-consistent force computation is still in developmental phase. The default option is false.");

	    prm.declare_entry("ION OPT", "false",
			      Patterns::Bool(),
			      "[Standard] Boolean parameter specifying if atomic forces are to be relaxed.");

	    prm.declare_entry("FORCE TOL", "1e-4",
			      Patterns::Double(0,1.0),
			      "[Standard] Sets the tolerance of the maximum force (in a.u.) on an ion when forces are considered to be relaxed.");

	    prm.declare_entry("ION RELAX FLAGS FILE", "",
			      Patterns::Anything(),
			      "[Standard] File specifying the atomic position update permission flags. 1- update 0- no update. File format (example for two atoms with atom 1 fixed and atom 2 free): 0 0 0 (row1), 1 1 1 (row2).");

	    prm.declare_entry("CELL STRESS", "false",
			      Patterns::Bool(),
			      "[Standard] Boolean parameter specifying if cell stress is to be computed. Automatically set to true if CELL OPT is true.");

	    prm.declare_entry("CELL OPT", "false",
			      Patterns::Bool(),
			      "[Standard] Boolean parameter specifying if cell stress is to be relaxed");

	    prm.declare_entry("STRESS TOL", "1e-6",
			      Patterns::Double(0,1.0),
			      "[Standard] Sets the tolerance of the cell stress (in a.u.) when cell stress is considered to be relaxed.");

	    prm.declare_entry("CELL CONSTRAINT TYPE", "12",
			      Patterns::Integer(1,13),
			      "[Standard] Cell relaxation constraint type, 1(isotropic shape-fixed volume optimization), 2(volume-fixed shape optimization), 3(relax only cell component v1x), 4(relax only cell component v2x), 5(relax only cell component v3x), 6(relax only cell components v2x and v3x), 7(relax only cell components v1x and v3x), 8(relax only cell components v1x and v2x), 9(volume optimization- relax only v1x, v2x and v3x), 10(2D- relax only x and y components relaxed), 11(2D- relax only x and y shape components- inplane area fixed), 12(relax all cell components), 13 automatically decides the constraints based on boundary conditions. CAUTION: A majority of these options only make sense in an orthorhombic cell geometry.");

	}
	prm.leave_subsection ();

    }
    prm.leave_subsection ();

    prm.enter_subsection ("Boundary conditions");
    {
        prm.declare_entry("SELF POTENTIAL RADIUS", "0.0",
                      Patterns::Double(0.0,10),
                      "[Developer] The radius (in a.u) of the ball around an atom on which self-potential of the associated nuclear charge is solved. For the default value of 0.0, the radius value is automatically determined to accomodate the largest radius possible for the given finite element mesh. The default approach works for most problems.");

	prm.declare_entry("PERIODIC1", "false",
			  Patterns::Bool(),
			  "[Standard] Periodicity along the first domain bounding vector.");

	prm.declare_entry("PERIODIC2", "false",
			  Patterns::Bool(),
			  "[Standard] Periodicity along the second domain bounding vector.");

	prm.declare_entry("PERIODIC3", "false",
			  Patterns::Bool(),
			  "[Standard] Periodicity along the third domain bounding vector.");
    }
    prm.leave_subsection ();


    prm.enter_subsection ("Finite element mesh parameters");
    {

      prm.declare_entry("POLYNOMIAL ORDER", "4",
                        Patterns::Integer(1,12),
                       "[Standard] The degree of the finite-element interpolating polynomial. Default value is 4. POLYNOMIAL ORDER= 4 or 5 is usually a good choice for most pseudopotential as well as all-electron problems.");

      prm.declare_entry("MESH FILE", "",
                       Patterns::Anything(),
                       "[Developer] External mesh file path. If nothing is given auto mesh generation is performed. The option is only for testing purposes.");

      prm.enter_subsection ("Auto mesh generation parameters");
      {

	prm.declare_entry("BASE MESH SIZE", "0.0",
			  Patterns::Double(0,20),
			  "[Developer] Mesh size of the base mesh on which refinement is performed. For the default value of 0.0, a heuristically determined base mesh size is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.");

	prm.declare_entry("ATOM BALL RADIUS","2.0",
			  Patterns::Double(0,3),
			  "[Developer] Radius of ball enclosing every atom inside which the mesh size is set close to MESH SIZE AROUND ATOM. The default value of 2.0 is good enough for most cases. On rare cases, where the nonlocal pseudopotential projectors have a compact supportbeyond 2.0, a slightly larger ATOM BALL RADIUS between 2.0 to 2.5 may be required. Standard users do not need to tune this parameter. Units: a.u.");

	prm.declare_entry("MESH SIZE AROUND ATOM", "1.0",
			  Patterns::Double(0.0001,10),
			  "[Standard] Mesh size in a ball of radius ATOM BALL RADIUS around every atom. For pseudopotential calculations, a value between 0.5 to 1.0 is usually a good choice. For all-electron calculations, a value between 0.1 to 0.3 would be a good starting choice. MESH SIZE AROUND ATOM is the only parameter standard users need to tune to achieve the desired accuracy in their results with respect to the mesh refinement. Units: a.u.");

	prm.declare_entry("MESH SIZE AT ATOM", "0.0",
			  Patterns::Double(0.0,10),
			  "[Developer] Mesh size of the finite elements in the immediate vicinity of the atom. For the default value of 0.0, a heuristically determined MESH SIZE AT ATOM is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.");

      }
      prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Brillouin zone k point sampling options");
    {
        prm.enter_subsection ("Monkhorst-Pack (MP) grid generation");
        {
	    prm.declare_entry("SAMPLING POINTS 1", "1",
			      Patterns::Integer(1,1000),
			      "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal latttice vector 1.");

	    prm.declare_entry("SAMPLING POINTS 2", "1",
			      Patterns::Integer(1,1000),
			      "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal latttice vector 2.");

	    prm.declare_entry("SAMPLING POINTS 3", "1",
			      Patterns::Integer(1,1000),
			      "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal latttice vector 3.");

	    prm.declare_entry("SAMPLING SHIFT 1", "0",
			      Patterns::Integer(0,1),
			      "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal latttice vector 1.");

	    prm.declare_entry("SAMPLING SHIFT 2", "0",
			      Patterns::Integer(0,1),
			      "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal latttice vector 2.");

	    prm.declare_entry("SAMPLING SHIFT 3", "0",
			      Patterns::Integer(0,1),
			      "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal latttice vector 3.");

	}
	prm.leave_subsection ();

	prm.declare_entry("kPOINT RULE FILE", "",
			  Patterns::Anything(),
			  "[Developer] File specifying the k-Point quadrature rule to sample Brillouin zone. CAUTION: This option is only used for postprocessing, for example band structure calculation. To set k point rule for DFT solve use the Monkhorst-Pack (MP) grid generation.");

	prm.declare_entry("USE GROUP SYMMETRY", "false",
			  Patterns::Bool(),
			  "[Standard] Flag to control whether to use point group symmetries. Currently this feature cannot be used if ION FORCE or CELL STRESS input parameters are set to true.");

	prm.declare_entry("USE TIME REVERSAL SYMMETRY", "false",
			  Patterns::Bool(),
			  "[Standard] Flag to control usage of time reversal symmetry.");

    }
    prm.leave_subsection ();

    prm.enter_subsection ("DFT functional parameters");
    {

	prm.declare_entry("PSEUDOPOTENTIAL CALCULATION", "true",
			  Patterns::Bool(),
			  "[Standard] Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed. For all-electron DFT calculation set to false.");

	prm.declare_entry("PSEUDO TESTS FLAG", "false",
			  Patterns::Bool(),
			  "[Developer] Boolean parameter specifying the explicit path of pseudopotential upf format files used for ctests");

	prm.declare_entry("PSEUDOPOTENTIAL FILE NAMES LIST", "",
			  Patterns::Anything(),
			  "[Standard] Pseudopotential file. This file contains the list of pseudopotential file names in UPF format corresponding to the atoms involved in the calculations. UPF version greater than 2.0 and norm-conserving pseudopotentials in UPF format are only accepted. File format (example for two atoms Mg(z=12), Al(z=13)): 12 filename1.upf(row1), 13 filename2.upf (row2)");

	prm.declare_entry("EXCHANGE CORRELATION TYPE", "1",
			  Patterns::Integer(1,4),
			  "[Standard] Parameter specifying the type of exchange-correlation to be used: 1(LDA: Perdew Zunger Ceperley Alder correlation with Slater Exchange[PRB. 23, 5048 (1981)]), 2(LDA: Perdew-Wang 92 functional with Slater Exchange [PRB. 45, 13244 (1992)]), 3(LDA: Vosko, Wilk \\& Nusair with Slater Exchange[Can. J. Phys. 58, 1200 (1980)]), 4(GGA: Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)]).");

	prm.declare_entry("SPIN POLARIZATION", "0",
			  Patterns::Integer(0,1),
			  "[Standard] Spin polarization: 0 for no spin polarization and 1 for spin polarization. Default option is 0.");

	prm.declare_entry("START MAGNETIZATION", "0.0",
			  Patterns::Double(-0.5,0.5),
			  "[Standard] Magnetization to start with (must be between -0.5 and +0.5). Corresponding magnetization per unit cell will be (2 x START MAGNETIZATION x Ne) a.u. , where Ne is the number of electrons in the unit cell ");
    }
    prm.leave_subsection ();


    prm.enter_subsection ("SCF parameters");
    {
	prm.declare_entry("TEMPERATURE", "500.0",
			  Patterns::Double(0.0),
			  "[Standard] Fermi-Dirac smearing temperature (in Kelvin).");

	prm.declare_entry("MAXIMUM ITERATIONS", "50",
			  Patterns::Integer(1,1000),
			  "[Standard] Maximum number of iterations to be allowed for SCF convergence");

	prm.declare_entry("TOLERANCE", "1e-06",
			  Patterns::Double(1e-12,1.0),
			  "[Standard] SCF iterations stopping tolerance in terms of L2 norm of the electron-density difference between two successive iterations. CAUTION: A tolerance close to 1e-7 or lower can detoriate the SCF convergence due to the round-off errors.");

	prm.declare_entry("ANDERSON SCHEME MIXING HISTORY", "10",
			  Patterns::Integer(1,1000),
			  "[Standard] Number of SCF iteration history to be considered for mixing the electron-density. For metallic systems, typically a mixing history larger than the default value provides better scf convergence.");

	prm.declare_entry("ANDERSON SCHEME MIXING PARAMETER", "0.5",
			  Patterns::Double(0.0,1.0),
			  "[Standard] Mixing parameter to be used in Anderson scheme.");

	prm.declare_entry("STARTING WFC","RANDOM",
			  Patterns::Selection("ATOMIC|RANDOM"),
			  "[Standard] Sets the type of the starting Kohn-Sham wavefunctions guess: Atomic(Superposition of single atom atomic orbitals. Wavefunctions for which atomic orbitals are not available, random wavefunctions are taken. Currently, atomic orbitals data is not available for all atoms.), Random(The starting guess for all wavefunctions are taken to be random). Default: RANDOM.");

	prm.declare_entry("COMPUTE ENERGY EACH ITER", "true",
			  Patterns::Bool(),
			  "[Developer] Boolean parameter specifying whether to compute the total energy at the end of every scf. Setting it to false can lead to some time savings.");

	prm.enter_subsection ("Eigen-solver parameters");
	{

	    prm.declare_entry("NUMBER OF KOHN-SHAM WAVEFUNCTIONS", "10",
			      Patterns::Integer(0),
			      "[Standard] Number of Kohn-Sham wavefunctions to be computed. For insulators use N/2+(10-20) and for metals use 20 percent more than N/2 (atleast 10 more). N is the total number of electrons. For spin-polarized calculations this parameter denotes the number of Kohn-Sham wavefunctions to be computed for each spin.");

	    prm.declare_entry("LOWER BOUND WANTED SPECTRUM", "-10.0",
			      Patterns::Double(),
			      "[Developer] The lower bound of the wanted eigen spectrum. It is only used for the first iteration of the Chebyshev filtered subspace iteration procedure. A rough estimate based on single atom eigen values can be used here. Default value is good enough for most problems.");

	    prm.declare_entry("CHEBYSHEV POLYNOMIAL DEGREE", "0",
			      Patterns::Integer(0,2000),
			      "[Developer] Chebyshev polynomial degree to be employed for the Chebyshev filtering subspace iteration procedure to dampen the unwanted spectrum of the Kohn-Sham Hamiltonian. If set to 0, a default value depending on the upper bound of the eigen-spectrum is used.");

	    prm.declare_entry("LOWER BOUND UNWANTED FRAC UPPER", "0",
			      Patterns::Double(0,1),
			      "[Developer] The value of the fraction of the upper bound of the unwanted spectrum, the lower bound of the unwanted spectrum will be set. Default value is 0.");

	    prm.declare_entry("CHEBYSHEV FILTER TOLERANCE","1e-02",
			      Patterns::Double(1e-10),
			      "[Developer] Parameter specifying the tolerance to which eigenvectors need to computed using chebyshev filtering approach.");

	    prm.declare_entry("CHEBYSHEV FILTER BLOCK SIZE", "400",
			       Patterns::Integer(1),
			       "[Developer] The maximum number of wavefunctions which are handled by one call to the Chebyshev filter. This is useful for optimization purposes. The optimum value is dependent on the computing architecture.");

	    prm.declare_entry("BATCH GEMM", "true",
			      Patterns::Bool(),
			      "[Developer] Boolean parameter specifying whether to use gemm batch blas routines to perform matrix-matrix multiplication operations with groups of matrices, processing a number of groups at once using threads instead of the standard serial route. CAUTION: gemm batch blas routines will only be activated if the CHEBYSHEV FILTER BLOCK SIZE is less than 1000, and intel mkl blas library linked with the dealii installation. Default option is true.");

	    prm.declare_entry("ORTHOGONALIZATION TYPE","LW",
			      Patterns::Selection("GS|LW|PGS"),
			      "[Standard] Parameter specifying the type of orthogonalization to be used: GS(Gram-Schmidt Orthogonalization using SLEPc library), LW(Lowden Orthogonalization using LAPACK, extension to ScaLAPACK not implemented yet), PGS(Pseudo-Gram-Schmidt Orthogonalization, if dealii library is compiled with ScaLAPACK, ScaLAPACK functions are used otherwise LAPACK functions are used). LW is the default option.");

	    prm.declare_entry("ENABLE SWITCH TO GS", "true",
			      Patterns::Bool(),
			      "[Developer] Controls automatic switching to Gram-Schimdt orthogonalization if Lowden Orthogonalization or Pseudo-Gram-Schimdt orthogonalization are unstable. Default option is true.");

	    prm.declare_entry("ORTHO RR WFC BLOCK SIZE", "200",
			       Patterns::Integer(1),
			       "[Developer] This block size is used for memory optimization purposes in the orthogonalization and Rayleigh-Ritz steps. This optimization is only activated if dealii library is compiled with ScaLAPACK. Default value is 200.");

	    prm.declare_entry("SUBSPACE ROT DOFS BLOCK SIZE", "2000",
			       Patterns::Integer(1),
			       "[Developer] This block size is used for memory optimization purposes in subspace rotation step in Pseudo-Gram-Schmidt orthogonalization and Rayleigh-Ritz steps. This optimization is only activated if dealii library is compiled with ScaLAPACK. Default value is 2000.");

	    prm.declare_entry("SCALAPACKPROCS", "0",
			      Patterns::Integer(0,300),
			      "[Developer] Uses a processor grid of SCALAPACKPROCS times SCALAPACKPROCS for parallel distribution of the subspace projected matrix in the Rayleigh-Ritz step and the overlap matrix in the Pseudo-Gram-Schmidt step. Default value is 0 for which a thumb rule is used (see http://netlib.org/scalapack/slug/node106.html\\#SECTION04511000000000000000). This parameter is only used if dealii library is compiled with ScaLAPACK.");
	}
	prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Poisson problem parameters");
    {
	prm.declare_entry("MAXIMUM ITERATIONS", "5000",
			  Patterns::Integer(0,20000),
			  "[Developer] Maximum number of iterations to be allowed for Poisson problem convergence.");

	prm.declare_entry("TOLERANCE", "1e-12",
			  Patterns::Double(0,1.0),
			  "[Developer] Relative tolerance as stopping criterion for Poisson problem convergence.");
    }
    prm.leave_subsection ();

  }

  void parse_parameters(ParameterHandler &prm)
  {
    dftParameters::verbosity                     = prm.get_integer("VERBOSITY");
    dftParameters::reproducible_output           = prm.get_bool("REPRODUCIBLE OUTPUT");
    dftParameters::writeWfcSolutionFields           = prm.get_bool("WRITE WFC");
    dftParameters::writeDensitySolutionFields           = prm.get_bool("WRITE DENSITY");

    prm.enter_subsection ("Parallelization");
    {
	dftParameters::npool             = prm.get_integer("NPKPT");
	dftParameters::nbandGrps         = prm.get_integer("NPBAND");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Checkpointing and Restart");
    {
	chkType=prm.get_integer("CHK TYPE");
	restartFromChk=prm.get_bool("RESTART FROM CHK") && chkType!=0;
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Geometry");
    {
        dftParameters::natoms                        = prm.get_integer("NATOMS");
        dftParameters::natomTypes                    = prm.get_integer("NATOM TYPES");
        dftParameters::coordinatesFile               = prm.get("ATOMIC COORDINATES FILE");
        dftParameters::domainBoundingVectorsFile     = prm.get("DOMAIN VECTORS FILE");
	prm.enter_subsection ("Optimization");
	{
	    dftParameters::isIonOpt                      = prm.get_bool("ION OPT");
	    dftParameters::nonSelfConsistentForce        = prm.get_bool("NON SELF CONSISTENT FORCE");
	    dftParameters::isIonForce                    = dftParameters::isIonOpt || prm.get_bool("ION FORCE");
	    dftParameters::forceRelaxTol                 = prm.get_double("FORCE TOL");
	    dftParameters::ionRelaxFlagsFile             = prm.get("ION RELAX FLAGS FILE");
	    dftParameters::isCellOpt                     = prm.get_bool("CELL OPT");
	    dftParameters::isCellStress                  = dftParameters::isCellOpt || prm.get_bool("CELL STRESS");
	    dftParameters::stressRelaxTol                = prm.get_double("STRESS TOL");
	    dftParameters::cellConstraintType            = prm.get_integer("CELL CONSTRAINT TYPE");
	}
	prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Boundary conditions");
    {
        dftParameters::radiusAtomBall                = prm.get_double("SELF POTENTIAL RADIUS");
	dftParameters::periodicX                     = prm.get_bool("PERIODIC1");
	dftParameters::periodicY                     = prm.get_bool("PERIODIC2");
	dftParameters::periodicZ                     = prm.get_bool("PERIODIC3");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Finite element mesh parameters");
    {
        dftParameters::finiteElementPolynomialOrder  = prm.get_integer("POLYNOMIAL ORDER");
        dftParameters::meshFileName                  = prm.get("MESH FILE");
	prm.enter_subsection ("Auto mesh generation parameters");
	{
	    dftParameters::outerAtomBallRadius           = prm.get_double("ATOM BALL RADIUS");
	    dftParameters::meshSizeOuterDomain           = prm.get_double("BASE MESH SIZE");
	    dftParameters::meshSizeInnerBall             = prm.get_double("MESH SIZE AT ATOM");
	    dftParameters::meshSizeOuterBall             = prm.get_double("MESH SIZE AROUND ATOM");
	}
        prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Brillouin zone k point sampling options");
    {
	prm.enter_subsection ("Monkhorst-Pack (MP) grid generation");
	{
	    dftParameters::nkx        = prm.get_integer("SAMPLING POINTS 1");
	    dftParameters::nky        = prm.get_integer("SAMPLING POINTS 2");
	    dftParameters::nkz        = prm.get_integer("SAMPLING POINTS 3");
	    dftParameters::offsetFlagX        = prm.get_integer("SAMPLING SHIFT 1");
	    dftParameters::offsetFlagY        = prm.get_integer("SAMPLING SHIFT 2");
	    dftParameters::offsetFlagZ        = prm.get_integer("SAMPLING SHIFT 3");
	}
	prm.leave_subsection ();

	dftParameters::useSymm                  = prm.get_bool("USE GROUP SYMMETRY");
	dftParameters::timeReversal                   = prm.get_bool("USE TIME REVERSAL SYMMETRY");
	dftParameters::kPointDataFile                = prm.get("kPOINT RULE FILE");

    }
    prm.leave_subsection ();

    prm.enter_subsection ("DFT functional parameters");
    {
	dftParameters::isPseudopotential             = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
	dftParameters::pseudoTestsFlag               = prm.get_bool("PSEUDO TESTS FLAG");
	dftParameters::pseudoPotentialFile           = prm.get("PSEUDOPOTENTIAL FILE NAMES LIST");
	dftParameters::xc_id                         = prm.get_integer("EXCHANGE CORRELATION TYPE");
	dftParameters::spinPolarized                 = prm.get_integer("SPIN POLARIZATION");
	dftParameters::start_magnetization           = prm.get_double("START MAGNETIZATION");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("SCF parameters");
    {
	dftParameters::TVal                          = prm.get_double("TEMPERATURE");
	dftParameters::numSCFIterations              = prm.get_integer("MAXIMUM ITERATIONS");
	dftParameters::selfConsistentSolverTolerance = prm.get_double("TOLERANCE");
	dftParameters::mixingHistory                 = prm.get_integer("ANDERSON SCHEME MIXING HISTORY");
	dftParameters::mixingParameter               = prm.get_double("ANDERSON SCHEME MIXING PARAMETER");
        dftParameters::startingWFCType               = prm.get("STARTING WFC");
	dftParameters::computeEnergyEverySCF         = prm.get_bool("COMPUTE ENERGY EACH ITER");


	prm.enter_subsection ("Eigen-solver parameters");
	{
	   dftParameters::numberEigenValues             = prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
	   dftParameters::lowerEndWantedSpectrum        = prm.get_double("LOWER BOUND WANTED SPECTRUM");
	   dftParameters::lowerBoundUnwantedFracUpper   = prm.get_double("LOWER BOUND UNWANTED FRAC UPPER");
	   dftParameters::chebyshevOrder                = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");
	   dftParameters::chebyshevBlockSize= prm.get_integer("CHEBYSHEV FILTER BLOCK SIZE");
	   dftParameters::useBatchGEMM= prm.get_bool("BATCH GEMM");
	   dftParameters::orthogType        = prm.get("ORTHOGONALIZATION TYPE");
	   dftParameters::chebyshevTolerance = prm.get_double("CHEBYSHEV FILTER TOLERANCE");
	   dftParameters::orthoRRWaveFuncBlockSize= prm.get_integer("ORTHO RR WFC BLOCK SIZE");
	   dftParameters::subspaceRotDofsBlockSize= prm.get_integer("SUBSPACE ROT DOFS BLOCK SIZE");
	   dftParameters::enableSwitchToGS= prm.get_bool("ENABLE SWITCH TO GS");
	   dftParameters::scalapackParalProcs= prm.get_integer("SCALAPACKPROCS");
	}
	prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Poisson problem parameters");
    {
       dftParameters::maxLinearSolverIterations     = prm.get_integer("MAXIMUM ITERATIONS");
       dftParameters::relLinearSolverTolerance      = prm.get_double("TOLERANCE");
    }
    prm.leave_subsection ();

  //
    check_print_parameters(prm);
    setHeuristicParameters();
  }



  void check_print_parameters(const dealii::ParameterHandler &prm)
  {
     if (dftParameters::verbosity >=1 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
     {
        std::cout << "==========================================================================================================" << std::endl ;
        std::cout << "==========================================================================================================" << std::endl ;
        std::cout << "			Welcome to the Open Source program DFT-FE v0.5.0-pre			        " << std::endl ;
        std::cout << "This is a C++ code for materials modeling from first principles using Kohn-Sham density functional theory " << std::endl ;
        std::cout << "It is based on adaptive finite-element based methodologies.		        " << std::endl ;
        std::cout << "For details and citing please refer: P.Motamarri et. al., Comp. Phys. Comm., Vol xx, Issue xx, pp xx, 2018" << std::endl ;
	std::cout << "==========================================================================================================" << std::endl ;
	std::cout << " DFT-FE authors (alphabetically) :									" << std::endl ;
	std::cout << "														" << std::endl ;
	std::cout << " Sambit Das               - University of Michigan, Ann Arbor" << std::endl ;
	std::cout << " Denis Davydov            - University of Erlangen-Nuremberg " << std::endl ;
	std::cout << " Vikram Gavini (Mentor)   - University of Michigan, Ann Arbor" << std::endl ;
	std::cout << " Krishnendu Ghosh         - University of Michigan, Ann Arbor" << std::endl ;
	std::cout << " Phani Motamarri          - University of Michigan, Ann Arbor" << std::endl ;
	std::cout << " Shiva Rudraraju          - University of Wisconsin-Madison  " << std::endl ;
        std::cout << "==========================================================================================================" << std::endl ;
        std::cout << " 	     Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors         " << std::endl ;
        std::cout << " 			DFT-FE is published under [LGPL v2.1 or newer] 				" << std::endl ;
        std::cout << "==========================================================================================================" << std::endl ;
        std::cout << "==========================================================================================================" << std::endl ;
    }

    const bool printParametersToFile=false;
    if (printParametersToFile && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
    {
	prm.print_parameters (std::cout, ParameterHandler::OutputStyle::LaTeX);
	exit(0);
    }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0 &&  dftParameters::verbosity>=1)
    {
      prm.print_parameters (std::cout, ParameterHandler::ShortText);
    }
#ifdef USE_COMPLEX
    if (dftParameters::isIonForce || dftParameters::isCellStress)
       AssertThrow(!dftParameters::useSymm,ExcMessage("DFT-FE Error: USE GROUP SYMMETRY must be set to false if either ION FORCE or CELL STRESS is set to true. This functionality will be added in a future release"));

#else
    AssertThrow(!dftParameters::isCellStress,ExcMessage("DFT-FE Error: Currently CELL STRESS cannot be set true in real mode for periodic Gamma point problems. This functionality will be added soon."));

    AssertThrow( dftParameters::nkx==1 &&  dftParameters::nky==1 &&  dftParameters::nkz==1
             && dftParameters::offsetFlagX==0 &&  dftParameters::offsetFlagY==0 &&  dftParameters::offsetFlagZ==0
	    ,ExcMessage("DFT-FE Error: Real executable cannot be used for non-zero k point."));
#endif
    AssertThrow(!(dftParameters::chkType==2 && (dftParameters::isIonOpt || dftParameters::isCellOpt)),ExcMessage("DFT-FE Error: CHK TYPE=2 cannot be used if geometry optimization is being performed."));

    AssertThrow(!(dftParameters::chkType==1 && (dftParameters::isIonOpt && dftParameters::isCellOpt)),ExcMessage("DFT-FE Error: CHK TYPE=1 cannot be used if both ION OPT and CELL OPT are set to true."));

    AssertThrow(dftParameters::nbandGrps<=dftParameters::numberEigenValues
	    ,ExcMessage("DFT-FE Error: NPBAND is greater than NUMBER OF KOHN-SHAM WAVEFUNCTIONS."));

    if (dftParameters::nonSelfConsistentForce)
       AssertThrow(false,ExcMessage("DFT-FE Error: Implementation of this feature is not completed yet."));

    AssertThrow(!dftParameters::coordinatesFile.empty()
	        ,ExcMessage("DFT-FE Error: ATOMIC COORDINATES FILE not given."));

    AssertThrow(!dftParameters::domainBoundingVectorsFile.empty()
	        ,ExcMessage("DFT-FE Error: DOMAIN VECTORS FILE not given."));

    if (dftParameters::isPseudopotential)
      AssertThrow(!dftParameters::pseudoPotentialFile.empty(),
	        ExcMessage("DFT-FE Error: PSEUDOPOTENTIAL FILE NAMES LIST not given."));

    AssertThrow(dftParameters::numberEigenValues!=0
	        ,ExcMessage("DFT-FE Error: Number of wavefunctions not specified or given value of zero, which is not allowed."));

    AssertThrow(dftParameters::natoms!=0
	        ,ExcMessage("DFT-FE Error: Number of atoms not specified or given a value of zero, which is not allowed."));

    AssertThrow(dftParameters::natomTypes!=0
	        ,ExcMessage("DFT-FE Error: Number of atom types not specified or given a value of zero, which is not allowed."));

#ifndef WITH_MKL;
    dftParameters::useBatchGEMM=false;
    if (dftParameters::verbosity >=1 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
        std::cout <<"Setting USE BATCH GEMM=false as intel mkl blas library is not being linked to."<<std::endl;

#endif
  }


  //FIXME: move this to triangulation manager, and set data members there
  //without changing the global dftParameters
  void setHeuristicParameters()
  {
    if (dftParameters::meshSizeOuterDomain<1.0e-6)
       if (dftParameters::periodicX ||dftParameters::periodicY ||dftParameters::periodicZ)
	   dftParameters::meshSizeOuterDomain=4.0;
       else
	   dftParameters::meshSizeOuterDomain=10.0;

    if (dftParameters::meshSizeInnerBall<1.0e-6)
       if (dftParameters::isPseudopotential)
	   dftParameters::meshSizeInnerBall=dftParameters::meshSizeOuterBall;
       else
	   dftParameters::meshSizeInnerBall=0.1*dftParameters::meshSizeOuterBall;
  }

}

}
