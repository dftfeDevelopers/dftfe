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

  unsigned int finiteElementPolynomialOrder=1,n_refinement_steps=1,numberEigenValues=1,xc_id=1, spinPolarized=0, nkx=1,nky=1,nkz=1, pseudoProjector=1;
  unsigned int chebyshevOrder=1,numPass=1, numSCFIterations=1,maxLinearSolverIterations=1, mixingHistory=1, npool=1;

  double radiusAtomBall=0.0, mixingParameter=0.5, dkx=0.0, dky=0.0, dkz=0.0;
  double lowerEndWantedSpectrum=0.0,relLinearSolverTolerance=1e-10,selfConsistentSolverTolerance=1e-10,TVal=500, start_magnetization=0.0;
  double chebyshevTolerance = 1e-02;

  bool isPseudopotential=false,periodicX=false,periodicY=false,periodicZ=false, useSymm=false, timeReversal=false;
  std::string meshFileName="",coordinatesFile="",domainBoundingVectorsFile="",kPointDataFile="", ionRelaxFlagsFile="",orthogType="";

  double outerAtomBallRadius=2.0, meshSizeOuterDomain=10.0;
  double meshSizeInnerBall=1.0, meshSizeOuterBall=1.0;

  bool isIonOpt=false, isCellOpt=false, isIonForce=false, isCellStress=false;
  bool nonSelfConsistentForce=false;
  double forceRelaxTol  = 5e-5;//Hartree/Bohr
  double stressRelaxTol = 5e-7;//Hartree/Bohr^3
  unsigned int cellConstraintType=12;// all cell components to be relaxed

  unsigned int verbosity=0; unsigned int chkType=0;
  bool restartFromChk=false;
  bool reproducible_output=false;
  bool electrostaticsPRefinement=false;

  unsigned int chebyshevBlockSize=1000;
  bool useBatchGEMM=false;
  unsigned int chebyshevOMPThreads=0;
  unsigned int orthoRROMPThreads=0;


  void declare_parameters(ParameterHandler &prm)
  {
    prm.declare_entry("REPRODUCIBLE OUTPUT", "false",
                      Patterns::Bool(),
                      "[Developer] Limit output to that which is reprodicible, i.e. don't print timing or absolute paths. This parameter is only used for testing purposes.");

    prm.declare_entry("VERBOSITY", "1",
                      Patterns::Integer(0,4),
                      "[Standard] Parameter to control verbosity of terminal output. 0 for low, 1 for medium, and 2 for high.");

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
	prm.declare_entry("ATOMIC COORDINATES FILE", "",
			  Patterns::Anything(),
			  "[Standard] Atomic-coordinates file. For fully non-periodic domain give cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic give fractional coordinates of atoms. File format (example for two atoms): x1 y1 z1 (row1), x2 y2 z2 (row2).");

	prm.declare_entry("DOMAIN BOUNDING VECTORS FILE", "",
			  Patterns::Anything(),
			  "[Standard] Set file specifying the domain bounding vectors v1, v2 and v3 in a.u. with the following format: v1x v1y v1z (row1), v2x v2y v2z (row2), v3x v3y v3z (row3). Domain bounding vectors are the typical lattice vectors in a fully periodic calculation.");

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

	    prm.declare_entry("FORCE TOL", "5e-5",
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

	    prm.declare_entry("STRESS TOL", "5e-7",
			      Patterns::Double(0,1.0),
			      "[Standard] Sets the tolerance of the cell stress (in a.u.) when cell stress is considered to be relaxed.");

	    prm.declare_entry("CELL CONSTRAINT TYPE", "12",
			      Patterns::Integer(1,13),
			      "[Standard] Cell relaxation constraint type, 1(isotropic shape-fixed volume optimization), 2(volume-fixed shape optimization), 3(relax only cell component v1x), 4(relax only cell component v2x), 5(relax only cell component v3x), 6(relax only cell components v2x and v3x), 7(relax only cell components v1x and v3x), 8(relax only cell components v1x and v2x), 9(volume optimization- relax only v1x, v2x and v3x), 10(2D- relax only x and y components relaxed), 11(2D- relax only x and y shape components- inplane area fixed), 12(relax all cell components), 13 automatically decides the constraints based boundary conditions. CAUTION: A majority of these options only make sense in an orthorhombic cell geometry.");

	}
	prm.leave_subsection ();

    }
    prm.leave_subsection ();

    prm.enter_subsection ("Boundary conditions");
    {
        prm.declare_entry("SELF POTENTIAL ATOM BALL RADIUS", "0.0",
                      Patterns::Double(0.0,10),
                      "[Developer] The radius (in a.u) of the ball around an atom on which self-potential of the associated nuclear charge is solved. For the default value of 0.0, the radius value is automatically determined to accomodate the largest radius possible for the given finite element mesh. The default approach works for most problems.");

	prm.declare_entry("PERIODIC1", "false",
			  Patterns::Bool(),
			  "[Standard] Periodicity along domain bounding vector, v1.");

	prm.declare_entry("PERIODIC2", "false",
			  Patterns::Bool(),
			  "[Standard] Periodicity along domain bounding vector, v2.");

	prm.declare_entry("PERIODIC3", "false",
			  Patterns::Bool(),
			  "[Standard] Periodicity along domain bounding vector, v3.");
    }
    prm.leave_subsection ();


    prm.enter_subsection ("Finite element mesh parameters");
    {

      prm.declare_entry("POLYNOMIAL ORDER", "4",
                        Patterns::Integer(1,12),
                       "[Standard] The degree of the finite-element interpolating polynomial");

      prm.declare_entry("MESH FILE", "",
                       Patterns::Anything(),
                       "[Developer] External mesh file path. If nothing is given auto mesh generation is performed");

      prm.enter_subsection ("Auto mesh generation parameters");
      {

	prm.declare_entry("BASE MESH SIZE", "2.0",
			  Patterns::Double(0,20),
			  "[Developer] Mesh size of the base mesh on which refinement is performed.");

	prm.declare_entry("ATOM BALL RADIUS","2.0",
			  Patterns::Double(0,10),
			  "[Developer] Radius of ball enclosing atom.");

	prm.declare_entry("MESH SIZE ATOM BALL", "0.5",
			  Patterns::Double(0,10),
			  "[Developer] Mesh size in a ball around atom.");

	prm.declare_entry("MESH SIZE NEAR ATOM", "0.5",
			  Patterns::Double(0,10),
			  "[Developer] Mesh size near atom. Useful for all-electron case.");

        prm.declare_entry("MAX REFINEMENT STEPS", "10",
                        Patterns::Integer(1,10),
                        "[Developer] Maximum number of refinement steps to be used. The default value is good enough in most cases.");


      }
      prm.leave_subsection ();
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Brillouin zone k point sampling options");
    {
        prm.enter_subsection ("Monkhorst-Pack (MP) grid generation");
        {
	    prm.declare_entry("SAMPLING POINTS 1", "1",
			      Patterns::Integer(1,100),
			      "[Standard] Number of Monkhorts-Pack grid points to be used along reciprocal latttice vector 1.");

	    prm.declare_entry("SAMPLING POINTS 2", "1",
			      Patterns::Integer(1,100),
			      "[Standard] Number of Monkhorts-Pack grid points to be used along reciprocal latttice vector 2.");

	    prm.declare_entry("SAMPLING POINTS 3", "1",
			      Patterns::Integer(1,100),
			      "[Standard] Number of Monkhorts-Pack grid points to be used along reciprocal latttice vector 3.");

	    prm.declare_entry("SAMPLING SHIFT 1", "0.0",
			      Patterns::Double(0.0,1.0),
			      "[Standard] Fractional shifting to be used along reciprocal latttice vector 1.");

	    prm.declare_entry("SAMPLING SHIFT 2", "0.0",
			      Patterns::Double(0.0,1.0),
			      "[Standard] Fractional shifting to be used along reciprocal latttice vector 2.");

	    prm.declare_entry("SAMPLING SHIFT 3", "0.0",
			      Patterns::Double(0.0,1.0),
			      "[Standard] Fractional shifting to be used along reciprocal latttice vector 3.");

	}
	prm.leave_subsection ();

	prm.declare_entry("kPOINT RULE FILE", "",
			  Patterns::Anything(),
			  "[Developer] File specifying the k-Point quadrature rule to sample Brillouin zone. CAUTION: This option is only used for postprocessing, for example band structure calculation. To set k point rule for DFT solve use the Monkhorst-Pack (MP) grid generation.");

	prm.declare_entry("USE GROUP SYMMETRY", "false",
			  Patterns::Bool(),
			  "[Standard] Flag to control whether to use point group symmetries (set to false for relaxation calculation).");

	prm.declare_entry("USE TIME REVERSAL SYMMETRY", "false",
			  Patterns::Bool(),
			  "[Standard] Flag to control usage of time reversal symmetry.");

	prm.declare_entry("NUMBER OF POOLS", "1",
			  Patterns::Integer(1),
			  "[Standard] Number of pools the irreducible k-points to be split on should be a divisor of total number of procs and be less than or equal to the number of irreducible k-points.");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("DFT functional related parameters");
    {

	prm.declare_entry("PSEUDOPOTENTIAL CALCULATION", "true",
			  Patterns::Bool(),
			  "[Standard] Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed.");

	prm.declare_entry("PSEUDOPOTENTIAL TYPE", "1",
			  Patterns::Integer(1,2),
			  "[Standard] Type of nonlocal projector to be used: 1 for KB, 2 for ONCV, default is KB.");

	prm.declare_entry("EXCHANGE CORRELATION TYPE", "1",
			  Patterns::Integer(1,4),
			  "[Standard] Parameter specifying the type of exchange-correlation to be used: 1(LDA: Perdew Zunger Ceperley Alder correlation with Slater Exchange[PRB. 23, 5048 (1981)]), 2(LDA: Perdew-Wang 92 functional with Slater Exchange [PRB. 45, 13244 (1992)]), 3(LDA: Vosko, Wilk \\& Nusair with Slater Exchange[Can. J. Phys. 58, 1200 (1980)]), 4(GGA: Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)]).");

	prm.declare_entry("SPIN POLARIZATION", "0",
			  Patterns::Integer(0,1),
			  "[Standard] Spin polarization: 0 for no spin polarization and 1 for spin polarization.");

	prm.declare_entry("START MAGNETIZATION", "0.0",
			  Patterns::Double(-0.5,0.5),
			  "[Standard] Magnetization to start with (must be between -0.5 and +0.5).");
    }
    prm.leave_subsection ();


    prm.enter_subsection ("SCF parameters");
    {
	prm.declare_entry("TEMPERATURE", "500.0",
			  Patterns::Double(),
			  "[Standard] Fermi-Dirac smearing temperature (in Kelvin).");

	prm.declare_entry("MAXIMUM ITERATIONS", "50",
			  Patterns::Integer(1,1000),
			  "[Standard] Maximum number of iterations to be allowed for SCF convergence");

	prm.declare_entry("TOLERANCE", "1e-08",
			  Patterns::Double(0,1.0),
			  "[Standard] SCF iterations stopping tolerance in terms of electron-density difference between two successive iterations.");

	prm.declare_entry("ANDERSON SCHEME MIXING HISTORY", "70",
			  Patterns::Integer(1,1000),
			  "[Standard] Number of SCF iterations to be considered for mixing the electron-density.");

	prm.declare_entry("ANDERSON SCHEME MIXING PARAMETER", "0.5",
			  Patterns::Double(0.0,1.0),
			  "[Standard] Mixing parameter to be used in Anderson scheme.");
    }
    prm.leave_subsection ();


    prm.enter_subsection ("Eigen-solver/Chebyshev solver related parameters");
    {

	prm.declare_entry("NUMBER OF KOHN-SHAM WAVEFUNCTIONS", "10",
			  Patterns::Integer(0),
			  "[Standard] Number of Kohn-Sham wavefunctions to be computed. For insulators use N/2+(10-20) and for metals use 20 percent more than N/2 (atleast 10 more). N is the total number of electrons.");

	prm.declare_entry("LOWER BOUND WANTED SPECTRUM", "-10.0",
			  Patterns::Double(),
			  "[Developer] The lower bound of the wanted eigen spectrum");

	prm.declare_entry("CHEBYSHEV POLYNOMIAL DEGREE", "0",
			  Patterns::Integer(0,2000),
			  "[Developer] The degree of the Chebyshev polynomial to be employed for filtering out the unwanted spectrum (Default value is used when the input parameter value is 0.");

	prm.declare_entry("CHEBYSHEV FILTER PASSES", "1",
			  Patterns::Integer(1,20),
			  "[Developer] The initial number of the Chebyshev filter passes per SCF. More Chebyshev filter passes beyond the value set in this parameter can still happen due to additional algorithms used in the code.");


	prm.declare_entry("CHEBYSHEV FILTER TOLERANCE","5e-02",
			  Patterns::Double(0),
			  "[Developer] Parameter specifying the tolerance to which eigenvectors need to computed using chebyshev filtering approach");

	prm.declare_entry("CHEBYSHEV FILTER BLOCK SIZE", "1000",
			  Patterns::Integer(1),
			  "[Developer] The maximum number of wavefunctions which are handled by one call to the Chebyshev filter. This is useful for optimization purposes. The optimum value is dependent on the computing architecture.");

	prm.declare_entry("BATCH GEMM", "false",
			  Patterns::Bool(),
			  "[Developer] Boolean parameter specifying whether to use gemm_batch blas routines to perform matrix-matrix multiplication operations with groups of matrices, processing a number of groups at once using threads instead of the standard serial route. CAUTION: batch blas routines will only be activated if the CHEBYSHEV FILTER BLOCK SIZE is less than 1000.");

        prm.declare_entry("CHEBYSHEV FILTER NUM OMP THREADS", "0",
			  Patterns::Integer(0,300),
			  "[Developer] Sets the number of OpenMP threads to be used in the blas linear algebra calls inside the Chebyshev filtering. The default value is 0, for which no action is taken. CAUTION: For non zero values, CHEBYSHEV FILTER NUM OMP THREADS takes precedence over the OMP_NUM_THREADS environment variable.");

	prm.declare_entry("ORTHO RR NUM OMP THREADS", "0",
			  Patterns::Integer(0,300),
			  "[Developer] Sets the number of OpenMP threads to be used in the blas linear algebra calls inside Lowden Orthogonalization and Rayleigh-Ritz projection steps. The default value is 0, for which no action is taken. CAUTION: For non-zero values, CHEBYSHEV FILTER NUM OMP THREADS takes precedence over the OMP_NUM_THREADS environment variable.");


	prm.declare_entry("ORTHOGONALIZATION TYPE","GS",
			  Patterns::Anything(),
			  "[Standard] Parameter specifying the type of orthogonalization to be used: GS(Gram-Schmidt Orthogonalization), Lowden(Lowden Orthogonalization)");

    }
    prm.leave_subsection ();


    prm.enter_subsection ("Poisson problem paramters");
    {
	prm.declare_entry("MAXIMUM ITERATIONS", "5000",
			  Patterns::Integer(0,20000),
			  "[Developer] Maximum number of iterations to be allowed for Poisson problem convergence.");

	prm.declare_entry("TOLERANCE", "1e-12",
			  Patterns::Double(0,1.0),
			  "[Developer] Relative tolerance as stopping criterion for Poisson problem convergence.");

	prm.declare_entry("P REFINEMENT", "false",
			  Patterns::Bool(),
			  "[Standard] Boolean parameter specifying whether to project the ground-state electron density to a p refined mesh, and solve for the electrostatic fields on the p refined mesh. This step is not performed for each SCF, but only at the ground-state. The purpose is to improve the accuracy of the ground-state electrostatic energy.");
    }
    prm.leave_subsection ();

  }

  void parse_parameters(ParameterHandler &prm)
  {
    dftParameters::verbosity                     = prm.get_integer("VERBOSITY");
    dftParameters::reproducible_output           = prm.get_bool("REPRODUCIBLE OUTPUT");

    prm.enter_subsection ("Checkpointing and Restart");
    {
	chkType=prm.get_integer("CHK TYPE");
	restartFromChk=prm.get_bool("RESTART FROM CHK") && chkType!=0;
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Geometry");
    {
        dftParameters::coordinatesFile               = prm.get("ATOMIC COORDINATES FILE");
        dftParameters::domainBoundingVectorsFile     = prm.get("DOMAIN BOUNDING VECTORS FILE");
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
        dftParameters::radiusAtomBall                = prm.get_double("SELF POTENTIAL ATOM BALL RADIUS");
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
	    dftParameters::meshSizeInnerBall             = prm.get_double("MESH SIZE NEAR ATOM");
	    dftParameters::meshSizeOuterBall             = prm.get_double("MESH SIZE ATOM BALL");
	    dftParameters::n_refinement_steps            = prm.get_integer("MAX REFINEMENT STEPS");
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
	    dftParameters::dkx        = prm.get_double("SAMPLING SHIFT 1");
	    dftParameters::dky        = prm.get_double("SAMPLING SHIFT 2");
	    dftParameters::dkz        = prm.get_double("SAMPLING SHIFT 3");
	}
	prm.leave_subsection ();

	dftParameters::useSymm                  = prm.get_bool("USE GROUP SYMMETRY");
	dftParameters::timeReversal                   = prm.get_bool("USE TIME REVERSAL SYMMETRY");
	dftParameters::npool             = prm.get_integer("NUMBER OF POOLS");
	dftParameters::kPointDataFile                = prm.get("kPOINT RULE FILE");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("DFT functional related parameters");
    {
	dftParameters::isPseudopotential             = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
	dftParameters::pseudoProjector               = prm.get_integer("PSEUDOPOTENTIAL TYPE");
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
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Eigen-solver/Chebyshev solver related parameters");
    {
       dftParameters::numberEigenValues             = prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
       dftParameters::lowerEndWantedSpectrum        = prm.get_double("LOWER BOUND WANTED SPECTRUM");
       dftParameters::chebyshevOrder                = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");
       dftParameters::numPass           = prm.get_integer("CHEBYSHEV FILTER PASSES");
       dftParameters::chebyshevBlockSize= prm.get_integer("CHEBYSHEV FILTER BLOCK SIZE");
       dftParameters::useBatchGEMM= prm.get_bool("BATCH GEMM");
       dftParameters::orthogType        = prm.get("ORTHOGONALIZATION TYPE");
       dftParameters::chebyshevTolerance = prm.get_double("CHEBYSHEV FILTER TOLERANCE");
       dftParameters::chebyshevOMPThreads = prm.get_integer("CHEBYSHEV FILTER NUM OMP THREADS");
       dftParameters::orthoRROMPThreads= prm.get_integer("ORTHO RR NUM OMP THREADS");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Poisson problem paramters");
    {
       dftParameters::maxLinearSolverIterations     = prm.get_integer("MAXIMUM ITERATIONS");
       dftParameters::relLinearSolverTolerance      = prm.get_double("TOLERANCE");
       dftParameters::electrostaticsPRefinement        = prm.get_bool("P REFINEMENT");
    }
    prm.leave_subsection ();

    check_print_parameters(prm);
  }



  void check_print_parameters(const dealii::ParameterHandler &prm)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0 &&  dftParameters::verbosity>=1)
    {
      prm.print_parameters (std::cout, ParameterHandler::Text);
    }

    const bool printParametersToFile=false;
    if (printParametersToFile && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
    {
	std::ofstream output ("parameterFile.tex");
	prm.print_parameters (output, ParameterHandler::OutputStyle::LaTeX);
	exit(0);
    }
#ifdef USE_COMPLEX
    if (dftParameters::electrostaticsPRefinement)
       AssertThrow(!dftParameters::useSymm,ExcMessage("DFT-FE Error: P REFINEMENT=true is not yet extended to USE GROUP SYMMETRY=true case"));

    if (dftParameters::isIonForce || dftParameters::isCellStress)
       AssertThrow(!dftParameters::useSymm,ExcMessage("DFT-FE Error: USE GROUP SYMMETRY must be set to false if either ION FORCE or CELL STRESS is set to true. This functionality will be added in a future release"));
#else
    AssertThrow(!dftParameters::isCellStress,ExcMessage("DFT-FE Error: Currently CELL STRESS cannot be set true in double mode for periodic Gamma point problems. This functionality will be added soon."));
#endif
    AssertThrow(!(dftParameters::chkType==2 && (dftParameters::isIonOpt || dftParameters::isCellOpt)),ExcMessage("DFT-FE Error: CHK TYPE=2 cannot be used if geometry optimization is being performed."));

    AssertThrow(!(dftParameters::chkType==1 && (dftParameters::isIonOpt && dftParameters::isCellOpt)),ExcMessage("DFT-FE Error: CHK TYPE=1 cannot be used if both ION OPT and CELL OPT are set to true."));
  }

}

}
