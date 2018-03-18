// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri (2017)
//
#include <dftParameters.h>
#include <iostream>
#include <fstream>
#include <deal.II/base/data_out_base.h>

using namespace dealii;

namespace dftParameters
{
  
  unsigned int finiteElementPolynomialOrder=1,n_refinement_steps=1,numberEigenValues=1,xc_id=1, spinPolarized=0, nkx=1,nky=1,nkz=1, pseudoProjector=1;
  unsigned int chebyshevOrder=1,numPass=1, numSCFIterations=1,maxLinearSolverIterations=1, mixingHistory=1, npool=1;

  double radiusAtomBall=1.0, mixingParameter=0.5, dkx=0.0, dky=0.0, dkz=0.0;
  double lowerEndWantedSpectrum=0.0,relLinearSolverTolerance=1e-10,selfConsistentSolverTolerance=1e-10,TVal=500, start_magnetization=0.0;

  bool isPseudopotential=false,periodicX=false,periodicY=false,periodicZ=false, useSymm=false, symmFromFile=false, timeReversal=false;
  std::string meshFileName=" ",coordinatesFile=" ",currentPath=" ",domainBoundingVectorsFile=" ",kPointDataFile=" ", symmDataFile=" ", ionRelaxFlagsFile=" ";

  double innerDomainSizeX=1.0, innerDomainSizeY=1.0, innerDomainSizeZ=1.0, outerAtomBallRadius=2.0, meshSizeOuterDomain=10.0, meshSizeInnerDomain=5.0;
  double meshSizeInnerBall=1.0, meshSizeOuterBall=1.0;

  bool isIonOpt=false, isCellOpt=false, isIonForce=false, isCellStress=false;
  double forceRelaxTol=5e-5;//Hartree/Bohr
  double stressRelaxTol=5e-7;//Hartree/Bohr^3
  unsigned int cellConstraintType=12;// all cell components to be relaxed



  void declare_parameters(ParameterHandler &prm)
  {
    prm.declare_entry("OPTIMIZED MODE", "true",
                      Patterns::Bool(),
                      "Flag to control optimized/debug modes");

    prm.declare_entry("DFT PATH", "",
                      Patterns::Anything(),
                      "Path specifying the location of the build directory");

    prm.declare_entry("MESH FILE", "",
                      Patterns::Anything(),
                      "Finite-element mesh file to be used for the given problem");

    prm.declare_entry("INNER DOMAIN SIZE X","0.0",
                      Patterns::Double(),
                      "Inner Domain Size along 1-direction");

    prm.declare_entry("INNER DOMAIN SIZE Y","0.0",
                      Patterns::Double(),
                      "Inner Domain Size along 2-direction");

    prm.declare_entry("INNER DOMAIN SIZE Z","0.0",
                      Patterns::Double(),
                      "Inner Domain Size along 3-direction");


    prm.declare_entry("OUTER ATOM BALL RADIUS","0.0",
                      Patterns::Double(),
                      "Radius of outer ball enclosing atom");

    prm.declare_entry("MESH SIZE OUTER DOMAIN", "0.0",
                      Patterns::Double(),
                      "Outer Domain Mesh Size");

    prm.declare_entry("MESH SIZE INNER DOMAIN", "0.0",
                      Patterns::Double(),
                      "Inner Domain Mesh Size");

    prm.declare_entry("MESH SIZE NEAR ATOM", "0.0",
                      Patterns::Double(),
                      "Mesh Size near atom");

    prm.declare_entry("MESH SIZE OUTER ATOM BALL", "0.0",
                      Patterns::Double(),
                      "Mesh Size in a ball around atom");


    prm.declare_entry("ATOMIC COORDINATES FILE", "",
                      Patterns::Anything(),
                      "File specifying the coordinates of the atoms in the given material system");

    prm.declare_entry("DOMAIN BOUNDING VECTORS FILE", "",
                      Patterns::Anything(),
                      "File specifying the domain bounding vectors associated with the domain. In fully periodic case, these are the lattice vectors.");

    prm.declare_entry("kPOINT RULE FILE", "",
                      Patterns::Anything(),
                      "File specifying the k-Point quadrature rule to sample Brillouin zone");
    prm.declare_entry("READ SYMMETRY FROM FILE", "false",
                      Patterns::Bool(),
                      "Flag to control whether to read symmetries supplied by user");
    prm.declare_entry("SYMMETRY MATRIX FILE", "",
                      Patterns::Anything(),
                      "File specifying the symmetry matrices for obtaining the irreducible BZ");
    prm.declare_entry("ION RELAX FLAGS FILE", "",
                      Patterns::Anything(),
                      "File specifying the atomic position update permission flags. 1- update 0- no update");
    prm.declare_entry("BZ SAMPLING POINTS ALONG X", "2",
                      Patterns::Integer(1,100),
                      "Number of Monkhorts-Pack grid points to be used along X direction for BZ sampling");
    prm.declare_entry("BZ SAMPLING POINTS ALONG Y", "2",
                      Patterns::Integer(1,100),
                      "Number of Monkhorts-Pack grid points to be used along Y direction for BZ sampling");
    prm.declare_entry("BZ SAMPLING POINTS ALONG Z", "2",
                      Patterns::Integer(1,100),
                      "Number of Monkhorts-Pack grid points to be used along Z direction for BZ sampling");
    prm.declare_entry("BZ SAMPLING SHIFT ALONG X", "0.0",
                      Patterns::Double(0.0,1.0),
                      "Fractional shifting to be used along X direction for BZ sampling");
    prm.declare_entry("BZ SAMPLING SHIFT ALONG Y", "0.0",
                      Patterns::Double(0.0,1.0),
                      "Fractional shifting to be used along Y direction for BZ sampling");
    prm.declare_entry("BZ SAMPLING SHIFT ALONG Z", "0.0",
                      Patterns::Double(0.0,1.0),
                      "Fractional shifting to be used along Z direction for BZ sampling");
    prm.declare_entry("USE GROUP SYMMETRY", "true",
                      Patterns::Bool(),
                      "Flag to control usage of space group symmetries (only for periodic calculation)");
    prm.declare_entry("USE TIME REVERSAL SYMMETRY", "false",
                      Patterns::Bool(),
                      "Flag to control usage of time reversal symmetry ");

    prm.declare_entry("FINITE ELEMENT POLYNOMIAL ORDER", "2",
                      Patterns::Integer(1,12),
                      "The degree of the finite-element interpolating polynomial");


    prm.declare_entry("SELF POTENTIAL ATOM BALL RADIUS", "3.0",
                      Patterns::Double(),
                      "The radius of the ball around an atom on which self-potential of the associated nuclear charge is solved");

    prm.declare_entry("SPIN POLARIZATION", "0",
                      Patterns::Integer(0,1),
                      "Is spin polarization to be included?");

    prm.declare_entry("START MAGNETIZATION", "0.0",
                      Patterns::Double(),
                      "Magnetization to start with");

    prm.declare_entry("PERIODIC BOUNDARY CONDITION X", "false",
                      Patterns::Bool(),
                      "Periodicity in X-direction");

    prm.declare_entry("PERIODIC BOUNDARY CONDITION Y", "false",
                      Patterns::Bool(),
                      "Periodicity in Y-direction");

    prm.declare_entry("PERIODIC BOUNDARY CONDITION Z", "false",
                      Patterns::Bool(),
                      "Periodicity in Z-direction");

    prm.declare_entry("PSEUDOPOTENTIAL CALCULATION", "false",
                      Patterns::Bool(),
                      "Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed");

    prm.declare_entry("PSEUDOPOTENTIAL TYPE", "1",
                      Patterns::Integer(1,2),
                      "Type of nonlocal projector to be used: 1 for KB, 2 for ONCV, default is KB");

    prm.declare_entry("EXCHANGE CORRELATION TYPE", "1",
                      Patterns::Integer(1,4),
                      "Parameter specifying the type of exchange-correlation to be used");

    prm.declare_entry("ION OPT", "false",
                      Patterns::Bool(),
                      "Boolean parameter specifying if atomic forces are to be relaxed");

    prm.declare_entry("CELL OPT", "false",
                      Patterns::Bool(),
                      "Boolean parameter specifying if cell stress is to be relaxed");

    prm.declare_entry("ION FORCE", "false",
                      Patterns::Bool(),
                      "Boolean parameter specifying if atomic forces are to be computed");

    prm.declare_entry("CELL STRESS", "false",
                      Patterns::Bool(),
                      "Boolean parameter specifying if cell stress is to be computed");

    prm.declare_entry("FORCE TOL", "5e-5",
                      Patterns::Double(),
                      "Sets the tolerance of the maximum force (in atomic units) on an ion when forces are considered to be relaxed.");   

    prm.declare_entry("STRESS TOL", "5e-7",
                      Patterns::Double(),
                      "Sets the tolerance of the cell stress (in atomic units) when cell stress is considered to be relaxed.");

    prm.declare_entry("CELL CONSTRAINT TYPE", "12",
                      Patterns::Integer(1,12),
                      "Type of the constraint to be used during cell stress relaxation");    

    prm.declare_entry("NUMBER OF REFINEMENT STEPS", "4",
                      Patterns::Integer(1,10),
                      "Number of refinement steps to be used");

    prm.declare_entry("LOWER BOUND WANTED SPECTRUM", "-10.0",
                      Patterns::Double(),
                      "The lower bound of the wanted eigen spectrum");

    prm.declare_entry("CHEBYSHEV POLYNOMIAL DEGREE", "0",
                      Patterns::Integer(),
                      "The degree of the Chebyshev polynomial to be employed for filtering out the unwanted spectrum (Default value is used when the input parameter value is 0");

    prm.declare_entry("CHEBYSHEV FILTER PASSES", "1",
                      Patterns::Integer(),
                      "The number of the Chebyshev filter passes per SCF  (Default value is used when the input parameter is not specified");

    prm.declare_entry("NUMBER OF KOHN-SHAM WAVEFUNCTIONS", "10",
                      Patterns::Integer(),
                      "Number of Kohn-Sham wavefunctions to be computed");

    prm.declare_entry("TEMPERATURE", "500.0",
                      Patterns::Double(),
                      "Fermi-Dirac smearing temperature");

    prm.declare_entry("SCF CONVERGENCE MAXIMUM ITERATIONS", "50",
                      Patterns::Integer(),
                      "Maximum number of iterations to be allowed for SCF convergence");

    prm.declare_entry("SCF CONVERGENCE TOLERANCE", "1e-08",
                      Patterns::Double(),
                      "SCF iterations stopping tolerance in terms of electron-density difference between two successive iterations");

    prm.declare_entry("ANDERSON SCHEME MIXING HISTORY", "70",
                      Patterns::Integer(),
                      "Number of SCF iterations to be considered for mixing the electron-density");

    prm.declare_entry("ANDERSON SCHEME MIXING PARAMETER", "0.5",
                      Patterns::Double(0.0,1.0),
                      "Mixing parameter to be used in Anderson scheme");

    prm.declare_entry("POISSON SOLVER CONVERGENCE MAXIMUM ITERATIONS", "5000",
                      Patterns::Integer(),
                      "Maximum number of iterations to be allowed for Poisson problem convergence");

    prm.declare_entry("POISSON SOLVER CONVERGENCE TOLERANCE", "1e-12",
                      Patterns::Double(),
                      "Relative tolerance as stopping criterion for Poisson problem convergence");

    prm.declare_entry("NUMBER OF POOLS", "1",
                      Patterns::Integer(),
                      "Number of pools to be generated for k-point parallelization");
  }

  void parse_parameters(const ParameterHandler &prm)
  {
    dftParameters::currentPath                   = prm.get("DFT PATH");
    dftParameters::currentPath.erase(std::remove(dftParameters::currentPath.begin(),dftParameters::currentPath.end(),'"'),dftParameters::currentPath.end());
    dftParameters::meshFileName                  = prm.get("MESH FILE");
    dftParameters::finiteElementPolynomialOrder  = prm.get_integer("FINITE ELEMENT POLYNOMIAL ORDER");
    dftParameters::n_refinement_steps            = prm.get_integer("NUMBER OF REFINEMENT STEPS");
    dftParameters::coordinatesFile               = prm.get("ATOMIC COORDINATES FILE");
    dftParameters::radiusAtomBall                = prm.get_double("SELF POTENTIAL ATOM BALL RADIUS");
    dftParameters::innerDomainSizeX              = prm.get_double("INNER DOMAIN SIZE X");
    dftParameters::innerDomainSizeY              = prm.get_double("INNER DOMAIN SIZE Y");
    dftParameters::innerDomainSizeZ              = prm.get_double("INNER DOMAIN SIZE Z");
    dftParameters::outerAtomBallRadius           = prm.get_double("OUTER ATOM BALL RADIUS");
    dftParameters::meshSizeOuterDomain           = prm.get_double("MESH SIZE OUTER DOMAIN");
    dftParameters::meshSizeInnerDomain           = prm.get_double("MESH SIZE INNER DOMAIN");
    dftParameters::spinPolarized                 = prm.get_integer("SPIN POLARIZATION");
    dftParameters::start_magnetization           = prm.get_double("START MAGNETIZATION");
    dftParameters::meshSizeInnerBall             = prm.get_double("MESH SIZE NEAR ATOM");
    dftParameters::meshSizeOuterBall             = prm.get_double("MESH SIZE OUTER ATOM BALL");
    dftParameters::periodicX                     = prm.get_bool("PERIODIC BOUNDARY CONDITION X");
    dftParameters::periodicY                     = prm.get_bool("PERIODIC BOUNDARY CONDITION Y");
    dftParameters::periodicZ                     = prm.get_bool("PERIODIC BOUNDARY CONDITION Z");
    dftParameters::domainBoundingVectorsFile     = prm.get("DOMAIN BOUNDING VECTORS FILE");
    dftParameters::kPointDataFile                = prm.get("kPOINT RULE FILE");
    dftParameters::symmFromFile                  = prm.get_bool("READ SYMMETRY FROM FILE");
    dftParameters::symmDataFile                  = prm.get("SYMMETRY MATRIX FILE");
    dftParameters::ionRelaxFlagsFile             = prm.get("ION RELAX FLAGS FILE");
    dftParameters::nkx        = prm.get_integer("BZ SAMPLING POINTS ALONG X");
    dftParameters::nky        = prm.get_integer("BZ SAMPLING POINTS ALONG Y");
    dftParameters::nkz        = prm.get_integer("BZ SAMPLING POINTS ALONG Z");
    dftParameters::dkx        = prm.get_double("BZ SAMPLING SHIFT ALONG X");
    dftParameters::dky        = prm.get_double("BZ SAMPLING SHIFT ALONG Y");
    dftParameters::dkz        = prm.get_double("BZ SAMPLING SHIFT ALONG Z");
    dftParameters::useSymm                  = prm.get_bool("USE GROUP SYMMETRY");
    dftParameters::timeReversal                   = prm.get_bool("USE TIME REVERSAL SYMMETRY");
    dftParameters::isPseudopotential             = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
    dftParameters::pseudoProjector               = prm.get_integer("PSEUDOPOTENTIAL TYPE");
    dftParameters::xc_id                         = prm.get_integer("EXCHANGE CORRELATION TYPE");
    dftParameters::numberEigenValues             = prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
    dftParameters::isIonOpt                      = prm.get_bool("ION OPT");
    dftParameters::isCellOpt                     = prm.get_bool("CELL OPT");
    dftParameters::isIonForce                    = dftParameters::isIonOpt || prm.get_bool("ION FORCE");
    dftParameters::isCellStress                  = dftParameters::isCellOpt || prm.get_bool("CELL STRESS");
    dftParameters::forceRelaxTol                 = prm.get_double("FORCE TOL");
    dftParameters::stressRelaxTol                = prm.get_double("STRESS TOL"); 
    dftParameters::cellConstraintType            = prm.get_integer("CELL CONSTRAINT TYPE"); 
    dftParameters::lowerEndWantedSpectrum        = prm.get_double("LOWER BOUND WANTED SPECTRUM");
    dftParameters::chebyshevOrder                = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");
    dftParameters::numPass           = prm.get_integer("CHEBYSHEV FILTER PASSES");
    dftParameters::numSCFIterations              = prm.get_integer("SCF CONVERGENCE MAXIMUM ITERATIONS");
    dftParameters::selfConsistentSolverTolerance = prm.get_double("SCF CONVERGENCE TOLERANCE");
    dftParameters::mixingHistory                 = prm.get_integer("ANDERSON SCHEME MIXING HISTORY");
    dftParameters::mixingParameter               = prm.get_double("ANDERSON SCHEME MIXING PARAMETER");
    dftParameters::TVal                          = prm.get_double("TEMPERATURE");
    dftParameters::maxLinearSolverIterations     = prm.get_integer("POISSON SOLVER CONVERGENCE MAXIMUM ITERATIONS");
    dftParameters::relLinearSolverTolerance      = prm.get_double("POISSON SOLVER CONVERGENCE TOLERANCE");
    dftParameters::npool             = prm.get_integer("NUMBER OF POOLS");
  }

}
