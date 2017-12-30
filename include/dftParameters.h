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

#ifndef dftParameters_H_
#define dftParameters_H_

#include <string>
//
//Declare dftUtils functions
//
namespace dftParameters
{

  extern unsigned int finiteElementPolynomialOrder,n_refinement_steps,numberEigenValues,xc_id;
  extern unsigned int chebyshevOrder,numSCFIterations,maxLinearSolverIterations, mixingHistory;

  extern double radiusAtomBall, domainSizeX, domainSizeY, domainSizeZ, mixingParameter;
  extern double lowerEndWantedSpectrum,relLinearSolverTolerance,selfConsistentSolverTolerance,TVal;

  extern bool isPseudopotential,periodicX,periodicY,periodicZ;
  extern std::string meshFileName,coordinatesFile,currentPath,latticeVectorsFile,kPointDataFile;

  extern double innerDomainSize, outerBallRadius, innerBallRadius, meshSizeOuterDomain, meshSizeInnerDomain;
  extern double meshSizeOuterBall, meshSizeInnerBall, baseRefinementLevel;

};

#endif
