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

  extern unsigned int finiteElementPolynomialOrder,n_refinement_steps,numberEigenValues,xc_id, spinPolarized, nkx,nky,nkz, pseudoProjector;
  extern unsigned int chebyshevOrder,numPass,numSCFIterations,maxLinearSolverIterations, mixingHistory;

  extern double radiusAtomBall, domainSizeX, domainSizeY, domainSizeZ, mixingParameter, dkx, dky, dkz;
  extern double lowerEndWantedSpectrum,relLinearSolverTolerance,selfConsistentSolverTolerance,TVal, start_magnetization;

  extern bool isPseudopotential,periodicX,periodicY,periodicZ, useSymm, symmFromFile;
  extern std::string meshFileName,coordinatesFile,currentPath,latticeVectorsFile,kPointDataFile, symmDataFile;

  extern double innerDomainSize, outerBallRadius, innerBallRadius, meshSizeOuterDomain, meshSizeInnerDomain;
  extern double meshSizeOuterBall, meshSizeInnerBall, baseRefinementLevel;

};

#endif

