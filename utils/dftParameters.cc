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
#include "../include/dftParameters.h" 

namespace dftParameters
{
  
  unsigned int finiteElementPolynomialOrder=1,n_refinement_steps=1,numberEigenValues=1,xc_id=1, spinPolarized=0, nkx=1,nky=1,nkz=1, pseudoProjector=1;
  unsigned int chebyshevOrder=1,numPass=1, numSCFIterations=1,maxLinearSolverIterations=1, mixingHistory=1;

  double radiusAtomBall=1.0, domainSizeX=1.0, domainSizeY=1.0, domainSizeZ=1.0, mixingParameter=0.5, dkx=0.0, dky=0.0, dkz=0.0;
  double lowerEndWantedSpectrum=0.0,relLinearSolverTolerance=1e-10,selfConsistentSolverTolerance=1e-10,TVal=500, start_magnetization=0.0;

  bool isPseudopotential=false,periodicX=false,periodicY=false,periodicZ=false, useSymm=false, symmFromFile=false;
  std::string meshFileName=" ",coordinatesFile=" ",currentPath=" ",latticeVectorsFile=" ",kPointDataFile=" ", symmDataFile=" ";

  double innerDomainSizeX=1.0, innerDomainSizeY=1.0, innerDomainSizeZ=1.0, meshSizeOuterDomain=10.0, meshSizeInnerDomain=5.0;
  double meshSizeInnerBall=1.0;

}
