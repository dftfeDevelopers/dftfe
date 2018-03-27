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
// @author  Phani Motamarri (2018)
//

#include "../../include/dftParameters.h"


//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initElectronicFields(const bool usePreviousGroundStateRho){
  TimerOutput::Scope scope (computing_timer,"init electronic fields");

  //
  //initialize eigen vectors
  //
  matrix_free_data.initialize_dof_vector(vChebyshev,eigenDofHandlerIndex);
  v0Chebyshev.reinit(vChebyshev);
  fChebyshev.reinit(vChebyshev);

  //
  //temp STL d_v and d_f vectors required for upper bound computation filled here
  //

  if(dftParameters::spinPolarized!=1)
  {
     d_tempResidualNormWaveFunctions.clear();
     d_tempResidualNormWaveFunctions.resize(d_maxkPoints);
     for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
     {
        d_tempResidualNormWaveFunctions[kPoint].resize(eigenVectors[kPoint].size());
     }
  }


  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_maxkPoints; ++kPoint)
    {
      for(unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  eigenVectors[kPoint][i].reinit(vChebyshev);
	}
    }

  //
  //initialize density
  //
  if (!usePreviousGroundStateRho)
     initRho();
  else
     projectPreviousGroundStateRho();

  //
  //update serial and parallel unmoved previous mesh
  //
  d_mesh.generateSerialAndParallelUnmovedPreviousMesh(atomLocations,
				                      d_imagePositions,
				                      d_domainBoundingVectors);
  //
  //initialize PSI
  //
  pcout <<std::endl<< "Reading initial guess for PSI...."<<std::endl;
  readPSI();
}
