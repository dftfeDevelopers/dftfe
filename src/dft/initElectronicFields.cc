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

using namespace dftParameters ;

//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initElectronicFields(){
  computing_timer.enter_section("moved setup");

  //
  //initialize eigen vectors
  //
  matrix_free_data.initialize_dof_vector(vChebyshev,eigenDofHandlerIndex);
  v0Chebyshev.reinit(vChebyshev);
  fChebyshev.reinit(vChebyshev);
  aj[0].reinit(vChebyshev); aj[1].reinit(vChebyshev); aj[2].reinit(vChebyshev);
  aj[3].reinit(vChebyshev); aj[4].reinit(vChebyshev);
  for (unsigned int i=0; i<eigenVectors[0].size(); ++i)
    {  
      PSI[i]->reinit(vChebyshev);
      tempPSI[i]->reinit(vChebyshev);
      tempPSI2[i]->reinit(vChebyshev);
      tempPSI3[i]->reinit(vChebyshev);
    } 
  
  for(unsigned int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
      for(unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  eigenVectors[kPoint][i]->reinit(vChebyshev);
	  eigenVectorsOrig[kPoint][i]->reinit(vChebyshev);
	}
    }

   
  if ( (Utilities::MPI::this_mpi_process(interpoolcomm)) > 1 && (Utilities::MPI::this_mpi_process(mpi_communicator))==0 )
	std::cout << " check 2.1 " << std::endl ;
	

  //
  //initialize density 
  //
  initRho();

  
  //
  //initialize PSI
  //
  pcout << "reading initial guess for PSI\n";
  readPSI();
  computing_timer.exit_section("moved setup");   
}
