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
// @author Sambit Das (2018)
//

#ifndef geoOpt_H_
#define geoOpt_H_
#include "headers.h"
#include "constants.h"

using namespace dealii;
typedef dealii::parallel::distributed::Vector<double> vectorType;
template <unsigned int FEOrder> class forceClass;
template <unsigned int FEOrder> class dftClass;
//
//Define geoOpt class
//
template <unsigned int FEOrder>
class geoOpt
{

public:
  geoOpt(dftClass<FEOrder>* _dftPtr, forceClass<FEOrder>* _forcePtr);
  void init();
  void relax();   
private:
  void relaxAtomsForces();
  void relaxStress();
  void relaxAtomsForcesStress();
  //pointer to dft class
  dftClass<FEOrder>* dftPtr;
  //pointer to force class
  forceClass<FEOrder>* forcePtr;
  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;
};

#endif
