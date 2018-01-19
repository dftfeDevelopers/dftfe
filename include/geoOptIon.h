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

#ifndef geoOptIon_H_
#define geoOptIon_H_
#include "solverFunction.h"
#include "constants.h"

using namespace dealii;
template <unsigned int FEOrder> class dftClass;
//
//Define geoOpt class
//
template <unsigned int FEOrder>
class geoOptIon : public solverFunction
{
public:
  geoOptIon(dftClass<FEOrder>* _dftPtr);
  void init();
  void run();   
private:

    
  int getNumberUnknowns() const ;
  double value() const;
  void value(std::vector<double> & functionValue);
  void gradient(std::vector<double> & gradient);
  void precondition(std::vector<double>       & s,
			      const std::vector<double> & gradient) const;
  void update(const std::vector<double> & solution);
  void solution(std::vector<double> & solution);
  std::vector<int> getUnknownCountFlag() const;

  //member data
  std::vector<int> d_relaxationFlags;
  double d_maximumAtomForceToBeRelaxed;
  int d_totalUpdateCalls;

  //pointer to dft class
  dftClass<FEOrder>* dftPtr;
  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;
};

#endif
