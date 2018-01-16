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

#include "../../include/geoOptIon.h"
#include "../../include/cgPRPNonLinearSolver.h"
#include "../../include/force.h"
#include "../../include/dft.h"


//
//constructor
//
template<unsigned int FEOrder>
geoOptIon<FEOrder>::geoOptIon(dftClass<FEOrder>* _dftPtr):
  dftPtr(_dftPtr),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}

//
//
template<unsigned int FEOrder>
void geoOptIon<FEOrder>::init()
{

}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::run()
{
   cgPRPNonLinearSolver cgSolver(5e-6,100,1);
   cgSolver.solve(*this);
}


template<unsigned int FEOrder>
int geoOptIon<FEOrder>::getNumberUnknowns() const
{
}



template<unsigned int FEOrder>
double geoOptIon<FEOrder>::value() const
{
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::value(std::vector<double> & functionValue)
{
}


template<unsigned int FEOrder>
void geoOptIon<FEOrder>::gradient(std::vector<double> & gradient)
{
}


template<unsigned int FEOrder>
void geoOptIon<FEOrder>::precondition(std::vector<double>       & s,
			              const std::vector<double> & gradient) const
{
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::update(const std::vector<double> & solution)
{
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::solution(std::vector<double> & solution)
{
}

template<unsigned int FEOrder>
std::vector<int>  geoOptIon<FEOrder>::getUnknownCountFlag() const
{
}


template class geoOptIon<1>;
template class geoOptIon<2>;
template class geoOptIon<3>;
template class geoOptIon<4>;
template class geoOptIon<5>;
template class geoOptIon<6>;
template class geoOptIon<7>;
template class geoOptIon<8>;
template class geoOptIon<9>;
template class geoOptIon<10>;
template class geoOptIon<11>;
template class geoOptIon<12>;
