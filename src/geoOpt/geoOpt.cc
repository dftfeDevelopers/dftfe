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

#include "../../include/geoOpt.h"
#include "../../include/force.h"
#include "../../include/dft.h"
#include "../../include/fileReaders.h"


//
//constructor
//
template<unsigned int FEOrder>
geoOpt<FEOrder>::geoOpt(dftClass<FEOrder>* _dftPtr, forceClass<FEOrder>* _forcePtr):
  dftPtr(_dftPtr),
  forcePtr(_forcePtr),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}

//
//
template<unsigned int FEOrder>
void geoOpt<FEOrder>::init()
{

}

template<unsigned int FEOrder>
void geoOpt<FEOrder>::relax()
{

}

template<unsigned int FEOrder>
void geoOpt<FEOrder>::relaxAtomsForces()
{
}

template<unsigned int FEOrder>
void geoOpt<FEOrder>::relaxStress()
{
}

template<unsigned int FEOrder>
void geoOpt<FEOrder>::relaxAtomsForcesStress()
{
}

template class geoOpt<1>;
template class geoOpt<2>;
template class geoOpt<3>;
template class geoOpt<4>;
template class geoOpt<5>;
template class geoOpt<6>;
template class geoOpt<7>;
template class geoOpt<8>;
template class geoOpt<9>;
template class geoOpt<10>;
template class geoOpt<11>;
template class geoOpt<12>;
