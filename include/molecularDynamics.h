// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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

#ifndef molecularDynamics_H_
#define molecularDynamics_H_
#include "constants.h"
#include "headers.h"

namespace dftfe
{
  using namespace dealii;
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class dftClass;

  /** @file molecularDynamics.h
   *
   *  @author Sambit Das
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class molecularDynamics
  {
  public:
    molecularDynamics(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
                      const MPI_Comm &                   mpi_comm_parent,
                      const MPI_Comm &                   mpi_comm_domain);


    void
    run();

    void
    timingRun();


  private:
    /// pointer to dft class
    dftClass<FEOrder, FEOrderElectro> *dftPtr;

    /// parallel communication objects
    const MPI_Comm     d_mpiCommParent;
    const MPI_Comm     mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    /// conditional stream object
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif
