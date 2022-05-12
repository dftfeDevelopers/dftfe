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
//
// @author Nikhil Kodali
//

#ifndef geometryOptimizationClass_H_
#define geometryOptimizationClass_H_
#include "constants.h"
#include "headers.h"
#include "dftBase.h"
#include "dftfeWrapper.h"
#include <geoOptCell.h>
#include <geoOptIon.h>

namespace dftfe
{
  using namespace dealii;
  class geometryOptimizationClass
  {
  public:
    /**
     * @brief geometryOptimizationClass constructor: copy data from dftparameters to the memebrs of molecularDynamicsClass
     *
     *
     *  @param[in] dftBase *_dftBasePtr pointer to base class of dftClass
     *  @param[in] mpi_comm_parent parent mpi communicator
     */
    geometryOptimizationClass(dftfeWrapper &  dftfeWrapper,
                              const MPI_Comm &mpi_comm_parent);


    /**
     * @brief runOpt:
     *
     *
     */
    void
    runOpt();

  private:
    // pointer to dft class
    dftBase *   d_dftPtr;
    geoOptIon * geoOptIonPtr;
    geoOptCell *geoOptCellPtr;

    // parallel communication objects
    const MPI_Comm d_mpiCommParent;

    // conditional stream object
    dealii::ConditionalOStream pcout;
  };
} // namespace dftfe
#endif
