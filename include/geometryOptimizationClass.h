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
    geometryOptimizationClass(const std::string parameter_file,
                              const std::string restartFilesPath,
                              const MPI_Comm &  mpi_comm_parent,
                              const bool        restart);


    void
    init(const std::string parameter_file);

    /**
     * @brief runOpt:
     *
     *
     */
    void
    runOpt();

  private:
    // pointers to dft class and optimization classes
    std::unique_ptr<dftfeWrapper> d_dftfeWrapper;
    std::unique_ptr<geoOptIon>    d_geoOptIonPtr;
    std::unique_ptr<geoOptCell>   d_geoOptCellPtr;
    dftBase *                     d_dftPtr;

    // restart parameters
    const bool        d_isRestart;
    const std::string d_restartFilesPath;
    // status parameters
    int d_status, d_cycle, d_optMode;
    // parallel communication objects
    const MPI_Comm d_mpiCommParent;

    // conditional stream object
    dealii::ConditionalOStream pcout;
  };
} // namespace dftfe
#endif
