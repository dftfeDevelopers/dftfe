// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_FUNCTIONALTEST_H
#define DFTFE_FUNCTIONALTEST_H

#include "headers.h"
#include "BLASWrapper.h"
#include "FEBasisOperations.h"
#include "dftParameters.h"

namespace functionalTest
{
  void
  testTransferFromParentToChildIncompatiblePartitioning(
    const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
      dftfe::utils::MemorySpace::HOST>>     BLASWrapperPtr,
    const MPI_Comm &                        mpi_comm_parent,
    const MPI_Comm &                        mpi_comm_domain,
    const MPI_Comm &                        interpoolcomm,
    const MPI_Comm &                        interbandgroup_comm,
    const unsigned int                      FEOrder,
    const dftfe::dftParameters &            dftParams,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imageAtomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<double> &             nearestAtomDistances,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria,
    const bool                              generateElectrostaticsTria);

  // template <dftfe::utils::MemorySpace memorySpace>
  void
  testMultiVectorPoissonSolver(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                            basisOperationsPtr,
    dealii::MatrixFree<3, double> &matrixFreeData,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                            BLASWrapperPtr,
    std::vector<const dealii::AffineConstraints<double> *> &constraintMatrixVec,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                inputVec,
    const unsigned int matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhsDensity,
    const unsigned int matrixFreeQuadratureComponentAX,
    const unsigned int verbosity,
    const MPI_Comm &   mpi_comm_parent,
    const MPI_Comm &   mpi_comm_domain);

  void
  testAccumulateInsert(const MPI_Comm &mpiComm);

} // end of namespace functionalTest

#endif // DFTFE_FUNCTIONALTEST_H
