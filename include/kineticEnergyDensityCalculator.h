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

#ifndef kineticEnergyDensityCalculator_H_
#define kineticEnergyDensityCalculator_H_

#include <headers.h>
#include "dftParameters.h"
#include "FEBasisOperations.h"

namespace dftfe
{
  template <typename NumberType, dftfe::utils::MemorySpace memorySpace>
  void
  computeKineticEnergyDensity(
    const dftfe::linearAlgebra::BLASWrapper<memorySpace>       &BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *X,
    const dftfe::uInt                       totalNumWaveFunctions,
    const std::vector<std::vector<double>> &partialOccupancies,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
                              &basisOperationsPtr,
    const dftfe::uInt          quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                        &kineticEnergyDensityValues,
    const MPI_Comm      &mpiCommParent,
    const MPI_Comm      &interpoolcomm,
    const MPI_Comm      &interBandGroupComm,
    const MPI_Comm      &mpiCommDomain,
    const dftParameters &dftParams);

  template <typename NumberType>
  void
  computeKineticEnergyDensityFromInterpolatedValues(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kcoord,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double                                   *kineticCellsWfcContributions,
    double                                   *kineticEnergyDensity,
    const MPI_Comm                           &mpiCommDomain);

} // namespace dftfe
#endif
