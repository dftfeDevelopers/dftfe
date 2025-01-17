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
    const dftfe::linearAlgebra::BLASWrapper<memorySpace> &      BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *X,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &                        basisOperationsPtr,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                  kineticEnergyDensityValues,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const MPI_Comm &     mpiCommDomain,
    const dftParameters &dftParams);

  template <typename NumberType>
  void
  computeKineticEnergyDensityFromInterpolatedValues(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>
      &BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
      &                                         basisOperationsPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    double *                                    kcoord,
    NumberType *                                wfcQuadPointData,
    NumberType *                                gradWfcQuadPointData,
    double *                                    kineticCellsWfcContributions,
    double *                                    kineticEnergyDensity,
    const MPI_Comm &                            mpiCommDomain);

#if defined(DFTFE_WITH_DEVICE)
  template <typename NumberType>
  void
  computeKineticEnergyDensityFromInterpolatedValues(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
      &BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &                                         basisOperationsPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    double *                                    kcoord,
    NumberType *                                wfcQuadPointData,
    NumberType *                                gradWfcQuadPointData,
    double *        kineticEnergyCellsWfcContributions,
    double *        kineticEnergyDensity,
    const MPI_Comm &mpiCommDomain);
#endif

} // namespace dftfe
#endif
