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

#ifndef densityCalculatorDeviceKernels_H_
#define densityCalculatorDeviceKernels_H_
#if defined(DFTFE_WITH_DEVICE)

#  include <BLASWrapper.h>
#  include <DataTypeOverloads.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceTypeConfig.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <memory>
namespace dftfe
{
  template <typename NumberType>
  void
  computeRhoGradRhoFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                               &BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    const unsigned int                          nQuadsPerCell,
    double                                     *partialOccupVec,
    NumberType                                 *wfcQuadPointData,
    NumberType                                 *gradWfcQuadPointData,
    double                                     *rhoCellsWfcContributions,
    double                                     *gradRhoCellsWfcContributions,
    double                                     *rho,
    double                                     *gradRho,
    const bool                                  isEvaluateGradRho);

  template <typename NumberType>
  void
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                               &BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    const unsigned int                          nQuadsPerCell,
    double                                     *onesVec,
    double                                     *partialOccupVecPrime,
    NumberType                                 *wfcQuadPointData,
    NumberType                                 *wfcPrimeQuadPointData,
    double *rhoResponseHamCellsWfcContributions,
    double *rhoResponseFermiEnergyCellsWfcContributions,
    double *rhoResponseHam,
    double *rhoResponseFermiEnergy);

  template <typename NumberType>
  void
  computeKineticEnergyDensityFromInterpolatedValues(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
                                               &BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    const unsigned int                          nQuadsPerCell,
    double                                     *partialOccupVec,
    double                                     *kcoord,
    NumberType                                 *wfcQuadPointData,
    NumberType                                 *gradWfcQuadPointData,
    double         *kineticEnergyCellsWfcContributions,
    double         *kineticEnergyDensity,
    const MPI_Comm &mpiCommDomain);

} // namespace dftfe
#endif
#endif
