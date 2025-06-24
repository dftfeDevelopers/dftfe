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

#ifndef kohnShamHamiltonianOperatorDeviceKernels_H_
#define kohnShamHamiltonianOperatorDeviceKernels_H_


#include <DataTypeOverloads.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherHelpers.h>
#include <MemoryStorage.h>
#include <memory>
namespace dftfe
{
  namespace internal
  {
    template <dftfe::utils::MemorySpace memorySpace>
    void
    computeVeffJxWEntries(
      const std::pair<unsigned int, unsigned int>             cellRange,
      const unsigned int                                      numQuadsPerCell,
      const dftfe::utils::MemoryStorage<double, memorySpace> &phiVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdecVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdexVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &jxwVector,
      dftfe::utils::MemoryStorage<double, memorySpace>       &VeffJxW);

    template <dftfe::utils::MemorySpace memorySpace>
    void
    computeInvJacderExcWithSigmaTimesGradRhoJxWEntries(
      const std::pair<unsigned int, unsigned int>             cellRange,
      const unsigned int                                      numQuadsPerCell,
      const dftfe::Int                                        spinIndex,
      const dftfe::Int                                        cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdecVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdexVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &jxwVector,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &invJacobianEntries,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &gradientRhoSpinIndex,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &gradientRhoOtherSpinIndex,
      dftfe::utils::MemoryStorage<double, memorySpace>
        &invJacderExcWithSigmaTimesGradRhoJxW);

    template <dftfe::utils::MemorySpace memorySpace>
    void
    computeHalfInvJacinvJacderExcWithTauJxWEntries(
      const std::pair<unsigned int, unsigned int>             cellRange,
      const unsigned int                                      numQuadsPerCell,
      const dftfe::Int                                        cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdecVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdexVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &jxwVector,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, memorySpace>
        &invJacinvJacderExcWithTauJxW);
    template <dftfe::utils::MemorySpace memorySpace>
    void
    computeKPointDependenderExcWithTauJxWEntries(
      const std::pair<unsigned int, unsigned int>             cellRange,
      const unsigned int                                      numQuadsPerCell,
      const dftfe::Int                                        cellsTypeFlag,
      const dftfe::uInt                                       offset,
      const dftfe::utils::MemoryStorage<double, memorySpace> &kPointCoordinate,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdecVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &pdexVector,
      const dftfe::utils::MemoryStorage<double, memorySpace> &jxwVector,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, memorySpace>
        &halfKSquareTimesDerExcwithTauJxW,
      dftfe::utils::MemoryStorage<double, memorySpace>
        &invJacKpointTimesderExcwithTauJxW);
  }; // namespace internal
} // namespace dftfe
#endif
