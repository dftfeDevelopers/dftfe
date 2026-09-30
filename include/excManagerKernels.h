// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_EXCMANAGERDEVICEKERNELS_H
#define DFTFE_EXCMANAGERDEVICEKERNELS_H

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
    fillRhoVector(
      const dftfe::uInt                                       numQuadPoints,
      const dftfe::utils::MemoryStorage<double, memorySpace> &densitySpinUp,
      const dftfe::utils::MemoryStorage<double, memorySpace> &densitySpinDown,
      dftfe::utils::MemoryStorage<double, memorySpace>       &rhoVector);

    template <dftfe::utils::MemorySpace memorySpace>
    void
    fillRhoSigmaVector(
      const dftfe::uInt                                       numQuadPoints,
      const dftfe::utils::MemoryStorage<double, memorySpace> &densitySpinUp,
      const dftfe::utils::MemoryStorage<double, memorySpace> &densitySpinDown,
      const dftfe::utils::MemoryStorage<double, memorySpace> &gradDensitySpinUp,
      const dftfe::utils::MemoryStorage<double, memorySpace>
                                                       &gradDensitySpinDown,
      dftfe::utils::MemoryStorage<double, memorySpace> &rhoVector,
      dftfe::utils::MemoryStorage<double, memorySpace> &sigmaVector);

    template <dftfe::utils::MemorySpace memorySpace>
    void
    fillRhoSigmaTauVector(
      const dftfe::uInt                                       numQuadPoints,
      const dftfe::utils::MemoryStorage<double, memorySpace> &densitySpinUp,
      const dftfe::utils::MemoryStorage<double, memorySpace> &densitySpinDown,
      const dftfe::utils::MemoryStorage<double, memorySpace> &gradDensitySpinUp,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &gradDensitySpinDown,
      const dftfe::utils::MemoryStorage<double, memorySpace> &tauSpinUp,
      const dftfe::utils::MemoryStorage<double, memorySpace> &tauSpinDown,
      dftfe::utils::MemoryStorage<double, memorySpace>       &rhoVector,
      dftfe::utils::MemoryStorage<double, memorySpace>       &sigmaVector,
      dftfe::utils::MemoryStorage<double, memorySpace>       &tauVector,
      const double                                            rhoThreshold,
      const double                                            sigmaThreshold,
      const double                                            tauThreshold);

  } // namespace internal
} // namespace dftfe

#endif // DFTFE_EXCMANAGERDEVICEKERNELS_H
