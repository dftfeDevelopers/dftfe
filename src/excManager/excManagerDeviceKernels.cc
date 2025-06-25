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

#include <excManagerDeviceKernels.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherHelpers.h>
#include <BLASWrapper.h>
namespace dftfe
{
  namespace internal
  {
    DFTFE_CREATE_KERNEL(
      void,
      fillRhoVectorKernel,
      {
        const dftfe::uInt numberEntries = numQuadPoints;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            rhoVector[2 * index + 0] = densitySpinUp[index];
            rhoVector[2 * index + 1] = densitySpinDown[index];
          }
      },
      const dftfe::uInt numQuadPoints,
      const double     *densitySpinUp,
      const double     *densitySpinDown,
      double           *rhoVector);


    DFTFE_CREATE_KERNEL(
      void,
      fillRhoSigmaVectorKernel,
      {
        const dftfe::uInt numberEntries = numQuadPoints;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            rhoVector[2 * index + 0] = densitySpinUp[index];
            rhoVector[2 * index + 1] = densitySpinDown[index];
            for (dftfe::uInt j = 0; j < 3; j++)
              {
                sigmaVector[3 * index + 0] = gradDensitySpinUp[3 * index + j] *
                                             gradDensitySpinUp[3 * index + j];
                sigmaVector[3 * index + 1] = gradDensitySpinUp[3 * index + j] *
                                             gradDensitySpinDown[3 * index + j];
                sigmaVector[3 * index + 2] =
                  gradDensitySpinDown[3 * index + j] *
                  gradDensitySpinDown[3 * index + j];
              }
          }
      },
      const dftfe::uInt numQuadPoints,
      const double     *densitySpinUp,
      const double     *densitySpinDown,
      const double     *gradDensitySpinUp,
      const double     *gradDensitySpinDown,
      double           *rhoVector,
      double           *sigmaVector);



    DFTFE_CREATE_KERNEL(
      void,
      fillRhoSigmaTauVectorKernel,
      {
        const dftfe::uInt numberEntries = numQuadPoints;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            rhoVector[2 * index + 0] = densitySpinUp[index];
            rhoVector[2 * index + 1] = densitySpinDown[index];
            for (dftfe::uInt j = 0; j < 3; j++)
              {
                sigmaVector[3 * index + 0] = gradDensitySpinUp[3 * index + j] *
                                             gradDensitySpinUp[3 * index + j];
                sigmaVector[3 * index + 1] = gradDensitySpinUp[3 * index + j] *
                                             gradDensitySpinDown[3 * index + j];
                sigmaVector[3 * index + 2] =
                  gradDensitySpinDown[3 * index + j] *
                  gradDensitySpinDown[3 * index + j];
              }
            tauVector[2 * index + 0] = max(tauSpinUp[index], tauMax);
            tauVector[2 * index + 1] = max(tauSpinDown[index], tauMax);
          }
      },
      const dftfe::uInt numQuadPoints,
      const double      tauMax,
      const double     *densitySpinUp,
      const double     *densitySpinDown,
      const double     *gradDensitySpinUp,
      const double     *gradDensitySpinDown,
      const double     *tauSpinUp,
      const double     *tauSpinDown,
      double           *rhoVector,
      double           *sigmaVector,
      double           *tauVector);

    template <>
    void
    fillRhoVector(
      const dftfe::uInt numQuadPoints,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &densitySpinUp,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &densitySpinDown,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &rhoVector)
    {
      DFTFE_LAUNCH_KERNEL(
        fillRhoVectorKernel,
        (numQuadPoints + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numQuadPoints,
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinUp.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinDown.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(rhoVector.data()));
    }

    template <>
    void
    fillRhoSigmaVector(
      const dftfe::uInt numQuadPoints,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &densitySpinUp,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &densitySpinDown,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradDensitySpinUp,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradDensitySpinDown,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &rhoVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &sigmaVector)
    {
      DFTFE_LAUNCH_KERNEL(
        fillRhoSigmaVectorKernel,
        (numQuadPoints + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numQuadPoints,
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinUp.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinDown.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinUp.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinDown.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(rhoVector.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(sigmaVector.data()));
    }

    template <>
    void
    fillRhoSigmaTauVector(
      const dftfe::uInt numQuadPoints,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &densitySpinUp,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &densitySpinDown,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradDensitySpinUp,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradDensitySpinDown,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tauSpinUp,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tauSpinDown,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &rhoVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &sigmaVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
                  &tauVector,
      const double tauThreshold)
    {
      DFTFE_LAUNCH_KERNEL(
        fillRhoSigmaTauVectorKernel,
        (numQuadPoints + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numQuadPoints,
        tauThreshold,
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinUp.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinDown.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinUp.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinDown.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(tauSpinUp.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(tauSpinDown.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(rhoVector.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(sigmaVector.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(tauVector.data()));
    }
  }; // namespace internal
} // namespace dftfe
