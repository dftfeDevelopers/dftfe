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

#include <excManagerKernels.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherHelpers.h>
#include <BLASWrapper.h>
#ifndef utils_funcs
#  define max(x, y) (((x) > (y)) ? (x) : (y))
#  define abs(x) (((x) > (0)) ? (x) : (-x))
#endif
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
                sigmaVector[3 * index + 0] += gradDensitySpinUp[3 * index + j] *
                                              gradDensitySpinUp[3 * index + j];
                sigmaVector[3 * index + 1] +=
                  gradDensitySpinUp[3 * index + j] *
                  gradDensitySpinDown[3 * index + j];
                sigmaVector[3 * index + 2] +=
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
            rhoVector[2 * index + 0] =
              max(abs(densitySpinUp[index]), rhoThreshold);
            rhoVector[2 * index + 1] =
              max(abs(densitySpinDown[index]), rhoThreshold);
            for (dftfe::uInt j = 0; j < 3; j++)
              {
                sigmaVector[3 * index + 0] += gradDensitySpinUp[3 * index + j] *
                                              gradDensitySpinUp[3 * index + j];
                sigmaVector[3 * index + 1] +=
                  gradDensitySpinUp[3 * index + j] *
                  gradDensitySpinDown[3 * index + j];
                sigmaVector[3 * index + 2] +=
                  gradDensitySpinDown[3 * index + j] *
                  gradDensitySpinDown[3 * index + j];
              }
            sigmaVector[3 * index + 0] =
              max(abs(sigmaVector[3 * index + 0]), sigmaThreshold);
            sigmaVector[3 * index + 2] =
              max(abs(sigmaVector[3 * index + 2]), sigmaThreshold);
            tauVector[2 * index + 0] = max(abs(tauSpinUp[index]), tauThreshold);
            tauVector[2 * index + 1] =
              max(abs(tauSpinDown[index]), tauThreshold);
          }
      },
      const dftfe::uInt numQuadPoints,
      const double     *densitySpinUp,
      const double     *densitySpinDown,
      const double     *gradDensitySpinUp,
      const double     *gradDensitySpinDown,
      const double     *tauSpinUp,
      const double     *tauSpinDown,
      double           *rhoVector,
      double           *sigmaVector,
      double           *tauVector,
      const double      rhoThreshold,
      const double      sigmaThreshold,
      const double      tauThreshold);

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
      const auto *densitySpinUpPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinUp.data());
      const auto *densitySpinDownPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinDown.data());
      auto *rhoVectorPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(rhoVector.data());
      DFTFE_LAUNCH_KERNEL(fillRhoVectorKernel,
                          (numQuadPoints +
                           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numQuadPoints,
                          densitySpinUpPtr,
                          densitySpinDownPtr,
                          rhoVectorPtr);
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
      const auto *densitySpinUpPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinUp.data());
      const auto *densitySpinDownPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinDown.data());
      const auto *gradDensitySpinUpPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinUp.data());
      const auto *gradDensitySpinDownPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinDown.data());
      auto *rhoVectorPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(rhoVector.data());
      auto *sigmaVectorPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(sigmaVector.data());

      DFTFE_LAUNCH_KERNEL(fillRhoSigmaVectorKernel,
                          (numQuadPoints +
                           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numQuadPoints,
                          densitySpinUpPtr,
                          densitySpinDownPtr,
                          gradDensitySpinUpPtr,
                          gradDensitySpinDownPtr,
                          rhoVectorPtr,
                          sigmaVectorPtr);
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
      const double rhoThreshold,
      const double sigmaThreshold,
      const double tauThreshold)
    {
      const auto *densitySpinUpPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinUp.data());
      const auto *densitySpinDownPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(densitySpinDown.data());
      const auto *gradDensitySpinUpPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinUp.data());
      const auto *gradDensitySpinDownPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(gradDensitySpinDown.data());
      const auto *tauSpinUpPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(tauSpinUp.data());
      const auto *tauSpinDownPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(tauSpinDown.data());
      auto *rhoVectorPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(rhoVector.data());
      auto *sigmaVectorPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(sigmaVector.data());
      auto *tauVectorPtr =
        dftfe::utils::makeDataTypeDeviceCompatible(tauVector.data());
      DFTFE_LAUNCH_KERNEL(fillRhoSigmaTauVectorKernel,
                          (numQuadPoints +
                           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numQuadPoints,
                          densitySpinUpPtr,
                          densitySpinDownPtr,
                          gradDensitySpinUpPtr,
                          gradDensitySpinDownPtr,
                          tauSpinUpPtr,
                          tauSpinDownPtr,
                          rhoVectorPtr,
                          sigmaVectorPtr,
                          tauVectorPtr,
                          rhoThreshold,
                          sigmaThreshold,
                          tauThreshold);
    }
  }; // namespace internal
} // namespace dftfe
