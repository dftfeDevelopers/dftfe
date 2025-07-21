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
// @author Kartick Ramakrishnan, Nikhil Kodali
//
#include <KohnShamDFTOperatorDeviceKernels.h>
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

    template <>
    void
    computeVeffJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &phiVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &VeffJxW)
    {
      // Not yet implemented
    }
    template <>
    void
    computeInvJacderExcWithSigmaTimesGradRhoJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          spinIndex,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &invJacobianEntries,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradientRhoSpinIndex,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradientRhoOtherSpinIndex,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &invJacderExcWithSigmaTimesGradRhoJxW)
    {
      // Not yet implemented
    }

    template <>
    void
    computeHalfInvJacinvJacderExcWithTauJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &invJacinvJacderExcWithTauJxW)
    {
      // Not yet implemented
    }

    template <>
    void
    computeKPointDependenderExcWithTauJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::uInt                         offset,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &kPointCoordinate,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &halfKSquareTimesDerExcwithTauJxW,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &invJacKpointTimesderExcwithTauJxW)
    {
      // Not yet implemented
    }
  }; // namespace internal
} // namespace dftfe
