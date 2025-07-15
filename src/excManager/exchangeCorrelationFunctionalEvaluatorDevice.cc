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

#include <exchangeCorrelationFunctionalEvaluator.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherHelpers.h>
#include <BLASWrapper.h>



namespace dftfe
{


  namespace
  {

#define DFTFE_FUNCTIONALEVALUATOR_LDA_X(NAME, BODY)    \
  DFTFE_CREATE_KERNEL(void,                            \
                      exchangeEvaluationKernel##NAME,  \
                      BODY,                            \
                      const dftfe::uInt numPoints,     \
                      const double     *densityValues, \
                      double           *excEnergyOut,  \
                      double           *pdexDensity);

#define DFTFE_FUNCTIONALEVALUATOR_LDA_C(NAME, BODY)      \
  DFTFE_CREATE_KERNEL(void,                              \
                      correlationEvaluationKernel##NAME, \
                      BODY,                              \
                      const dftfe::uInt numPoints,       \
                      const double     *densityValues,   \
                      double           *corrEnergyOut,   \
                      double           *pdecDensity);

  } // namespace
#include <exchangeCorrelationFunctionalEvaluation.def>
} // namespace dftfe

#undef DFTFE_FUNCTIONALEVALUATOR_LDA_X
#undef DFTFE_FUNCTIONALEVALUATOR_LDA_C
namespace dftfe
{
#define DFTFE_FUNCTIONALEVALUATOR_LDA_X(NAME, BODY)                        \
  template <>                                                              \
  void LDAX_##NAME(                                                        \
    dftfe::uInt numPoints,                                                 \
    const dftfe::utils::MemoryStorage<double,                              \
                                      dftfe::utils::MemorySpace::DEVICE>   \
      &densityValues,                                                      \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> \
      &excEnergyOut,                                                       \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> \
      &pdexDensity)                                                        \
  {                                                                        \
    DFTFE_LAUNCH_KERNEL(                                                   \
      exchangeEvaluationKernel##NAME,                                      \
      (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) /                  \
        dftfe::utils::DEVICE_BLOCK_SIZE,                                   \
      dftfe::utils::DEVICE_BLOCK_SIZE,                                     \
      dftfe::linearAlgebra::BLASWrapper<                                   \
        dftfe::utils::MemorySpace::DEVICE>::d_streamId,                    \
      numPoints,                                                           \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data()),    \
      dftfe::utils::makeDataTypeDeviceCompatible(excEnergyOut.data()),     \
      dftfe::utils::makeDataTypeDeviceCompatible(pdexDensity.data()));     \
  }


#define DFTFE_FUNCTIONALEVALUATOR_LDA_C(NAME, BODY)                        \
  template <>                                                              \
  void LDAC_##NAME(                                                        \
    dftfe::uInt numPoints,                                                 \
    const dftfe::utils::MemoryStorage<double,                              \
                                      dftfe::utils::MemorySpace::DEVICE>   \
      &densityValues,                                                      \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> \
      &corrEnergyOut,                                                      \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> \
      &pdecDensity)                                                        \
  {                                                                        \
    DFTFE_LAUNCH_KERNEL(                                                   \
      correlationEvaluationKernel##NAME,                                   \
      (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) /                  \
        dftfe::utils::DEVICE_BLOCK_SIZE,                                   \
      dftfe::utils::DEVICE_BLOCK_SIZE,                                     \
      dftfe::linearAlgebra::BLASWrapper<                                   \
        dftfe::utils::MemorySpace::DEVICE>::d_streamId,                    \
      numPoints,                                                           \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data()),    \
      dftfe::utils::makeDataTypeDeviceCompatible(corrEnergyOut.data()),    \
      dftfe::utils::makeDataTypeDeviceCompatible(pdecDensity.data()));     \
  }
#include <exchangeCorrelationFunctionalEvaluation.def>
} // namespace dftfe

#undef DFTFE_FUNCTIONALEVALUATOR_LDA_X
#undef DFTFE_FUNCTIONALEVALUATOR_LDA_C
