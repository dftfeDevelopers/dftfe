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
#ifndef DFTFE_EXCORRFUNCTIONALEVALUATOR_H
#define DFTFE_EXCORRFUNCTIONALEVALUATOR_H
#include <MemoryStorage.h>
#include <cmath>
#include <XCfunctionalDefs/xc_params.h>

namespace dftfe
{
#define DFTFE_FUNCTIONALEVALUATOR_LDA_X(NAME, BODY)                        \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void LDAX_##NAME(                                                        \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &exEnergyOut,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdexDensity);

#define DFTFE_FUNCTIONALEVALUATOR_LDA_C(NAME, BODY)                        \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void LDAC_##NAME(                                                        \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &corrEnergyOut, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdecDensity);

#define DFTFE_FUNCTIONALEVALUATOR_GGA_X(NAME, BODY)                        \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void GGAX_##NAME(                                                        \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    const dftfe::utils::MemoryStorage<double, memorySpace> &sigmaValues,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &exEnergyOut,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdexDensity,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdexSigma);

#define DFTFE_FUNCTIONALEVALUATOR_GGA_C(NAME, BODY)                        \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void GGAC_##NAME(                                                        \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    const dftfe::utils::MemoryStorage<double, memorySpace> &sigmaValues,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &corrEnergyOut, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdecDensity,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdecSigma);

#define DFTFE_FUNCTIONALEVALUATOR_MGGA_X(NAME, BODY)                       \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void MGGAX_##NAME(                                                       \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    const dftfe::utils::MemoryStorage<double, memorySpace> &sigmaValues,   \
    const dftfe::utils::MemoryStorage<double, memorySpace> &tauValues,     \
    dftfe::utils::MemoryStorage<double, memorySpace>       &exEnergyOut,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdexDensity,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdexSigma,     \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdexTau,       \
    bool                                                    tauNeededX,    \
    bool                                                    enforceFHCX);

#define DFTFE_FUNCTIONALEVALUATOR_MGGA_C(NAME, BODY)                       \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void MGGAC_##NAME(                                                       \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    const dftfe::utils::MemoryStorage<double, memorySpace> &sigmaValues,   \
    const dftfe::utils::MemoryStorage<double, memorySpace> &tauValues,     \
    dftfe::utils::MemoryStorage<double, memorySpace>       &corrEnergyOut, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdecDensity,   \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdecSigma,     \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdecTau,       \
    bool                                                    tauNeededC,    \
    bool                                                    enforceFHCC);

#include <exchangeCorrelationFunctionalEvaluation.def>

} // namespace dftfe


#undef DFTFE_FUNCTIONALEVALUATOR_LDA_X
#undef DFTFE_FUNCTIONALEVALUATOR_LDA_C

#undef DFTFE_FUNCTIONALEVALUATOR_GGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_GGA_C

#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_C
#endif // DFTFE_EXCORRFUNCTIONALEVALUATOR_H
