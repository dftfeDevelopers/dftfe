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


namespace dftfe
{
#define DFTFE_FUNCTIONALEVALUATOR_LDA_X(NAME, BODY)                        \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void LDAX_##NAME(                                                        \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &excEnergyOut,  \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdexDensity);

#define DFTFE_FUNCTIONALEVALUATOR_LDA_C(NAME, BODY)                        \
  template <dftfe::utils::MemorySpace memorySpace>                         \
  void LDAC_##NAME(                                                        \
    dftfe::uInt                                             numPoints,     \
    const dftfe::utils::MemoryStorage<double, memorySpace> &densityValues, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &corrEnergyOut, \
    dftfe::utils::MemoryStorage<double, memorySpace>       &pdecDensity);
#include <exchangeCorrelationFunctionalEvaluation.def>

} // namespace dftfe


#undef DFTFE_FUNCTIONALEVALUATOR_LDA_X
#undef DFTFE_FUNCTIONALEVALUATOR_LDA_C
#endif // DFTFE_EXCORRFUNCTIONALEVALUATOR_H
