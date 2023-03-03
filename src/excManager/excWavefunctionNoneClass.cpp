// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Vishal Subramanian
//

#include <excWavefunctionNoneClass.h>

namespace dftfe
{
  excWavefunctionNoneClass::excWavefunctionNoneClass(
    densityFamilyType densityFamilyTypeObj,
    xc_func_type *    funcXPtr,
    xc_func_type *    funcCPtr,
    double            factorForWavefunctionDependent,
    bool              scaleExchange,
    bool              computeCorrelation,
    double            scaleExchangeFactor)
    : excWavefunctionBaseClass(densityFamilyTypeObj,
                               funcXPtr,
                               funcCPtr,
                               factorForWavefunctionDependent,
                               scaleExchange,
                               computeCorrelation,
                               scaleExchangeFactor)
  {
    d_wavefunctionFamilyType = wavefunctionFamilyType::NONE;
  }
  void
  excWavefunctionNoneClass::applyWaveFunctionDependentVxc() const
  {
    // Do nothing
  }
  void
  excWavefunctionNoneClass::updateWaveFunctionDependentVxc() const
  {
    // Do nothing
  }
  double
  excWavefunctionNoneClass::computeWaveFunctionDependentExcEnergy() const
  {
    // Do nothing
  }

} // namespace dftfe
