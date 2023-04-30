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

#include <excDensityBaseClass.h>

namespace dftfe
{
  excDensityBaseClass::excDensityBaseClass(xc_func_type *funcXPtr,
                                           xc_func_type *funcCPtr,
                                           bool          isSpinPolarized,
                                           bool          scaleExchange,
                                           bool          computeCorrelation,
                                           double        scaleExchangeFactor)
    : d_funcXPtr(funcXPtr)
    , d_funcCPtr(funcCPtr)
    , d_isSpinPolarized(isSpinPolarized)
    , d_scaleExchange(scaleExchange)
    , d_computeCorrelation(computeCorrelation)
    , d_scaleExchangeFactor(scaleExchangeFactor)
  {}


  densityFamilyType
  excDensityBaseClass::getDensityBasedFamilyType() const
  {
    return d_familyType;
  }

} // namespace dftfe
