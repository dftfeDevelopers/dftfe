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
// @author Sambit Das and Phani Motamarri
//

#ifndef constants_H_
#define constants_H_
#include <dftfe/config.h>
#include <dftfe/TypeConfig.h>

namespace dftfe
{
  //
  // Add prefix C_ to all constants
  //

  /// Boltzmann constant
  const double C_kb                  = 3.166811429e-06;
  const double C_haPerBohrToeVPerAng = 27.211386245988 / 0.529177210903;
  const double C_haToeV              = 27.211386245988;
  const double C_bohrToAng           = 0.529177210903;
  const double C_pi                  = 3.14159265359;
  const double C_AngToBohr           = 1.0 / 0.529177210903;

  /// 1d quadrature rule order
  inline dftfe::uInt
  C_num1DQuad(dftfe::uInt FEOrder)
  {
    return FEOrder + 1;
  }

} // namespace dftfe
#endif
