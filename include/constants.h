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
#include <TypeConfig.h>

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

  /// 1d quad rule smeared nuclear charge
  constexpr dftfe::uInt
  C_num1DQuadSmearedCharge()
  {
    return 10;
  }

  /// number of copies 1d quad rule smeared nuclear charge
  constexpr dftfe::uInt
  C_numCopies1DQuadSmearedCharge()
  {
    return 2; // can be changed from 2 to 3
  }

  /// 1d quad rule smeared nuclear charge
  /// if a very coarse FE mesh is used (e.g. softer pseudopotentials)
  constexpr dftfe::uInt
  C_num1DQuadSmearedChargeHigh()
  {
    return 10;
  }

  /// number of copies 1d quad rule smeared nuclear charge
  /// if a very coarse FE mesh is used (e.g. softer pseudpotentials)
  constexpr dftfe::uInt
  C_numCopies1DQuadSmearedChargeHigh()
  {
    return 3;
  }

  /// 1d quad rule smeared nuclear charge if cell stress calculation is on
  constexpr dftfe::uInt
  C_num1DQuadSmearedChargeStress()
  {
    return 10;
  }

  /// number of copies 1d quad rule smeared nuclear charge if cell stress
  /// calculation is on
  constexpr dftfe::uInt
  C_numCopies1DQuadSmearedChargeStress()
  {
    return 5; //
  }

#ifdef DFTFE_WITH_HIGHERQUAD_PSP


  /// 1d quadrature rule order for non-local part of pseudopotential
  inline dftfe::uInt
  C_num1DQuadNLPSP(dftfe::uInt FEOrder)
  {
    return 14; // Can be changed from 14 to 18 Step 1
  }

  /// number of copies 1d quad rule non-local PSP
  inline dftfe::uInt
  C_numCopies1DQuadNLPSP()
  {
    return 1;
  }

  /// 1d quadrature rule order for local part of pseudopotential
  inline dftfe::uInt
  C_num1DQuadLPSP(dftfe::uInt FEOrder)
  {
    return 14;
  }

  /// number of copies 1d quad rule local PSP
  constexpr dftfe::uInt
  C_numCopies1DQuadLPSP()
  {
    return 1;
  }
#else



  /// 1d quadrature rule order for non-local part of pseudopotential
  inline dftfe::uInt
  C_num1DQuadNLPSP(dftfe::uInt FEOrder)
  {
    return 10;
  }

  /// number of copies 1d quad rule non-local PSP
  constexpr dftfe::uInt
  C_numCopies1DQuadNLPSP()
  {
    return 1;
  }

  /// 1d quadrature rule order for local part of pseudopotential
  inline dftfe::uInt
  C_num1DQuadLPSP(dftfe::uInt FEOrder)
  {
    return 10;
  }

  /// number of copies 1d quad rule local PSP
  constexpr dftfe::uInt
  C_numCopies1DQuadLPSP()
  {
    return 1;
  }

#endif
} // namespace dftfe
#endif
