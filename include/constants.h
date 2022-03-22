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
// @author Sambit Das and Phani Motamarri
//

#ifndef constants_H_
#define constants_H_

namespace dftfe
{
  //
  // Add prefix C_ to all constants
  //

  /// Boltzmann constant
  const double C_kb = 3.166811429e-06;

  /// 1d quadrature rule order
  template <unsigned int FEOrder>
  constexpr unsigned int
  C_num1DQuad()
  {
    return FEOrder + 1;
  }

  /// 1d quad rule smeared nuclear charge
  constexpr unsigned int
  C_num1DQuadSmearedCharge()
  {
    return 10;
  }

  /// number of copies 1d quad rule smeared nuclear charge
  constexpr unsigned int
  C_numCopies1DQuadSmearedCharge()
  {
    return 2;
  }

  /// 1d quad rule smeared nuclear charge
  /// if a very coarse FE mesh is used (e.g. softer pseudopotentials)
  constexpr unsigned int
  C_num1DQuadSmearedChargeHigh()
  {
    return 10;
  }

  /// number of copies 1d quad rule smeared nuclear charge
  /// if a very coarse FE mesh is used (e.g. softer pseudpotentials)
  constexpr unsigned int
  C_numCopies1DQuadSmearedChargeHigh()
  {
    return 3;
  }

  /// 1d quad rule smeared nuclear charge if cell stress calculation is on
  constexpr unsigned int
  C_num1DQuadSmearedChargeStress()
  {
    return 10;
  }

  /// number of copies 1d quad rule smeared nuclear charge if cell stress
  /// calculation is on
  constexpr unsigned int
  C_numCopies1DQuadSmearedChargeStress()
  {
    return 5;
  }

#ifdef DFTFE_WITH_HIGHERQUAD_PSP
  /// rho nodal polynomial order
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  constexpr unsigned int
  C_rhoNodalPolyOrder()
  {
    return ((FEOrder + 2) > FEOrderElectro ? (FEOrder + 2) : FEOrderElectro);
  }

  /// 1d quadrature rule order for non-local part of pseudopotential
  template <unsigned int FEOrder>
  constexpr unsigned int
  C_num1DQuadNLPSP()
  {
    return 14;
  }

  /// number of copies 1d quad rule non-local PSP
  constexpr unsigned int
  C_numCopies1DQuadNLPSP()
  {
    return 1;
  }

  /// 1d quadrature rule order for local part of pseudopotential
  template <unsigned int FEOrder>
  constexpr unsigned int
  C_num1DQuadLPSP()
  {
    return 14;
  }

  /// number of copies 1d quad rule local PSP
  constexpr unsigned int
  C_numCopies1DQuadLPSP()
  {
    return 1;
  }
#else

  /// rho nodal polynomial order
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  constexpr unsigned int
  C_rhoNodalPolyOrder()
  {
    return ((FEOrder + 2) > FEOrderElectro ? (FEOrder + 2) : FEOrderElectro);
  }

  /// 1d quadrature rule order for non-local part of pseudopotential
  template <unsigned int FEOrder>
  constexpr unsigned int
  C_num1DQuadNLPSP()
  {
    return 10;
  }

  /// number of copies 1d quad rule non-local PSP
  constexpr unsigned int
  C_numCopies1DQuadNLPSP()
  {
    return 1;
  }

  /// 1d quadrature rule order for local part of pseudopotential
  template <unsigned int FEOrder>
  constexpr unsigned int
  C_num1DQuadLPSP()
  {
    return 10;
  }

  /// number of copies 1d quad rule local PSP
  constexpr unsigned int
  C_numCopies1DQuadLPSP()
  {
    return 1;
  }
#endif
} // namespace dftfe
#endif
