// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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

namespace dftfe {
	//
	//Add prefix C_ to all constants
	//

	/// Boltzmann constant
	const double C_kb = 3.166811429e-06;

	/// problem space dimensions
	const int C_DIM = 3;

	/// 1d quadrature rule order
	template <unsigned int FEOrder> constexpr unsigned int C_num1DQuad(){return FEOrder+8;}

	//kerker Helmholtz solve polynomial Order
	template <unsigned int FEOrder> constexpr unsigned int C_num1DKerkerPoly(){return FEOrder;}


	/// 1d quadrature rule order for non-local part of pseudopotential
	template <unsigned int FEOrder> constexpr unsigned int C_num1DQuadPSP()
	{return FEOrder+4 > 8 ? 8 : FEOrder+4;}
}
#endif
