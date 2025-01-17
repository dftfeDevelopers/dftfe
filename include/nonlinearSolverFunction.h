// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
//
// @authors Bikash Kanungo, Vishal Subramanian
//

#ifndef DFTFE_NONLINEARSOLVERFUNCTION_H
#define DFTFE_NONLINEARSOLVERFUNCTION_H

#include <headers.h>

namespace dftfe
{
  class nonlinearSolverFunction
  {
  public:
    virtual ~nonlinearSolverFunction() = default;
    virtual void
    setInitialGuess(const std::vector<distributedCPUVec<double>> &x) = 0;

    virtual std::vector<distributedCPUVec<double>>
    getInitialGuess() const = 0;

    virtual void
    getForceVector(const std::vector<distributedCPUVec<double>> &x,
                   std::vector<distributedCPUVec<double>> &      force,
                   std::vector<double> &                         loss) = 0;

    virtual void
    setSolution(const std::vector<distributedCPUVec<double>> &x) = 0;
  }; // end of class nonlinearSolverFunction
} // end of namespace dftfe

#endif // DFTFE_NONLINEARSOLVERFUNCTION_H
