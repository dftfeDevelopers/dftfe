// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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


#ifndef MatrixFreeBase_H_
#define MatrixFreeBase_H_
#include <FEBasisOperations.h>

namespace dftfe
{
  /**
   * @brief MatrixFreeBase class.
   *
   * @author Gourab Panigrahi
   *
   */
  class MatrixFreeBase
  {
  public:
    virtual ~MatrixFreeBase(){};

    /**
     * @brief reinitialize data structures for matrixfree
     *
     */
    virtual void
    reinit(const dealii::MatrixFree<3, double> *matrixFreeDataPtr,
           const unsigned int &                 dofHandlerID,
           const int                            matrixFreeQuadratureID) = 0;


    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    virtual void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x) = 0;

    virtual void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x,
              double                        coeffHelmholtz) = 0;
  };

} // namespace dftfe
#endif // MatrixFreeBase_H_
