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
// @author  Vishal Subramanian, Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONBASE_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONBASE_H

#include <vector>
#include <boost/math/quadrature/gauss_kronrod.hpp>
namespace dftfe
{
  class AtomCenteredSphericalFunctionBase
  {
  public:
    /**
     * @brief Computes the Radial Value of the Function at distance r
     * @param[in] r radial distance
     * @return function value at distance r
     */
    virtual double
    getRadialValue(double r) const = 0;

    // The following functions need not be re-defined in the
    // derived classes. So it is being defined in this class
    /**
     * @brief returns the l-quantum number associated with the spherical function
     * @return Quantum number l
     */
    unsigned int
    getQuantumNumberl() const;

    /**
     * @brief COmputes the Radial-Integral value
     * @return Result of the radial-integration of the spherical function from d_rmin to d_cutOff
     */
    double
    getIntegralValue() const;
    /**
     * @brief Returns the maximum radial distance
     * @return Cutoff distance
     */
    double
    getRadialCutOff() const;
    /**
     * @brief Checks if the data is present
     * @return True if function is present, false if not.
     */
    bool
    isDataPresent() const;
    /**
     * @brief Computes the Radial Value, Radial-deriative and Radial-second derivative of the Function at distance r
     * @param[in] r radial distance
     * @return vector of size 3 comprising of function value, Radial-derivative value and Radial-secon derivative at distance r.
     */

    virtual std::vector<double>
    getDerivativeValue(double r) const = 0;

  protected:
    double       d_cutOff;
    unsigned int d_lQuantumNumber;
    bool         d_DataPresent;


  }; // end of class AtomCenteredSphericalFunctionBase
} // end of namespace dftfe
#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONBASE_H
