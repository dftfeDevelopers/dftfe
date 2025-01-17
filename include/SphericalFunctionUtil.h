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
// @author Bikash Kanungo
//

#ifndef DFTFE_SPHERICALFUNCTIONUTIL_H
#define DFTFE_SPHERICALFUNCTIONUTIL_H

#include <vector>

namespace dftfe
{
  namespace utils
  {
    namespace sphUtils
    {
      /*
       * @brief Function to convert cartesian coordinates into spherical coordinates
       * @param[in] x Vector of size 3 containing the cartesian coordinates of
       * the point
       * @param[out] r Stores the computed radius of the point
       * @param[out]theta Stores the computed polar angle of the point
       * @param[out] phiStores the computed azimuthal angle of the point
       * @param[in] rTol Defines a tolerance for the radius,
       *            below which theta and phi are set to zero.
       *            To elaborate, for radius->0, the angles are undefined.
       *            We set them to zero as a convenient choice
       * @param[in] angleTol Defines a tolerance for the polar angle (theta)
       *            approching the poles, below which the azimuthal angle is
       *            set to zero. To elaborate, for a point on the pole
       *            (theta = 0 or theta = pi), the azimuthal angle is undefined.
       *            We set phi to zero if |theta - 0| < angleTol or |theta - pi|
       * < angleTol.
       */
      void
      convertCartesianToSpherical(const std::vector<double> &x,
                                  double &                   r,
                                  double &                   theta,
                                  double &                   phi,
                                  const double               rTol,
                                  const double               angleTol);

      double
      Clm(const int l, const int m);

      /*
       * @brief Function to compute the azimuthal angle (phi) dependent
       *        part of the real spherical harmonics. It does not include
       *        any normalization constant
       *        Q(m,phi) =  cos(m*phi) if m > 0
       *        Qm(m,phi) = 1 , if m = 0
       *        Qm(m, phi = sin(|m|phi), if m < 0
       * @param[in] m order of the spherical harmonic (m-quantum number) for
       * which Qm is to be evaluated
       * @param[in] phi Azimuthal angle at which Qm is to be evaluated
       * @return Value of the Qm function
       */
      double
      Qm(const int m, const double phi);

      /*
       * @brief Function to compute derivative of the Qm(m,phi) function (defined above) with respect to phi.
       * @param[in] m order of the spherical harmonic (m-quantum number) for
       * which the derivative of Qm is to be evaluated
       * @param[in] phi Azimuthal angle at which the derivative of Qm is to be
       * evaluated
       * @return Value of the derivative of Qm with respect to phi function
       */
      double
      dQmDPhi(const int m, const double phi);

      /*
       * @brief Function to compute double derivative of the Qm(m,phi) function (defined above) with respect to phi.
       * @param[in] m order of the spherical harmonic (m-quantum number) for
       * which the double derivative of Qm is to be evaluated
       * @param[in] phi Azimuthal angle at which the double derivative of Qm is
       * to be evaluated
       * @return Value of the double derivative of Qm with respect to phi function
       */
      double
      d2QmDPhi2(const int m, const double phi);

      /*
       * @brief Function to compute the polar angle (theta) dependent
       *        part of the real spherical harmonics. Given the degree l and
       * order m (i.e., l and m quantum numbers), this amounts to just the
       *        P_{l,|m|}, that is the associated Legendre function evaluated at
       * |m|. It does not include any normalization constant or Condon-Shockley
       * factor.
       * @param[in] l degree of the spherical harmonic (l-quantum number)
       * @param[in] m order of the spherical harmonic (m-quantum number)
       * @param[in] theta Polar angle
       * @return Value of the P_{l,|m|}
       */
      double
      Plm(const int l, const int m, const double theta);

      /*
       * @brief Function to compute the derivative of the P_{l,|m|} function (defined above)
       *        with respect to the polar angle (theta).
       * @param[in] l degree of the spherical harmonic (l-quantum number)
       * @param[in] m order of the spherical harmonic (m-quantum number)
       * @param[in] theta Polar angle
       * @return Value of the derivative of P_{l,|m|} with respect to theta
       */
      double
      dPlmDTheta(const int l, const int m, const double theta);

      /*
       * @brief Function to compute the double derivative of the P_{l,|m|} function (defined above)
       *        with respect to the polar angle (theta).
       * @param[in] l degree of the spherical harmonic (l-quantum number)
       * @param[in] m order of the spherical harmonic (m-quantum number)
       * @param[in] theta Polar angle
       * @return Value of the double derivative of P_{l,|m|} with respect to theta
       */
      double
      d2PlmDTheta2(const int l, const int m, const double theta);

      /*
       * @brief Function to evaluate the real spherical harmonics YlmReal for a
       *        given degree (l), order (m), polar angle (theta), azimuthal
       * angle (phi)
       * @param[in] l degree of the spherical harmonic (l-quantum number)
       * @param[in] m order of the spherical harmonic (m-quantum number)
       * @param[in] theta Polar angle
       * @param[in] phi Azimuthal angle
       * @return Value of YlmReal
       */
      double
      YlmReal(const int l, const int m, const double theta, const double phi);

      /*
       * @brief Function to evaluate the parial derivatives of the YlmReal function defined above with respect to
       * polar angle (theta) and azimuthal angle (phi).
       * @param[in] l degree of the spherical harmonic (l-quantum number)
       * @param[in] m order of the spherical harmonic (m-quantum number)
       * @param[in] theta Polar angle
       * @param[in] phi Azimuthal angle
       * @return Vector containing the partial derivatives of YlmReal with respect to theta and phi, in that order.
       */
      std::vector<double>
      dYlmReal(const int l, const int m, const double theta, const double phi);

      /*
       * @brief Function to evaluate the second-order parial derivatives of the YlmReal function defined above with respect to
       * polar angle (theta) and azimuthal angle (phi).
       * @param[in] l degree of the spherical harmonic (l-quantum number)
       * @param[in] m order of the spherical harmonic (m-quantum number)
       * @param[in] theta Polar angle
       * @param[in] phi Azimuthal angle
       * @return Vector containing the second-order partial derivatives of YlmReal with respect to theta and phi, in that order.
       */
      std::vector<double>
      d2YlmReal(const int l, const int m, const double theta, const double phi);

      /*
       * @brief Function to evaluate the inverse of the Jacobian for the transform from cartesian to spherical coordinates
       * @param[in] r Radius of the point
       * @param[in] theta Polar angle of the point
       * @param[in] phi Azimuthal angle of the point
       * @return 2D Vector containing the inverse of the Jacobian
       */
      std::vector<std::vector<double>>
      getJInv(const double r, const double theta, const double phi);
    } // end of namespace sphUtils
  }   // end of namespace utils
} // end of namespace dftfe
#endif // DFTFE_SPHERICALFUNCTIONUTIL_H
