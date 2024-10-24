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
// @author Bikash Kanungo
//

#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include "Exceptions.h"
#include "SphericalFunctionUtil.h"
namespace dftfe
{
  namespace utils
  {
    namespace sphUtils
    {
      // local namespace
      namespace
      {
        inline double
        Am(const int m)
        {
          if (m == 0)
            return 1.0 / sqrt(2 * M_PI);
          else
            return 1.0 / sqrt(M_PI);
        }

        inline double
        Blm(const int l, const int m)
        {
          if (m == 0)
            return sqrt((2.0 * l + 1) / 2.0);
          else
            return Blm(l, m - 1) / sqrt((l - m + 1.0) * (l + m));
          // return sqrt(((2.0*l +
          // 1)*boost::math::factorial<double>(l-m))/(2.0*boost::math::factorial<double>(l+m)));
        }
      } // namespace

      void
      convertCartesianToSpherical(const std::vector<double> &x,
                                  double &                   r,
                                  double &                   theta,
                                  double &                   phi,
                                  const double               rTol,
                                  const double               angleTol)
      {
        r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        if (r < rTol)
          {
            theta = 0.0;
            phi   = 0.0;
          }
        else
          {
            theta = acos(x[2] / r);
            //
            // check if theta = 0 or PI (i.e, whether the point is on the
            // Z-axis) If yes, assign phi = 0.0. NOTE: In case theta = 0 or PI,
            // phi is undetermined. The actual value of phi doesn't matter in
            // computing the function value or its gradient. We assign phi = 0.0
            // here just as a dummy value
            //
            if (fabs(theta - 0.0) >= angleTol && fabs(theta - M_PI) >= angleTol)
              phi = atan2(x[1], x[0]);
            else
              phi = 0.0;
          }
      }

      double
      Clm(const int l, const int m)
      {
        int modM = std::abs(m);
        return Am(modM) * Blm(l, modM);
      }

      double
      Qm(const int m, const double phi)
      {
        if (m > 0)
          return cos(m * phi);
        else if (m == 0)
          return 1.0;
        else // if(m < 0)
          return sin(std::abs(m) * phi);
      }

      double
      dQmDPhi(const int m, const double phi)
      {
        if (m > 0)
          return -m * sin(m * phi);
        else if (m == 0)
          return 0.0;
        else //(m < 0)
          return std::abs(m) * cos(std::abs(m) * phi);
      }

      double
      d2QmDPhi2(const int m, const double phi)
      {
        if (m > 0)
          return -m * m * cos(m * phi);
        else if (m == 0)
          return 0.0;
        else //(m < 0)
          return -m * m * sin(std::abs(m) * phi);
      }

      double
      Plm(const int l, const int m, const double theta)
      {
        const double cosTheta = cos(theta);
        if (std::abs(m) > l)
          return 0.0;
        else
          //
          // NOTE: Multiplies by {-1}^m to remove the
          // implicit Condon-Shortley factor in the associated legendre
          // polynomial implementation of boost
          // This is done to be consistent with the QChem's implementation
          return pow(-1.0, m) * boost::math::legendre_p(l, m, cosTheta);
      }

      double
      dPlmDTheta(const int l, const int m, const double theta)
      {
        if (std::abs(m) > l)
          return 0.0;

        else if (l == 0)
          return 0.0;

        else if (m < 0)
          {
            const int    modM   = std::abs(m);
            const double factor = pow(-1, m) *
                                  boost::math::factorial<double>(l - modM) /
                                  boost::math::factorial<double>(l + modM);
            return factor * dPlmDTheta(l, modM, theta);
          }

        else if (m == 0)
          {
            return -1.0 * Plm(l, 1, theta);
          }

        else if (m == l)
          return l * Plm(l, l - 1, theta);

        else
          {
            const double term1 = (l + m) * (l - m + 1) * Plm(l, m - 1, theta);
            const double term2 = Plm(l, m + 1, theta);
            return 0.5 * (term1 - term2);
          }
      }

      double
      d2PlmDTheta2(const int l, const int m, const double theta)
      {
        if (std::abs(m) > l)
          return 0.0;

        else if (l == 0)
          return 0.0;

        else if (m < 0)
          {
            const int    modM   = std::abs(m);
            const double factor = pow(-1, m) *
                                  boost::math::factorial<double>(l - modM) /
                                  boost::math::factorial<double>(l + modM);
            return factor * d2PlmDTheta2(l, modM, theta);
          }

        else if (m == 0)
          return -1.0 * dPlmDTheta(l, 1, theta);

        else if (m == l)
          return l * dPlmDTheta(l, l - 1, theta);

        else
          {
            double term1 = (l + m) * (l - m + 1) * dPlmDTheta(l, m - 1, theta);
            double term2 = dPlmDTheta(l, m + 1, theta);
            return 0.5 * (term1 - term2);
          }
      }

      double
      YlmReal(const int l, const int m, const double theta, const double phi)
      {
        int          modM = std::abs(m);
        const double C    = Clm(l, m);
        const double P    = Plm(l, modM, theta);
        const double Q    = Qm(m, phi);
        return C * P * Q;
      }

      std::vector<double>
      dYlmReal(const int l, const int m, const double theta, const double phi)
      {
        std::vector<double> returnValue(2, 0.0);
        int                 modM = std::abs(m);
        const double        C    = Clm(l, m);
        const double        P    = Plm(l, modM, theta);
        const double        dP   = dPlmDTheta(l, modM, theta);
        const double        Q    = Qm(m, phi);
        const double        dQ   = dQmDPhi(m, phi);
        returnValue[0]           = C * dP * Q;
        returnValue[1]           = C * P * dQ;
        return returnValue;
      }

      std::vector<double>
      d2YlmReal(const int l, const int m, const double theta, const double phi)
      {
        std::vector<double> returnValue(2, 0.0);
        int                 modM = std::abs(m);
        const double        C    = Clm(l, m);
        const double        P    = Plm(l, modM, theta);
        const double        d2P  = d2PlmDTheta2(l, modM, theta);
        const double        Q    = Qm(m, phi);
        const double        d2Q  = d2QmDPhi2(m, phi);
        returnValue[0]           = C * d2P * Q;
        returnValue[1]           = C * P * d2Q;
        return returnValue;
      }

      std::vector<std::vector<double>>
      getJInv(const double r, const double theta, const double phi)
      {
        std::vector<std::vector<double>> jacobianInverse(
          3, std::vector<double>(3, 0.0));
        jacobianInverse[0][0] = sin(theta) * cos(phi);
        jacobianInverse[0][1] = cos(theta) * cos(phi) / r;
        jacobianInverse[0][2] = -1.0 * sin(phi) / (r * sin(theta));
        jacobianInverse[1][0] = sin(theta) * sin(phi);
        jacobianInverse[1][1] = cos(theta) * sin(phi) / r;
        jacobianInverse[1][2] = cos(phi) / (r * sin(theta));
        jacobianInverse[2][0] = cos(theta);
        jacobianInverse[2][1] = -1.0 * sin(theta) / r;
        jacobianInverse[2][2] = 0.0;
        return jacobianInverse;
      }
    } // namespace sphUtils
  }   // namespace utils
} // namespace dftfe
