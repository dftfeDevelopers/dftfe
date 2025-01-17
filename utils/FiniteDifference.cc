#include <FiniteDifference.h>
#include <cmath>
#include <string>
#include "Exceptions.h"

namespace dftfe
{
  namespace utils
  {
    std::vector<double>
    FiniteDifference::getStencilGridOneVariableCentral(
      const unsigned int totalStencilSize,
      const double       h)
    {
      std::vector<double> stencil(totalStencilSize, 0);

      std::string errMsg = "Stencil size invalid. ";
      dftfe::utils::throwException(totalStencilSize > 2 &&
                                     totalStencilSize % 2 == 1,
                                   errMsg);

      for (unsigned int i = 0; i < totalStencilSize; i++)
        stencil[i] = (-std::floor(totalStencilSize / 2) * h + i * h);

      return stencil;
    }


    void
    FiniteDifference::firstOrderDerivativeOneVariableCentral(
      const unsigned int totalStencilSize,
      const double       h,
      const unsigned int numQuadPoints,
      const double *     stencilDataAllQuadPoints,
      double *           firstOrderDerivative)
    {
      std::string errMsg = "Stencil size invalid. ";
      dftfe::utils::throwException(totalStencilSize > 2 &&
                                     totalStencilSize % 2 == 1,
                                   errMsg);

      switch (totalStencilSize)
        {
          case 3:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (-1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2]) /
                  (2.0 * h);
              }
            break;
          case 5:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   8.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   8.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4]) /
                  (12.0 * h);
              }
            break;
          case 7:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (-1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   9.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   45.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   45.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   9.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6]) /
                  (60.0 * h);
              }
            break;
          case 9:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (3.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   32.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   168 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   672.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   672.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] -
                   168.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] +
                   32.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   3.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8]) /
                  (840.0 * h);
              }
            break;
          case 11:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (-2.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   25.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   150 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   2100.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] +
                   2100.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] -
                   600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] +
                   150.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] -
                   25.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] +
                   2.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10]) /
                  (2520.0 * h);
              }
            break;
          case 13:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (5.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   72.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   495.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   2200.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   23760.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   23760.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] +
                   2200.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] -
                   495.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10] +
                   72.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 11] -
                   5.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 12]) /
                  (27720.0 * h);
              }
            break;
          default:
            std::string errMsg = "Stencil size not implemented. ";
            dftfe::utils::throwException(false, errMsg);
            break;
        }
    }

    void
    FiniteDifference::firstOrderDerivativeOneVariableCentral(
      const unsigned int totalStencilSize,
      const double *     h,
      const unsigned int numQuadPoints,
      const double *     stencilDataAllQuadPoints,
      double *           firstOrderDerivative)
    {
      std::string errMsg = "Stencil size invalid. ";
      dftfe::utils::throwException(totalStencilSize > 2 &&
                                     totalStencilSize % 2 == 1,
                                   errMsg);

      switch (totalStencilSize)
        {
          case 3:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (-1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2]) /
                  (2.0 * h[iquad]);
              }
            break;
          case 5:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   8.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   8.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4]) /
                  (12.0 * h[iquad]);
              }
            break;
          case 7:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (-1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   9.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   45.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   45.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   9.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6]) /
                  (60.0 * h[iquad]);
              }
            break;
          case 9:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (3.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   32.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   168 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   672.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   672.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] -
                   168.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] +
                   32.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   3.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8]) /
                  (840.0 * h[iquad]);
              }
            break;
          case 11:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (-2.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   25.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   150 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   2100.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] +
                   2100.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] -
                   600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] +
                   150.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] -
                   25.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] +
                   2.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10]) /
                  (2520.0 * h[iquad]);
              }
            break;
          case 13:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                firstOrderDerivative[iquad] =
                  (5.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   72.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   495.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   2200.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   23760.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   23760.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] +
                   2200.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] -
                   495.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10] +
                   72.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 11] -
                   5.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 12]) /
                  (27720.0 * h[iquad]);
              }
            break;
          default:
            std::string errMsg = "Stencil size not implemented. ";
            dftfe::utils::throwException(false, errMsg);
            break;
        }
    }

    void
    FiniteDifference::secondOrderDerivativeOneVariableCentral(
      const unsigned int totalStencilSize,
      const double       h,
      const unsigned int numQuadPoints,
      const double *     stencilDataAllQuadPoints,
      double *           secondOrderDerivative)
    {
      std::string errMsg = "Stencil size invalid. ";
      dftfe::utils::throwException(totalStencilSize > 2 &&
                                     totalStencilSize % 2 == 1,
                                   errMsg);

      switch (totalStencilSize)
        {
          case 3:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   2.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2]) /
                  (h * h);
              }
            break;
          case 5:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (-1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   16.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   30.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   16.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   -1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4]) /
                  (12.0 * h * h);
              }
            break;
          case 7:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (2.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   27.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   270.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   490.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   270.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   27.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   2.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6]) /
                  (180.0 * h * h);
              }
            break;
          case 9:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (-9.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   128.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   1008.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   8064.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   14350.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] +
                   8064.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] -
                   1008.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] +
                   128.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   9.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8]) /
                  (5040.0 * h * h);
              }
            break;
          case 11:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (8.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   125.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   1000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   6000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   42000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   73766.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   42000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] -
                   6000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] +
                   1000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] -
                   125.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] +
                   125.0 * 8.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10]) /
                  (25200.0 * h * h);
              }
            break;
          case 13:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (-50.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   864.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   44000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   222750.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] +
                   1425600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] -
                   2480478.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] +
                   1425600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   222750.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] +
                   44000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] -
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10] +
                   864.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 11] -
                   50.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 12]) /
                  (831600.0 * h * h);
              }
            break;
          default:
            std::string errMsg = "Stencil size not implemented. ";
            dftfe::utils::throwException(false, errMsg);
            break;
        }
    }

    void
    FiniteDifference::secondOrderDerivativeOneVariableCentral(
      const unsigned int totalStencilSize,
      const double *     h,
      const unsigned int numQuadPoints,
      const double *     stencilDataAllQuadPoints,
      double *           secondOrderDerivative)
    {
      std::string errMsg = "Stencil size invalid. ";
      dftfe::utils::throwException(totalStencilSize > 2 &&
                                     totalStencilSize % 2 == 1,
                                   errMsg);

      switch (totalStencilSize)
        {
          case 3:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   2.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2]) /
                  (h[iquad] * h[iquad]);
              }
            break;
          case 5:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (-1.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   16.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   30.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   16.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   -1.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4]) /
                  (12.0 * h[iquad] * h[iquad]);
              }
            break;
          case 7:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (2.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   27.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   270.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   490.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   270.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   27.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   2.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6]) /
                  (180.0 * h[iquad] * h[iquad]);
              }
            break;
          case 9:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (-9.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   128.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   1008.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   8064.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   14350.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] +
                   8064.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] -
                   1008.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] +
                   128.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   9.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8]) /
                  (5040.0 * h[iquad] * h[iquad]);
              }
            break;
          case 11:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (8.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] -
                   125.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] +
                   1000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] -
                   6000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] +
                   42000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] -
                   73766.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] +
                   42000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] -
                   6000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] +
                   1000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] -
                   125.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] +
                   125.0 * 8.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10]) /
                  (25200.0 * h[iquad] * h[iquad]);
              }
            break;
          case 13:
            for (unsigned int iquad = 0; iquad < numQuadPoints; iquad++)
              {
                secondOrderDerivative[iquad] =
                  (-50.0 * stencilDataAllQuadPoints[iquad * totalStencilSize] +
                   864.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 1] -
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 2] +
                   44000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 3] -
                   222750.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 4] +
                   1425600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 5] -
                   2480478.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 6] +
                   1425600.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 7] -
                   222750.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 8] +
                   44000.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 9] -
                   7425.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 10] +
                   864.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 11] -
                   50.0 *
                     stencilDataAllQuadPoints[iquad * totalStencilSize + 12]) /
                  (831600.0 * h[iquad] * h[iquad]);
              }
            break;
          default:
            std::string errMsg = "Stencil size not implemented. ";
            dftfe::utils::throwException(false, errMsg);
            break;
        }
    }

  } // namespace utils
} // namespace dftfe
