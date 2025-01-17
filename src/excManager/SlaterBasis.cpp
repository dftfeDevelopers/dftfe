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
// @author Arghadwip Paul, Bikash Kanungo
//

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <cmath>
#include <iostream>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/factorials.hpp>

#include "StringOperations.h"
#include "SphericalFunctionUtil.h"
#include "Exceptions.h"
#include "SlaterBasis.h"

namespace dftfe
{
  // local namespace
  namespace
  {
    int
    factorial(int n)
    {
      if (n == 0)
        return 1;
      else
        return n * factorial(n - 1);
    }

    //  std::unordered_map<std::string, std::string>
    //    readAtomToSlaterBasisName(const std::string &fileName)
    //    {
    //      std::unordered_map<std::string, std::string> atomToSlaterBasisName;
    //      std::ifstream                                file(fileName);
    //      if (file.is_open())
    //      {
    //        std::string line;
    //        int         lineNumber = 0;
    //        while (std::getline(file, line))
    //        {
    //          lineNumber++;
    //          std::istringstream iss(line);
    //          std::string        atomSymbol;
    //          std::string        slaterBasisName;
    //
    //          if (iss >> atomSymbol >> slaterBasisName)
    //          {
    //            std::string extra;
    //            std::string msg = "Error: More than two entries in line " +
    //              std::to_string(lineNumber) + "in" + fileName;
    //            if(iss >> extra)
    //              utils::throwException(false, msg);
    //
    //            atomToSlaterBasisName[atomSymbol] = slaterBasisName;
    //          }
    //          else
    //          {
    //            std::string msg = "Error: Invalid format in line " +
    //              std::to_string(lineNumber) + "in" +
    //              fileName;
    //            utils::throwException(false, msg);
    //          }
    //        }
    //        file.close();
    //      }
    //      else
    //      {
    //        std::string msg = "Unable to open the file." + fileName;
    //        utils::throwException(false, msg);
    //      }
    //      return atomToSlaterBasisName;
    //    }

    //    void
    //      getSlaterPrimitivesFromBasisFile(
    //          const std::string &basisName,
    //          std::vector<SlaterPrimitive *> & slaterPrimitivesPtr)
    //      {
    //        /*
    //         * Written in a format that ignores the first line
    //         */
    //        std::ifstream file(basisName);
    //        if (file.is_open())
    //        {
    //          std::string                          line;
    //          std::unordered_map<std::string, int> lStringToIntMap = {
    //            {"S", 0}, {"P", 1}, {"D", 2}, {"F", 3}, {"G", 4}, {"H", 5}};
    //          // First line - Ignore
    //          std::getline(file, line);
    //          std::istringstream iss(line);
    //          if (iss.fail())
    //          {
    //            std::string msg = "Error reading line in file: " +
    //              basisName;
    //            utils::throwException(false, msg);
    //          }
    //          std::string atomType, extra;
    //          iss >> atomType;
    //          if (iss >> extra)
    //          {
    //            std::string msg = "Error: More than one entry in line 1 in" +
    //              basisName;
    //            utils::throwException(false, msg);
    //          }
    //          // Second Line Onwards
    //          while (std::getline(file, line))
    //          {
    //            std::istringstream iss(line);
    //            if (iss.fail())
    //            {
    //              std::string msg = "Error reading line in file: " +
    //                basisName;
    //              utils::throwException(false, msg);
    //            }
    //
    //            std::string nlString;
    //            double      alpha;
    //            iss >> nlString >> alpha;
    //            if (iss >> extra)
    //            {
    //              std::string msg = "Error: More than two entries in a line
    //              in" + basisName; utils::throwException(false, msg);
    //            }
    //
    //            char lChar = nlString.back();
    //
    //            int n = std::stoi(nlString.substr(0, nlString.size() - 1));
    //            // normalization constant for the radial part of Slater
    //            function const double term1 = pow(2.0*alpha, n + 1.0/2.0);
    //            const double term2 = pow(factorial(2*n), 1.0/2.0);
    //            const double normConst = term1/term2;
    //
    //            int l;
    //            try
    //            {
    //              l = lStringToIntMap.at(std::string(1, lChar));
    //            }
    //            catch (const std::out_of_range &e)
    //            {
    //              std::string msg = "Character doesn't exist in the
    //              lStringToIntMap in "
    //                "SlaterBasis.cpp: " + std::string(1, lChar);
    //              utils::throwException(false, msg);
    //            }
    //            std::vector<int> mList;
    //            if (l == 1)
    //            {
    //              // Special ordering for p orbitals to be compatible with
    //              quantum chemistry codes like QChem
    //              // In most quantum chemistry codes, eben for spherical
    //              Gaussians, for the p-orbitals (l=1),
    //              // the m-ordering {1, -1, 0} (i.e., px, py, pz) instead of
    //              {-1, 0, 1} (i.e., py, pz, px) mList = {1, -1, 0};
    //            }
    //            else
    //            {
    //              for (int m = -l; m <= l; ++m)
    //              {
    //                mList.push_back(m);
    //              }
    //            }
    //            for (int m : mList)
    //            {
    //              SlaterPrimitive *sp = new SlaterPrimitive{n, l, m, alpha,
    //              normConst}; slaterPrimitivesPtr.push_back(sp);
    //            }
    //          }
    //        }
    //        else
    //        {
    //          std::string msg = "Unable to open file: " + basisName;
    //          utils::throwException(false, msg);
    //        }
    //      }

    void
    getSlaterPrimitivesFromBasisFile(
      const std::string &             basisName,
      std::vector<SlaterPrimitive *> &slaterPrimitivesPtr)
    {
      /*
       * Written in a format that ignores the first line
       */
      std::ifstream file(basisName);
      if (file.is_open())
        {
          std::string                          line;
          std::unordered_map<std::string, int> lStringToIntMap = {
            {"S", 0},
            {"P", 1},
            {"D", 2},
            {"F", 3},
            {"G", 4},
            {"H", 5},
            {"s", 0},
            {"p", 1},
            {"d", 2},
            {"f", 3},
            {"g", 4},
            {"h", 5},
          };

          // First line - Ignore which contain just the atom symbol
          std::getline(file, line);

          // Second Line Onwards
          while (std::getline(file, line))
            {
              std::istringstream iss(line);
              if (iss.fail())
                {
                  std::string msg = "Error reading line in file: " + basisName;
                  utils::throwException(false, msg);
                }
              std::vector<std::string> words =
                utils::stringOps::split(iss.str(), " ", true);
              std::string msg =
                "Undefined number of entries found in Slater basis file " +
                basisName +
                ". Expects only two entries: nl quantum numbers and the Slater exponent";
              utils::throwException(words.size() == 2, msg);

              // process n and l
              std::string nlString = words[0];
              char        lChar    = nlString.back();
              std::string nStr     = nlString.substr(0, nlString.size() - 1);
              int         n;
              bool        isValidN = utils::stringOps::strToInt(nStr, n);
              utils::throwException(
                isValidN,
                "Invalid n quantum number specified in file " + basisName);

              // process alpha
              double alpha;
              bool   isValidAlpha =
                utils::stringOps::strToDouble(words[1], alpha);
              utils::throwException(isValidAlpha,
                                    "Invalid exponent specified in file " +
                                      basisName);

              // normalization constant for the radial part of Slater function
              const double term1     = pow(2.0 * alpha, n + 1.0 / 2.0);
              const double term2     = pow(factorial(2 * n), 1.0 / 2.0);
              const double normConst = term1 / term2;
              int          l;
              try
                {
                  l = lStringToIntMap.at(std::string(1, lChar));
                }
              catch (const std::out_of_range &e)
                {
                  std::string msg =
                    "Character doesn't exist in the lStringToIntMap in "
                    "SlaterBasis.cpp: " +
                    std::string(1, lChar);
                  utils::throwException(false, msg);
                }
              std::vector<int> mList;
              if (l == 1)
                {
                  // Special ordering for p orbitals to be compatible with
                  // quantum chemistry codes like QChem In most quantum
                  // chemistry codes, eben for spherical Gaussians, for the
                  // p-orbitals (l=1), the m-ordering {1, -1, 0} (i.e., px, py,
                  // pz) instead of {-1, 0, 1} (i.e., py, pz, px)
                  mList = {1, -1, 0};
                }
              else
                {
                  for (int m = -l; m <= l; ++m)
                    {
                      mList.push_back(m);
                    }
                }
              for (int m : mList)
                {
                  SlaterPrimitive *sp =
                    new SlaterPrimitive{n, l, m, alpha, normConst};
                  slaterPrimitivesPtr.push_back(sp);
                }
            }
        }
      else
        {
          std::string msg = "Unable to open file: " + basisName;
          utils::throwException(false, msg);
        }
    }

    inline double
    slaterRadialPart(const double r, const int n, const double alpha)
    {
      if (n == 1)
        return exp(-alpha * r);
      else
        return pow(r, n - 1) * exp(-alpha * r);
    }

    inline double
    slaterRadialPartDerivative(const double r,
                               const double alpha,
                               const int    n,
                               const int    derOrder)
    {
      if (derOrder == 0 && n >= 1)
        return slaterRadialPart(r, n, alpha);
      else if (derOrder == 0 && n < 1)
        return 0.0;
      else
        return (n - 1) *
                 slaterRadialPartDerivative(r, alpha, n - 1, derOrder - 1) -
               alpha * slaterRadialPartDerivative(r, alpha, n, derOrder - 1);
    }

    double
    getSlaterValue(const std::vector<double> &x,
                   const int                  n,
                   const int                  l,
                   const int                  m,
                   const double               alpha,
                   const double               rTol,
                   const double               angleTol)
    {
      double r, theta, phi;
      utils::sphUtils::convertCartesianToSpherical(
        x, r, theta, phi, rTol, angleTol);
      const double Ylm         = utils::sphUtils::YlmReal(l, m, theta, phi);
      const double R           = slaterRadialPart(r, n, alpha);
      const double returnValue = R * Ylm;
      return returnValue;
    }

    std::vector<double>
    getSlaterGradientAtOrigin(const int    n,
                              const int    l,
                              const int    m,
                              const double alpha)
    {
      std::vector<double> returnValue(3);
      const int           modM = std::abs(m);
      const double        C    = utils::sphUtils::Clm(l, m);
      if (n == 1)
        {
          std::string message(
            "Gradient of slater orbital at atomic position is undefined for n=1");
          utils::throwException(false, message);
        }

      if (n == 2)
        {
          if (l == 0)
            {
              std::string message(
                "Gradient of slater orbital at atomic position is undefined for n=2 and l=0");
              utils::throwException(false, message);
            }
          if (l == 1)
            {
              if (m == -1)
                {
                  returnValue[0] = 0.0;
                  returnValue[1] = C;
                  returnValue[2] = 0.0;
                }

              if (m == 0)
                {
                  returnValue[0] = 0.0;
                  returnValue[1] = 0.0;
                  returnValue[2] = C;
                }

              if (m == 1)
                {
                  returnValue[0] = C;
                  returnValue[1] = 0.0;
                  returnValue[2] = 0.0;
                }
            }
        }

      else
        {
          returnValue[0] = 0.0;
          returnValue[1] = 0.0;
          returnValue[2] = 0.0;
        }

      return returnValue;
    }

    std::vector<double>
    getSlaterGradientAtPoles(const double r,
                             const double theta,
                             const int    n,
                             const int    l,
                             const int    m,
                             const double alpha,
                             const double angleTol)
    {
      const double        R    = slaterRadialPart(r, n, alpha);
      const double        dRDr = slaterRadialPartDerivative(r, alpha, n, 1);
      const double        C    = utils::sphUtils::Clm(l, m);
      std::vector<double> returnValue(3);
      if (std::fabs(theta - 0.0) < angleTol)
        {
          if (m == 0)
            {
              returnValue[0] = 0.0;
              returnValue[1] = 0.0;
              returnValue[2] = C * dRDr;
            }

          else if (m == 1)
            {
              returnValue[0] = C * (R / r) * l * (l + 1) / 2.0;
              returnValue[1] = 0.0;
              returnValue[2] = 0.0;
            }

          else if (m == -1)
            {
              returnValue[0] = 0.0;
              returnValue[1] = C * (R / r) * l * (l + 1) / 2.0;
              returnValue[2] = 0.0;
            }

          else
            {
              returnValue[0] = 0.0;
              returnValue[1] = 0.0;
              returnValue[2] = 0.0;
            }
        }

      else // the other possibility is std::fabs(theta-M_PI) < angleTol
        {
          if (m == 0)
            {
              returnValue[0] = 0.0;
              returnValue[1] = 0.0;
              returnValue[2] = C * dRDr * pow(-1, l + 1);
            }

          else if (m == 1)
            {
              returnValue[0] = C * (R / r) * l * (l + 1) / 2.0 * pow(-1, l + 1);
              returnValue[1] = 0.0;
              returnValue[2] = 0.0;
            }

          else if (m == -1)
            {
              returnValue[0] = 0.0;
              returnValue[1] = C * (R / r) * l * (l + 1) / 2.0 * pow(-1, l + 1);
              returnValue[2] = 0.0;
            }

          else
            {
              returnValue[0] = 0.0;
              returnValue[1] = 0.0;
              returnValue[2] = 0.0;
            }
        }
      return returnValue;
    }

    std::vector<double>
    getSlaterGradient(const std::vector<double> &x,
                      const int                  n,
                      const int                  l,
                      const int                  m,
                      const double               alpha,
                      const double               rTol,
                      const double               angleTol)
    {
      double r, theta, phi;
      utils::sphUtils::convertCartesianToSpherical(
        x, r, theta, phi, rTol, angleTol);
      std::vector<double> returnValue(3);
      if (r < rTol)
        {
          returnValue = getSlaterGradientAtOrigin(n, l, m, alpha);
        }

      else if (std::fabs(theta - 0.0) < angleTol ||
               std::fabs(theta - M_PI) < angleTol)
        {
          returnValue =
            getSlaterGradientAtPoles(r, theta, n, l, m, alpha, angleTol);
        }

      else
        {
          const double R    = slaterRadialPart(r, n, alpha);
          const double dRDr = slaterRadialPartDerivative(r, alpha, n, 1);
          const double Ylm  = utils::sphUtils::YlmReal(l, m, theta, phi);
          const std::vector<double> dYlm =
            utils::sphUtils::dYlmReal(l, m, theta, phi);
          std::vector<std::vector<double>> jacobianInverse =
            utils::sphUtils::getJInv(r, theta, phi);
          double partialDerivatives[3];
          partialDerivatives[0] = dRDr * Ylm;
          partialDerivatives[1] = R * dYlm[0];
          partialDerivatives[2] = R * dYlm[1];
          for (unsigned int i = 0; i < 3; ++i)
            {
              returnValue[i] = jacobianInverse[i][0] * partialDerivatives[0] +
                               jacobianInverse[i][1] * partialDerivatives[1] +
                               jacobianInverse[i][2] * partialDerivatives[2];
            }
        }
      return returnValue;
    }

    double
    getSlaterLaplacianAtOrigin(const int    n,
                               const int    l,
                               const int    m,
                               const double alpha)
    {
      if (n == 1 || n == 2)
        {
          std::string message(
            "Laplacian of slater function is undefined at atomic position for n=1 and n=2.");
          utils::throwException(false, message);
          return 0.0;
        }

      else if (n == 3)
        {
          if (l == 0)
            {
              const double C = utils::sphUtils::Clm(l, m);
              return 6.0 * C;
            }

          else if (l == 1)
            {
              std::string message(
                "Laplacian of slater function is undefined at atomic position for n=3, l=1.");
              utils::throwException(false, message);
              return 0.0;
            }

          else if (l == 2)
            {
              return 0.0;
            }

          else // l >= 3
            return 0.0;
        }

      else
        {
          return 0.0;
        }
    }

    double
    getSlaterLaplacianAtPoles(const double r,
                              const double theta,
                              const int    n,
                              const int    l,
                              const int    m,
                              const double alpha,
                              const double angleTol)
    {
      double returnValue = 0.0;
      if (m == 0)
        {
          const double C      = utils::sphUtils::Clm(l, m);
          const double R      = slaterRadialPart(r, n, alpha);
          const double dRdr   = slaterRadialPartDerivative(r, alpha, n, 1);
          const double d2Rdr2 = slaterRadialPartDerivative(r, alpha, n, 2);
          if (std::fabs(theta - 0.0) < angleTol)
            {
              const double term1 = C * (2.0 * dRdr / r + d2Rdr2);
              const double term2 = C * (R / (r * r)) * (-l * (l + 1));
              returnValue        = term1 + term2;
            }
          else // the other possibility is std::fabs(theta-M_PI) < angleTol
            {
              const double term1 = C * (2.0 * dRdr / r + d2Rdr2) * pow(-1, l);
              const double term2 =
                C * (R / (r * r)) * (-l * (l + 1)) * pow(-1, l);
              returnValue = term1 + term2;
            }
        }

      else
        returnValue = 0.0;

      return returnValue;
    }

    double
    getSlaterLaplacian(const std::vector<double> &x,
                       const int                  n,
                       const int                  l,
                       const int                  m,
                       const double               alpha,
                       const double               rTol,
                       const double               angleTol)
    {
      double r, theta, phi;
      utils::sphUtils::convertCartesianToSpherical(
        x, r, theta, phi, rTol, angleTol);
      double returnValue = 0.0;
      if (r < rTol)
        {
          returnValue = getSlaterLaplacianAtOrigin(n, l, m, alpha);
        }

      else if (std::fabs(theta - 0.0) < angleTol ||
               std::fabs(theta - M_PI) < angleTol)
        {
          returnValue =
            getSlaterLaplacianAtPoles(r, theta, n, l, m, alpha, angleTol);
        }

      else
        {
          const double cosTheta = cos(theta);
          const double sinTheta = sin(theta);
          const double R        = slaterRadialPart(r, n, alpha);
          const double dRdr     = slaterRadialPartDerivative(r, alpha, n, 1);
          const double d2Rdr2   = slaterRadialPartDerivative(r, alpha, n, 2);
          const double Ylm      = utils::sphUtils::YlmReal(l, m, theta, phi);
          const std::vector<double> dYlm =
            utils::sphUtils::dYlmReal(l, m, theta, phi);
          const std::vector<double> d2Ylm =
            utils::sphUtils::d2YlmReal(l, m, theta, phi);
          const double term1 = (2.0 * dRdr / r + d2Rdr2) * Ylm;
          const double term2 =
            (R / (r * r)) * ((cosTheta / sinTheta) * dYlm[0] + d2Ylm[0]);
          const double term3 =
            (R / (r * r)) * (d2Ylm[1] / (sinTheta * sinTheta));
          returnValue = term1 + term2 + term3;
        }
      return returnValue;
    }

  } // namespace

  SlaterBasis::SlaterBasis(const double rTol /*=1e-10*/,
                           const double angleTol /*=1e-10*/)
    : d_rTol(rTol)
    , d_angleTol(angleTol)
  {}

  SlaterBasis::~SlaterBasis()
  {
    // deallocate SlaterPrimitive pointers stored in the map
    for (auto &pair : d_atomToSlaterPrimitivesPtr)
      {
        std::vector<SlaterPrimitive *> &primitives = pair.second;
        for (SlaterPrimitive *sp : primitives)
          {
            if (sp != nullptr)
              {
                delete sp;
              }
          }
        primitives.clear();
      }
  }


  void
  SlaterBasis::constructBasisSet(
    const std::vector<std::pair<std::string, std::vector<double>>> &atomCoords,
    const std::unordered_map<std::string, std::string> &atomBasisFileNames)
  {
    d_atomSymbolsAndCoords = atomCoords;
    unsigned int natoms    = d_atomSymbolsAndCoords.size();
    d_atomToSlaterPrimitivesPtr.clear();
    for (const auto &pair : atomBasisFileNames)
      {
        const std::string &atomSymbol = pair.first;
        const std::string &basisName  = pair.second;
        d_atomToSlaterPrimitivesPtr[atomSymbol] =
          std::vector<SlaterPrimitive *>(0);
        getSlaterPrimitivesFromBasisFile(
          basisName, d_atomToSlaterPrimitivesPtr[atomSymbol]);
      }

    for (unsigned int i = 0; i < natoms; ++i)
      {
        const std::string &        atomSymbol = d_atomSymbolsAndCoords[i].first;
        const std::vector<double> &atomCenter =
          d_atomSymbolsAndCoords[i].second;
        unsigned int nprimitives =
          d_atomToSlaterPrimitivesPtr[atomSymbol].size();
        for (unsigned int j = 0; j < nprimitives; ++j)
          {
            SlaterBasisInfo info;
            info.symbol = &atomSymbol;
            info.center = atomCenter.data();
            info.sp     = d_atomToSlaterPrimitivesPtr[atomSymbol][j];
            d_slaterBasisInfo.push_back(info);
          }
      }
  }

  /*
  const std::vector<SlaterBasisInfo> &
    SlaterBasis::getSlaterBasisInfo() const
    {
      return d_slaterBasisInfo;
    }
  */

  int
  SlaterBasis::getNumBasis() const
  {
    return d_slaterBasisInfo.size();
  }


  std::vector<double>
  SlaterBasis::getBasisValue(const unsigned int         basisId,
                             const std::vector<double> &x) const
  {
    const SlaterBasisInfo &info      = d_slaterBasisInfo[basisId];
    const double *         x0        = info.center;
    const SlaterPrimitive *sp        = info.sp;
    const double           alpha     = sp->alpha;
    const int              n         = sp->n;
    const int              l         = sp->l;
    const int              m         = sp->m;
    const double           normConst = sp->normConst;
    const unsigned int     nPoints   = round(x.size() / 3);
    std::vector<double>    returnValue(nPoints, 0.0);
    std::vector<double>    dx(3);
    for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            dx[j] = x[iPoint * 3 + j] - x0[j];
          }
        returnValue[iPoint] =
          normConst * getSlaterValue(dx, n, l, m, alpha, d_rTol, d_angleTol);
      }
    return returnValue;
  }

  std::vector<double>
  SlaterBasis::getBasisGradient(const unsigned int         basisId,
                                const std::vector<double> &x) const
  {
    const SlaterBasisInfo &info      = d_slaterBasisInfo[basisId];
    const double *         x0        = info.center;
    const SlaterPrimitive *sp        = info.sp;
    const double           alpha     = sp->alpha;
    const int              n         = sp->n;
    const int              l         = sp->l;
    const int              m         = sp->m;
    const double           normConst = sp->normConst;
    const unsigned int     nPoints   = round(x.size() / 3);
    std::vector<double>    returnValue(3 * nPoints, 0.0);
    std::vector<double>    dx(3);
    for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            dx[j] = x[iPoint * 3 + j] - x0[j];
          }

        std::vector<double> tmp =
          getSlaterGradient(dx, n, l, m, alpha, d_rTol, d_angleTol);
        for (unsigned int j = 0; j < 3; ++j)
          returnValue[iPoint * 3 + j] = normConst * tmp[j];
      }
    return returnValue;
  }

  std::vector<double>
  SlaterBasis::getBasisLaplacian(const unsigned int         basisId,
                                 const std::vector<double> &x) const
  {
    const SlaterBasisInfo &info      = d_slaterBasisInfo[basisId];
    const double *         x0        = info.center;
    const SlaterPrimitive *sp        = info.sp;
    const double           alpha     = sp->alpha;
    const int              n         = sp->n;
    const int              l         = sp->l;
    const int              m         = sp->m;
    const double           normConst = sp->normConst;
    const unsigned int     nPoints   = round(x.size() / 3);
    std::vector<double>    returnValue(nPoints, 0.0);
    std::vector<double>    dx(3);
    for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            dx[j] = x[iPoint * 3 + j] - x0[j];
          }

        returnValue[iPoint] =
          normConst *
          getSlaterLaplacian(dx, n, l, m, alpha, d_rTol, d_angleTol);
      }
    return returnValue;
  }
} // namespace dftfe
