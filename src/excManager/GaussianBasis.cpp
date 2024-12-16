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

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iostream>
#include <cmath>

#include "Exceptions.h"
#include "GaussianBasis.h"
#include "SphericalFunctionUtil.h"
#include "StringOperations.h"

namespace dftfe
{
  // local namespace
  namespace
  {
    double
    doubleFactorial(int n)
    {
      if (n == 0 || n == -1)
        return 1.0;
      return n * doubleFactorial(n - 2);
    }

    void
    printContractedGaussian(const ContractedGaussian &cg)
    {
      const int nC = cg.nC;
      for (unsigned int i = 0; i < nC; ++i)
        {
          std::cout << "alpha: " << cg.alpha[i] << " c: " << cg.c[i]
                    << " norm: " << cg.norm[i] << std::endl;
        }
    }

    std::vector<double>
    getNormConsts(const std::vector<double> &alpha, const int l)
    {
      int                 L = alpha.size();
      std::vector<double> returnValue(L, 0.0);
      for (unsigned int i = 0; i < L; ++i)
        {
          const double term1 = doubleFactorial(2 * l + 1) * sqrt(M_PI);
          const double term2 = pow(2.0, 2 * l + 3.5) * pow(alpha[i], l + 1.5);
          const double overlapIntegral = term1 / term2;
          returnValue[i]               = 1.0 / sqrt(overlapIntegral);
        }
      return returnValue;
    }

    double
    gaussianRadialPart(const double r, const int l, const double alpha)
    {
      return pow(r, l) * exp(-alpha * r * r);
    }


    double
    gaussianRadialPartDerivative(const double r,
                                 const double alpha,
                                 const int    l,
                                 const int    derOrder)
    {
      if (derOrder == 0 && l >= 0)
        return pow(r, l) * exp(-alpha * r * r);
      else if (derOrder == 0 && l < 0)
        return 0.0;
      else
        return l * gaussianRadialPartDerivative(r, alpha, l - 1, derOrder - 1) -
               2 * alpha *
                 gaussianRadialPartDerivative(r, alpha, l + 1, derOrder - 1);
    }

    double
    getLimitingValueLaplacian(const int    l,
                              const int    m,
                              const double theta,
                              const double angleTol)
    {
      double returnValue = 0.0;
      if (std::fabs(theta - 0.0) < angleTol)
        {
          if (m == 0)
            returnValue = -0.5 * l * (l + 1);
          if (m == 2)
            returnValue = 0.25 * (l - 1) * l * (l + 1) * (l + 2);
        }

      if (std::fabs(theta - M_PI) < angleTol)
        {
          if (m == 0)
            returnValue = -0.5 * l * (l + 1) * pow(-1.0, l);
          if (m == 2)
            returnValue = 0.25 * (l - 1) * l * (l + 1) * (l + 2) * pow(-1.0, l);
          ;
        }

      return returnValue;
    }

    double
    getContractedGaussianValue(const ContractedGaussian * cg,
                               const std::vector<double> &x,
                               const double               rTol,
                               const double               angleTol)
    {
      const int nC = cg->nC;
      const int l  = cg->l;
      const int m  = cg->m;
      double    r, theta, phi;
      utils::sphUtils::convertCartesianToSpherical(
        x, r, theta, phi, rTol, angleTol);
      double returnValue = 0.0;
      for (unsigned int i = 0; i < nC; ++i)
        {
          const double alphaVal = cg->alpha[i];
          const double cVal     = cg->c[i];
          const double norm     = cg->norm[i];
          returnValue += cVal * norm * gaussianRadialPart(r, l, alphaVal);
        }
      const double Ylm = utils::sphUtils::YlmReal(l, m, theta, phi);
      returnValue *= Ylm;
      return returnValue;
    }

    std::vector<double>
    getContractedGaussianGradient(const ContractedGaussian * cg,
                                  const std::vector<double> &x,
                                  double                     rTol,
                                  double                     angleTol)
    {
      const int nC = cg->nC;
      const int l  = cg->l;
      const int m  = cg->m;
      double    r, theta, phi;
      utils::sphUtils::convertCartesianToSpherical(
        x, r, theta, phi, rTol, angleTol);
      std::vector<double> returnValue(3);
      double              R    = 0.0;
      double              dRdr = 0.0;
      double              T    = 0.0;
      for (unsigned int i = 0; i < nC; ++i)
        {
          const double alphaVal = cg->alpha[i];
          const double cVal     = cg->c[i];
          const double norm     = cg->norm[i];
          R += cVal * norm * gaussianRadialPart(r, l, alphaVal);
          dRdr += cVal * norm * gaussianRadialPartDerivative(r, alphaVal, l, 1);
          T += cVal * norm;
        }

      const int           modM     = std::abs(m);
      const double        C        = utils::sphUtils::Clm(l, m);
      const double        cosTheta = cos(theta);
      const double        P        = utils::sphUtils::Plm(l, modM, theta);
      double              Ylm      = utils::sphUtils::YlmReal(l, m, theta, phi);
      std::vector<double> dYlm = utils::sphUtils::dYlmReal(l, m, theta, phi);
      if (r < rTol)
        {
          if (l == 1)
            {
              if (m == -1)
                {
                  returnValue[0] = 0.0;
                  returnValue[1] = C * T;
                  returnValue[2] = 0.0;
                }

              if (m == 0)
                {
                  returnValue[0] = 0.0;
                  returnValue[1] = 0.0;
                  returnValue[2] = C * T;
                }

              if (m == 1)
                {
                  returnValue[0] = C * T;
                  returnValue[1] = 0.0;
                  returnValue[2] = 0.0;
                }
            }

          else
            {
              returnValue[0] = 0.0;
              returnValue[1] = 0.0;
              returnValue[2] = 0.0;
            }
        }
      else if (std::fabs(theta - 0.0) < angleTol)
        {
          if (m == 0)
            {
              returnValue[0] = 0.0;
              returnValue[1] = 0.0;
              returnValue[2] = C * dRdr * P * cosTheta;
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

      else if (std::fabs(theta - M_PI) < angleTol)
        {
          if (m == 0)
            {
              returnValue[0] = 0.0;
              returnValue[1] = 0.0;
              returnValue[2] = C * dRdr * P * cosTheta;
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

      else
        {
          std::vector<std::vector<double>> jacobianInverse =
            utils::sphUtils::getJInv(r, theta, phi);
          double partialDerivatives[3];
          partialDerivatives[0] = dRdr * Ylm;
          partialDerivatives[1] = R * dYlm[0];
          partialDerivatives[2] = R * dYlm[1];
          for (unsigned int i = 0; i < 3; ++i)
            {
              returnValue[i] = (jacobianInverse[i][0] * partialDerivatives[0] +
                                jacobianInverse[i][1] * partialDerivatives[1] +
                                jacobianInverse[i][2] * partialDerivatives[2]);
            }
        }

      return returnValue;
    }


    double
    getContractedGaussianLaplacian(const ContractedGaussian * cg,
                                   const std::vector<double> &x,
                                   double                     rTol,
                                   double                     angleTol)
    {
      const int nC          = cg->nC;
      const int l           = cg->l;
      const int m           = cg->m;
      double    returnValue = 0.0;
      double    r, theta, phi;
      utils::sphUtils::convertCartesianToSpherical(
        x, r, theta, phi, rTol, angleTol);
      const int    modM     = std::abs(m);
      const double C        = utils::sphUtils::Clm(l, modM);
      const double cosTheta = cos(theta);
      const double sinTheta = sin(theta);
      const double Ylm      = utils::sphUtils::YlmReal(l, m, theta, phi);
      const double Q        = utils::sphUtils::Qm(m, phi);
      const std::vector<double> dYlm =
        utils::sphUtils::dYlmReal(l, m, theta, phi);
      const std::vector<double> d2Ylm =
        utils::sphUtils::d2YlmReal(l, m, theta, phi);
      double R      = 0.0;
      double dRdr   = 0.0;
      double d2Rdr2 = 0.0;
      double S      = 0.0;
      for (unsigned int i = 0; i < nC; ++i)
        {
          const double alphaVal = cg->alpha[i];
          const double cVal     = cg->c[i];
          const double norm     = cg->norm[i];
          R += cVal * norm * gaussianRadialPart(r, l, alphaVal);
          dRdr += cVal * norm * gaussianRadialPartDerivative(r, alphaVal, l, 1);
          d2Rdr2 +=
            cVal * norm * gaussianRadialPartDerivative(r, alphaVal, l, 2);
          S += cVal * norm * alphaVal;
        }

      if (r < rTol)
        {
          if (l == 0)
            returnValue = -C * 6.0 * S;
          else
            returnValue = 0.0;
        }

      else
        {
          const double term1 = (2.0 * dRdr / r + d2Rdr2) * Ylm;
          if (std::fabs(theta - 0.0) < angleTol ||
              std::fabs(theta - M_PI) < angleTol)
            {
              const double limitingVal =
                getLimitingValueLaplacian(l, modM, theta, angleTol);
              const double term2 =
                C * (R / (r * r)) * Q * (limitingVal + limitingVal);
              const double term3 =
                -C * m * m * (R / (r * r)) * Q * (limitingVal / 2.0);
              returnValue = (term1 + term2 + term3);
            }

          else
            {
              const double term2 =
                (R / (r * r)) * ((cosTheta / sinTheta) * dYlm[0] + d2Ylm[0]);
              const double term3 =
                (R / (r * r)) * d2Ylm[1] / (sinTheta * sinTheta);
              returnValue = term1 + term2 + term3;
            }
        }
      return returnValue;
    }

    void
    getContractedGaussians(
      const std::string                  basisFileName,
      std::vector<ContractedGaussian *> &contractedGaussians)
    {
      std::unordered_map<char, int> lCharToIntMap = {{'S', 0},
                                                     {'P', 1},
                                                     {'D', 2},
                                                     {'F', 3},
                                                     {'G', 4},
                                                     {'H', 5},
                                                     {'s', 0},
                                                     {'p', 1},
                                                     {'d', 2},
                                                     {'f', 3},
                                                     {'g', 4},
                                                     {'h', 5}};

      // string to read line
      std::string   readLine;
      std::ifstream readFile(basisFileName);
      if (readFile.fail())
        {
          std::string msg = "Unable to open file " + basisFileName;
          utils::throwException(false, msg);
        }

      // ignore the first line which contains the atom symbol
      std::getline(readFile, readLine);
      while (std::getline(readFile, readLine))
        {
          std::istringstream       lineString(readLine);
          std::vector<std::string> words =
            utils::stringOps::split(lineString.str(), "\t ");
          std::string msg =
            "Unable to read l character(s) and the number of contracted Gaussian in file " +
            basisFileName;
          utils::throwException(words.size() >= 2, msg);

          std::string lChars = words[0];
          // check if it's a valid string
          // i.e., it contains one of the following string:
          // "S", "SP", "SPD", SPDF" ...
          std::size_t pos      = lChars.find_first_not_of("SPDFGHspdfgh");
          bool        validStr = (pos == std::string::npos);
          msg =
            "Undefined L character(s) for the contracted Gaussian read in file " +
            basisFileName;
          utils::throwException(validStr, msg);
          const int numLChars = lChars.size();

          // read the number of contracted gaussians
          std::string strNContracted = words[1];
          int         nContracted;
          bool isInt = utils::stringOps::strToInt(strNContracted, nContracted);
          msg =
            "Undefined number of contracted Gaussian in file " + basisFileName;
          utils::throwException(isInt, msg);
          std::vector<double>              alpha(nContracted, 0.0);
          std::vector<std::vector<double>> c(numLChars,
                                             std::vector<double>(nContracted,
                                                                 0.0));
          for (unsigned int i = 0; i < nContracted; ++i)
            {
              if (std::getline(readFile, readLine))
                {
                  if (readLine.empty())
                    {
                      std::string msg =
                        "Empty line found in Gaussian basis file " +
                        basisFileName;
                      utils::throwException(false, msg);
                    }
                  std::istringstream       lineContracted(readLine);
                  std::vector<std::string> wordsAlphaCoeff =
                    utils::stringOps::split(lineContracted.str(), " ");
                  std::string msg =
                    "Unable to read the exponent and the coefficients of contracted Gaussian in file " +
                    basisFileName;
                  utils::throwException(wordsAlphaCoeff.size() >= 1 + numLChars,
                                        msg);
                  std::string alphaStr = wordsAlphaCoeff[0];
                  msg                  = "Undefined value " + alphaStr +
                        " read for the Gaussian exponent in file " +
                        basisFileName;
                  bool isNumber =
                    utils::stringOps::strToDouble(alphaStr, alpha[i]);
                  utils::throwException(isNumber, msg);
                  for (unsigned int j = 0; j < numLChars; ++j)
                    {
                      std::string coeffStr = wordsAlphaCoeff[1 + j];
                      std::string msg =
                        "Undefined value " + coeffStr +
                        " read for the Gaussian coefficient in file " +
                        basisFileName;
                      bool isNumber =
                        utils::stringOps::strToDouble(coeffStr, c[j][i]);
                      utils::throwException(isNumber, msg);
                    }
                }
              else
                {
                  std::string msg =
                    "Undefined row for the contracted Gaussian detected in file" +
                    basisFileName;
                  utils::throwException(false, msg);
                }
            }

          for (unsigned int j = 0; j < numLChars; ++j)
            {
              int  l;
              char lChar = lChars[j];
              try
                {
                  l = lCharToIntMap.at(lChar);
                }
              catch (const std::out_of_range &e)
                {
                  std::string msg =
                    "Character " + std::string(1, lChar) +
                    " does not exist in the "
                    "character to integer map in GaussianBasis.cpp";
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

              for (unsigned int k = 0; k < mList.size(); ++k)
                {
                  ContractedGaussian *cg = new ContractedGaussian;
                  cg->nC                 = nContracted;
                  cg->l                  = l;
                  cg->m                  = mList[k];
                  cg->alpha              = alpha;
                  cg->norm               = getNormConsts(alpha, l);
                  cg->c                  = c[j];
                  contractedGaussians.push_back(cg);
                }
            }
        }
      readFile.close();
    }
  } // namespace

  //
  // Constructor
  //
  GaussianBasis::GaussianBasis(double rTol /*=1e-10*/,
                               double angleTol /*=1e-10*/)
    : d_rTol(rTol)
    , d_angleTol(angleTol)
  {}

  //
  // Destructor
  //
  GaussianBasis::~GaussianBasis()
  {
    for (auto &pair : d_atomToContractedGaussiansPtr)
      {
        std::vector<ContractedGaussian *> &contractedGaussians = pair.second;
        for (ContractedGaussian *cg : contractedGaussians)
          {
            if (cg != nullptr)
              {
                delete cg;
              }
          }
        contractedGaussians.clear();
      }
  }

  void
  GaussianBasis::constructBasisSet(
    const std::vector<std::pair<std::string, std::vector<double>>> &atomCoords,
    const std::unordered_map<std::string, std::string> &atomBasisFileName)
  {
    d_atomSymbolsAndCoords = atomCoords;
    unsigned int natoms    = d_atomSymbolsAndCoords.size();
    d_atomToContractedGaussiansPtr.clear();
    for (const auto &pair : atomBasisFileName)
      {
        const std::string &atomSymbol    = pair.first;
        const std::string &basisFileName = pair.second;
        d_atomToContractedGaussiansPtr[atomSymbol] =
          std::vector<ContractedGaussian *>(0);
        getContractedGaussians(basisFileName,
                               d_atomToContractedGaussiansPtr[atomSymbol]);
      }

    d_gaussianBasisInfo.resize(0);
    for (unsigned int i = 0; i < natoms; ++i)
      {
        const std::string &        atomSymbol = d_atomSymbolsAndCoords[i].first;
        const std::vector<double> &atomCenter =
          d_atomSymbolsAndCoords[i].second;
        unsigned int n = d_atomToContractedGaussiansPtr[atomSymbol].size();
        for (unsigned int j = 0; j < n; ++j)
          {
            GaussianBasisInfo info;
            info.symbol = &atomSymbol;
            info.center = atomCenter.data();
            info.cg     = d_atomToContractedGaussiansPtr[atomSymbol][j];
            d_gaussianBasisInfo.push_back(info);
          }
      }
  }


  int
  GaussianBasis::getNumBasis() const
  {
    return d_gaussianBasisInfo.size();
  }

  std::vector<double>
  GaussianBasis::getBasisValue(const unsigned int         basisId,
                               const std::vector<double> &x) const
  {
    const GaussianBasisInfo & info    = d_gaussianBasisInfo[basisId];
    const double *            x0      = info.center;
    const ContractedGaussian *cg      = info.cg;
    unsigned int              nPoints = round(x.size() / 3);
    std::vector<double>       returnValue(nPoints, 0.0);
    std::vector<double>       dx(3);
    for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint)
      {
        for (unsigned int j = 0; j < 3; ++j)
          dx[j] = x[iPoint * 3 + j] - x0[j];

        returnValue[iPoint] =
          getContractedGaussianValue(cg, dx, d_rTol, d_angleTol);
        // std::cout << returnValue[iPoint] << std::endl;
      }

    return returnValue;
  }


  std::vector<double>
  GaussianBasis::getBasisGradient(const unsigned int         basisId,
                                  const std::vector<double> &x) const
  {
    const GaussianBasisInfo & info    = d_gaussianBasisInfo[basisId];
    const double *            x0      = info.center;
    const ContractedGaussian *cg      = info.cg;
    unsigned int              nPoints = round(x.size() / 3);
    std::vector<double>       returnValue(3 * nPoints, 0.0);
    std::vector<double>       dx(3);
    for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint)
      {
        for (unsigned int j = 0; j < 3; ++j)
          dx[j] = x[iPoint * 3 + j] - x0[j];

        std::vector<double> tmp =
          getContractedGaussianGradient(cg, dx, d_rTol, d_angleTol);
        for (unsigned int j = 0; j < 3; ++j)
          returnValue[iPoint * 3 + j] = tmp[j];
      }
    return returnValue;
  }


  std::vector<double>
  GaussianBasis::getBasisLaplacian(const unsigned int         basisId,
                                   const std::vector<double> &x) const
  {
    const GaussianBasisInfo & info    = d_gaussianBasisInfo[basisId];
    const double *            x0      = info.center;
    const ContractedGaussian *cg      = info.cg;
    unsigned int              nPoints = round(x.size() / 3);
    std::vector<double>       returnValue(nPoints, 0.0);
    std::vector<double>       dx(3);
    for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint)
      {
        for (unsigned int j = 0; j < 3; ++j)
          dx[j] = x[iPoint * 3 + j] - x0[j];

        returnValue[iPoint] =
          getContractedGaussianLaplacian(cg, dx, d_rTol, d_angleTol);
      }
    return returnValue;
  }
} // namespace dftfe
