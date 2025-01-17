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

#ifndef DFTFE_GAUSSIANBASIS_H
#define DFTFE_GAUSSIANBASIS_H

#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <string>

#include "AtomicBasis.h"

namespace dftfe
{
  struct ContractedGaussian
  {
    int                 nC; // number of primitive Gaussians that are contracted
    int                 l;  // azimuthal (angular) quantum number
    int                 m;  // magnetic quantum number
    std::vector<double> alpha; // exponent of each of the primtive Gaussians
    std::vector<double> c;     // coefficient of each of the primtive Gaussians
    std::vector<double> norm;  // normalization constant for the radial part of
                               // each of the primitive Gaussians
  };

  struct GaussianBasisInfo
  {
    const std::string *       symbol; // atom symbol
    const double *            center; // atom center coordinates
    const ContractedGaussian *cg;     // pointer to the ContractedGaussian
  };

  class GaussianBasis : public AtomicBasis
  {
  public:
    GaussianBasis(const double rTol     = 1e-10,
                  const double angleTol = 1e-10); // Constructor
    ~GaussianBasis();                             // Destructor

    void
    constructBasisSet(
      const std::vector<std::pair<std::string, std::vector<double>>>
        &                                                 atomCoords,
      const std::unordered_map<std::string, std::string> &atomBasisFileNames);

    int
    getNumBasis() const;

    std::vector<double>
    getBasisValue(const unsigned int         basisId,
                  const std::vector<double> &x) const;

    std::vector<double>
    getBasisGradient(const unsigned int         basisId,
                     const std::vector<double> &x) const;

    std::vector<double>
    getBasisLaplacian(const unsigned int         basisId,
                      const std::vector<double> &x) const;

  private:
    std::unordered_map<std::string, std::vector<ContractedGaussian *>>
                                   d_atomToContractedGaussiansPtr;
    std::vector<GaussianBasisInfo> d_gaussianBasisInfo;
    std::vector<std::pair<std::string, std::vector<double>>>
           d_atomSymbolsAndCoords;
    double d_rTol;
    double d_angleTol;
  };
} // namespace dftfe
#endif // DFTFE_GAUSSIANBASIS_H
