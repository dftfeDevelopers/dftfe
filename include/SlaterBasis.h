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
// @author Arghadwip Paul, Bikash Kanungo
//

#ifndef DFTFE_SLATERBASIS_H
#define DFTFE_SLATERBASIS_H

#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include "AtomicBasis.h"
namespace dftfe
{
  struct SlaterPrimitive
  {
    int    n;         // principal quantum number
    int    l;         // azimuthal (angular) quantum number
    int    m;         // magnetic quantum number
    double alpha;     // exponent of the basis
    double normConst; // normalization constant for the radial part
  };

  struct SlaterBasisInfo
  {
    const std::string *    symbol; // atom symbol
    const double *         center; // atom center coordinates
    const SlaterPrimitive *sp;     // pointer to the SlaterPrimitive
  };

  class SlaterBasis : public AtomicBasis
  {
  public:
    SlaterBasis(const double rTol     = 1e-10,
                const double angleTol = 1e-10); // Constructor
    ~SlaterBasis();                             // Destructor


    virtual void
    constructBasisSet(
      const std::vector<std::pair<std::string, std::vector<double>>>
        &                                                 atomCoords,
      const std::unordered_map<std::string, std::string> &atomBasisFileNames)
      override;

    virtual int
    getNumBasis() const override;

    virtual std::vector<double>
    getBasisValue(const unsigned int         basisId,
                  const std::vector<double> &x) const override;

    virtual std::vector<double>
    getBasisGradient(const unsigned int         basisId,
                     const std::vector<double> &x) const override;

    virtual std::vector<double>
    getBasisLaplacian(const unsigned int         basisId,
                      const std::vector<double> &x) const override;

  private:
    std::unordered_map<std::string, std::vector<SlaterPrimitive *>>
                                 d_atomToSlaterPrimitivesPtr;
    std::vector<SlaterBasisInfo> d_slaterBasisInfo;
    std::vector<std::pair<std::string, std::vector<double>>>
           d_atomSymbolsAndCoords;
    double d_rTol;
    double d_angleTol;
  };
} // namespace dftfe
#endif // DFTFE_SLATERBASIS_H
