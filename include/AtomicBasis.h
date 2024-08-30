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

#ifndef DFTFE_ATOMICBASIS_H
#define DFTFE_ATOMICBASIS_H

#include <vector>
#include <unordered_map>
#include <string>
#include <utility>

namespace dftfe
{
  class AtomicBasis
  {
  public:
    enum class BasisType
    {
      SLATER,
      GAUSSIAN,
      BESSELORTHO
    };

    // Default destructor
    ~AtomicBasis() = default;

    virtual void
    constructBasisSet(
      const std::vector<std::pair<std::string, std::vector<double>>>
        &atomCoords,
      const std::unordered_map<std::string, std::string>
        &atomBasisFileNames) = 0;

    virtual int
    getNumBasis() const = 0;

    virtual std::vector<double>
    getBasisValue(const unsigned int         basisId,
                  const std::vector<double> &x) const = 0;

    virtual std::vector<double>
    getBasisGradient(const unsigned int         basisId,
                     const std::vector<double> &x) const = 0;

    virtual std::vector<double>
    getBasisLaplacian(const unsigned int         basisId,
                      const std::vector<double> &x) const = 0;
  };
} // namespace dftfe
#endif // DFTFE_ATOMICBASIS_H
