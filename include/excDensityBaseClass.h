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


#ifndef DFTFE_EXCDENSITYBASECLASS_H
#define DFTFE_EXCDENSITYBASECLASS_H

//#include <headers.h>
#include <xc.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
//#include <linearAlgebraOperations.h>

namespace dftfe
{
  enum class densityFamilyType
  {
    LDA,
    GGA
  };
  // enum class for identifying the relevant objects for exc manager class
  enum class rhoDataAttributes
  {
    values,
    sigmaGradValue
  };

  enum class VeffOutputDataAttributes
  {
    derEnergyWithDensity,
    derEnergyWithSigmaGradDensity
  };

  enum class fxcOutputDataAttributes
  {
    der2EnergyWithDensity,
    der2EnergyWithDensitySigma,
    der2EnergyWithSigma
  };


  class excDensityBaseClass
  {
  public:
    excDensityBaseClass(bool          isSpinPolarized);

    virtual void
    computeDensityBasedEnergyDensity(
      unsigned int                                                    sizeInput,
      const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
      std::vector<double> &outputExchangeEnergyDensity,
      std::vector<double> &outputCorrEnergyDensity) const = 0;

    virtual void
    computeDensityBasedVxc(
      unsigned int                                                    sizeInput,
      const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
      std::map<VeffOutputDataAttributes, std::vector<double> *>
        &outputDerExchangeEnergy,
      std::map<VeffOutputDataAttributes, std::vector<double> *>
        &outputDerCorrEnergy) const = 0;

    virtual void
    computeDensityBasedFxc(
      unsigned int                                                    sizeInput,
      const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
      std::map<fxcOutputDataAttributes, std::vector<double> *>
        &outputDer2ExchangeEnergy,
      std::map<fxcOutputDataAttributes, std::vector<double> *>
        &outputDer2CorrEnergy) const = 0;

    densityFamilyType
    getDensityBasedFamilyType() const;

  protected:
    densityFamilyType d_familyType;
    bool              d_isSpinPolarized;
  };

} // namespace dftfe

#endif // DFTFE_EXCDENSITYBASECLASS_H
