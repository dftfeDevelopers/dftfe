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
// @author Vishal Subramanian
//

#include <excDensityLDAClass.h>

namespace dftfe
{
  excDensityLDAClass::excDensityLDAClass(xc_func_type funcX,
                                         xc_func_type funcC,
                                         bool         scaleExchange,
                                         bool         computeCorrelation,
                                         double       scaleExchangeFactor)
    : excDensityBaseClass(funcX,
                          funcC,
                          scaleExchange,
                          computeCorrelation,
                          scaleExchangeFactor)
  {
    d_familyType = densityFamilyType::LDA;
  }

  void
  excDensityLDAClass::computeDensityBasedEnergyDensity(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::vector<double> &outputExchangeEnergyDensity,
    std::vector<double> &outputCorrEnergyDensity) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;

    xc_lda_exc(&d_funcX,
               sizeInput,
               &(*rhoValues)[0],
               &outputExchangeEnergyDensity[0]);
    xc_lda_exc(&d_funcC,
               sizeInput,
               &(*rhoValues)[0],
               &outputCorrEnergyDensity[0]);
  }

  void
  excDensityLDAClass::computeDensityBasedVxc(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::map<VeffOutputDataAttributes, std::vector<double> *>
      &outputDerExchangeEnergy,
    std::map<VeffOutputDataAttributes, std::vector<double> *>
      &outputDerCorrEnergy) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;

    auto exchangePotentialVal =
      outputDerExchangeEnergy
        .find(VeffOutputDataAttributes::derEnergyWithDensity)
        ->second;

    auto corrPotentialVal =
      outputDerCorrEnergy.find(VeffOutputDataAttributes::derEnergyWithDensity)
        ->second;

    xc_lda_vxc(&d_funcX,
               sizeInput,
               &(*rhoValues)[0],
               &(*exchangePotentialVal)[0]);
    xc_lda_vxc(&d_funcC, sizeInput, &(*rhoValues)[0], &(*corrPotentialVal)[0]);
  }

  void
  excDensityLDAClass::computeDensityBasedFxc(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::map<fxcOutputDataAttributes, std::vector<double> *>
      &outputDer2ExchangeEnergy,
    std::map<fxcOutputDataAttributes, std::vector<double> *>
      &outputDer2CorrEnergy) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;

    auto der2ExchangeEnergyWithDensity =
      outputDer2ExchangeEnergy
        .find(fxcOutputDataAttributes::der2EnergyWithDensity)
        ->second;

    auto der2CorrEnergyWithDensity =
      outputDer2CorrEnergy.find(fxcOutputDataAttributes::der2EnergyWithDensity)
        ->second;



    xc_lda_fxc(&d_funcX,
               sizeInput,
               &(*rhoValues)[0],
               &(*der2ExchangeEnergyWithDensity)[0]);

    xc_lda_fxc(&d_funcC,
               sizeInput,
               &(*rhoValues)[0],
               &(*der2CorrEnergyWithDensity)[0]);
  }



} // namespace dftfe
