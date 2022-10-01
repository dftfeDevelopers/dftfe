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

#include <excDensityGGAClass.h>

namespace dftfe
{
  excDensityGGAClass::excDensityGGAClass(xc_func_type funcX,
                                         xcfunc_type funcC,
                                         bool scaleExchange,
                                         bool computeCorrelation,
                                         double scaleExchangeFactor):
    excDensityBaseClass(funcX,funcC, scaleExchange, computeCorrelation,scaleExchangeFactor )
  {
    d_familyType = densityFamilyType::GGA;

  }

  void excDensityGGAClass::computeDensityBasedEnergyDensity(unsigned int sizeInput,
                                                       const std::map<rhoDataAttributes,const std::vector<double>*> &rhoData,
                                                       std::vector<double> &outputExchangeEnergyDensity,
                                                       std::vector<double> &outputCorrEnergyDensity) const
  {

    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;
    auto rhoSigmaGradValues = rhoData.find(rhoDataAttributes::sigmaGradValue)->second;


    // This * is not neccessary, unnessary referencing and de-referencing
    xc_gga_exc(&d_funcX,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &exchangeEnergyDensity[0]);
    xc_gga_exc(&d_funcC,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &corrEnergyDensity[0]);

  }

  void excDensityGGAClass::computeDensityBasedVxc(unsigned int sizeInput,
                                             const std::map<rhoDataAttributes,const std::vector<double>*> &rhoData,
                                             std::map<VeffOutputDataAttributes,const std::vector<double>*> &outputDerExchangeEnergy,
                                             std::map<VeffOutputDataAttributes,const std::vector<double>*> &outputDerCorrEnergy) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;
    auto rhoSigmaGradValues = rhoData.find(rhoDataAttributes::sigmaGradValue)->second;


    auto derExchangeEnergyWithDensity = outputDerExchangeEnergy.
                                find(derEnergyWithDensity)->second;
    auto derExchangeEnergyWithSigmaGradDensity = outputDerExchangeEnergy.
                                find(derEnergyWithSigmaGradDensity)->second;

    auto derCorrEnergyWithDensity = outputDerCorrEnergy.
                            find(derEnergyWithDensity)->second;

    auto derCorrEnergyWithSigmaGradDensity = outputDerCorrEnergy.
                                                 find(derEnergyWithSigmaGradDensity)->second;


    xc_gga_vxc(&d_funcX,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*derExchangeEnergyWithDensity)[0],
               &(*derExchangeEnergyWithSigmaGradDensity)[0]);
    xc_gga_vxc(&d_funcC,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*derCorrEnergyWithDensity)[0],
               &(*derCorrEnergyWithSigmaGradDensity)[0]);
  }



}