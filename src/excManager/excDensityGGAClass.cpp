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
                                         xc_func_type funcC,
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
               &outputExchangeEnergyDensity[0]);
    xc_gga_exc(&d_funcC,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &outputCorrEnergyDensity[0]);

  }

  void excDensityGGAClass::computeDensityBasedVxc(unsigned int sizeInput,
                                             const std::map<rhoDataAttributes,const std::vector<double>*> &rhoData,
                                             std::map<VeffOutputDataAttributes,std::vector<double>*> &outputDer2ExchangeEnergy,
                                             std::map<VeffOutputDataAttributes,std::vector<double>*> &outputDer2CorrEnergy) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;
    auto rhoSigmaGradValues = rhoData.find(rhoDataAttributes::sigmaGradValue)->second;


    auto derExchangeEnergyWithDensity = outputDerExchangeEnergy.
                                        find(VeffOutputDataAttributes::derEnergyWithDensity)->second;
    auto derExchangeEnergyWithSigmaGradDensity = outputDerExchangeEnergy.
                                                 find(VeffOutputDataAttributes::derEnergyWithSigmaGradDensity)->second;

    auto derCorrEnergyWithDensity = outputDerCorrEnergy.
                                    find(VeffOutputDataAttributes::derEnergyWithDensity)->second;

    auto derCorrEnergyWithSigmaGradDensity = outputDerCorrEnergy.
                                             find(VeffOutputDataAttributes::derEnergyWithSigmaGradDensity)->second;


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


  void excDensityGGAClass::computeDensityBasedFxc(unsigned int sizeInput,
                                             const std::map<rhoDataAttributes,const std::vector<double>*> &rhoData,
                                             std::map<fxcOutputDataAttributes,std::vector<double>*> &outputDer2ExchangeEnergy,
                                             std::map<fxcOutputDataAttributes,std::vector<double>*> &outputDer2CorrEnergy) const
  {

    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;
    auto rhoSigmaGradValues = rhoData.find(rhoDataAttributes::sigmaGradValue)->second;


    auto der2ExchangeEnergyWithDensity = outputDer2ExchangeEnergy.
                                         find(fxcOutputDataAttributes::der2EnergyWithDensity)->second;
    auto der2ExchangeEnergyWithDensitySigma = outputDer2ExchangeEnergy.
                                              find(fxcOutputDataAttributes::der2EnergyWithDensitySigma)->second;

    auto der2ExchangeEnergyWithSigmaGradDensity = outputDer2ExchangeEnergy.
                                                  find(VeffOutputDataAttributes::der2EnergyWithSigma)->second;

    auto der2CorrEnergyWithDensity = outputDer2CorrEnergy.
                                     find(fxcOutputDataAttributes::der2EnergyWithDensity)->second;
    auto der2CorrEnergyWithDensitySigma = outputDer2CorrEnergy.
                                          find(fxcOutputDataAttributes::der2EnergyWithDensitySigma)->second;

    auto der2CorrEnergyWithSigmaGradDensity = outputDer2CorrEnergy.
                                              find(VeffOutputDataAttributes::der2EnergyWithSigma)->second;



    xc_gga_fxc(&d_funcX,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*der2ExchangeEnergyWithDensity)[0],
               &(*der2ExchangeEnergyWithDensitySigma)[0],
               &(*der2ExchangeEnergyWithSigmaGradDensity)[0]);

    xc_gga_fxc(&d_funcC,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*der2CorrEnergyWithDensity)[0],
               &(*der2CorrEnergyWithDensitySigma)[0],
               &(*der2CorrEnergyWithSigmaGradDensity)[0]);

  }



}
