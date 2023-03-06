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
#include <NNGGA.h>

namespace dftfe
{
  excDensityGGAClass::excDensityGGAClass(xc_func_type *funcXPtr,
                                         xc_func_type *funcCPtr,
                                         bool          isSpinPolarized,
                                         bool          scaleExchange,
                                         bool          computeCorrelation,
                                         double        scaleExchangeFactor)
    : excDensityBaseClass(funcXPtr,
                          funcCPtr,
                          isSpinPolarized,
                          scaleExchange,
                          computeCorrelation,
                          scaleExchangeFactor)
  {
    d_familyType = densityFamilyType::GGA;
    d_NNGGAPtr   = nullptr;
  }


  excDensityGGAClass::excDensityGGAClass(xc_func_type *funcXPtr,
                                         xc_func_type *funcCPtr,
                                         bool          isSpinPolarized,
                                         std::string   modelXCInputFile,
                                         bool          scaleExchange,
                                         bool          computeCorrelation,
                                         double        scaleExchangeFactor)
    : excDensityBaseClass(funcXPtr,
                          funcCPtr,
                          isSpinPolarized,
                          scaleExchange,
                          computeCorrelation,
                          scaleExchangeFactor)
  {
    d_familyType = densityFamilyType::GGA;
    d_NNGGAPtr   = new NNGGA(modelXCInputFile, true);
  }

  excDensityGGAClass::~excDensityGGAClass()
  {
    if (d_NNGGAPtr != nullptr)
      delete d_NNGGAPtr;
  }

  void
  excDensityGGAClass::computeDensityBasedEnergyDensity(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::vector<double> &outputExchangeEnergyDensity,
    std::vector<double> &outputCorrEnergyDensity) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;
    auto rhoSigmaGradValues =
      rhoData.find(rhoDataAttributes::sigmaGradValue)->second;


    // This * is not neccessary, unnessary referencing and de-referencing
    xc_gga_exc(d_funcXPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &outputExchangeEnergyDensity[0]);
    xc_gga_exc(d_funcCPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &outputCorrEnergyDensity[0]);

    if (d_NNGGAPtr != nullptr)
      {
        std::vector<double> rhoValuesForNN(2 * sizeInput, 0);
        std::vector<double> sigmaValuesForNN(3 * sizeInput, 0);
        if (d_isSpinPolarized)
          {
            for (unsigned int i = 0; i < 2 * sizeInput; i++)
              rhoValuesForNN[i] = (*rhoValues)[i];

            for (unsigned int i = 0; i < 3 * sizeInput; i++)
              sigmaValuesForNN[i] = (*rhoSigmaGradValues)[i];
          }
        else
          {
            for (unsigned int i = 0; i < sizeInput; i++)
              {
                rhoValuesForNN[2 * i]     = 0.5 * (*rhoValues)[i];
                rhoValuesForNN[2 * i + 1] = 0.5 * (*rhoValues)[i];
              }

            for (unsigned int i = 0; i < sizeInput; i++)
              {
                sigmaValuesForNN[3 * i]     = (*rhoSigmaGradValues)[i] / 4.0;
                sigmaValuesForNN[3 * i + 1] = (*rhoSigmaGradValues)[i] / 2.0;
                sigmaValuesForNN[3 * i + 2] = (*rhoSigmaGradValues)[i] / 4.0;
              }
          }

        std::vector<double> excValuesFromNN(sizeInput, 0);
        d_NNGGAPtr->evaluateexc(&(rhoValuesForNN[0]),
                                &(sigmaValuesForNN[0]),
                                sizeInput,
                                &excValuesFromNN[0]);
        for (unsigned int i = 0; i < sizeInput; i++)
          outputExchangeEnergyDensity[i] += excValuesFromNN[i];
      }
  }

  void
  excDensityGGAClass::computeDensityBasedVxc(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::map<VeffOutputDataAttributes, std::vector<double> *>
      &outputDerExchangeEnergy,
    std::map<VeffOutputDataAttributes, std::vector<double> *>
      &outputDerCorrEnergy) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;
    auto rhoSigmaGradValues =
      rhoData.find(rhoDataAttributes::sigmaGradValue)->second;

    auto derExchangeEnergyWithDensity =
      outputDerExchangeEnergy
        .find(VeffOutputDataAttributes::derEnergyWithDensity)
        ->second;
    auto derExchangeEnergyWithSigmaGradDensity =
      outputDerExchangeEnergy
        .find(VeffOutputDataAttributes::derEnergyWithSigmaGradDensity)
        ->second;

    auto derCorrEnergyWithDensity =
      outputDerCorrEnergy.find(VeffOutputDataAttributes::derEnergyWithDensity)
        ->second;

    auto derCorrEnergyWithSigmaGradDensity =
      outputDerCorrEnergy
        .find(VeffOutputDataAttributes::derEnergyWithSigmaGradDensity)
        ->second;


    xc_gga_vxc(d_funcXPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*derExchangeEnergyWithDensity)[0],
               &(*derExchangeEnergyWithSigmaGradDensity)[0]);
    xc_gga_vxc(d_funcCPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*derCorrEnergyWithDensity)[0],
               &(*derCorrEnergyWithSigmaGradDensity)[0]);

    if (d_NNGGAPtr != nullptr)
      {
        std::vector<double> rhoValuesForNN(2 * sizeInput, 0);
        std::vector<double> sigmaValuesForNN(3 * sizeInput, 0);
        if (d_isSpinPolarized)
          {
            for (unsigned int i = 0; i < 2 * sizeInput; i++)
              rhoValuesForNN[i] = (*rhoValues)[i];

            for (unsigned int i = 0; i < 3 * sizeInput; i++)
              sigmaValuesForNN[i] = (*rhoSigmaGradValues)[i];
          }
        else
          {
            for (unsigned int i = 0; i < sizeInput; i++)
              {
                rhoValuesForNN[2 * i]     = 0.5 * (*rhoValues)[i];
                rhoValuesForNN[2 * i + 1] = 0.5 * (*rhoValues)[i];
              }

            for (unsigned int i = 0; i < sizeInput; i++)
              {
                sigmaValuesForNN[3 * i]     = (*rhoSigmaGradValues)[i] / 4.0;
                sigmaValuesForNN[3 * i + 1] = (*rhoSigmaGradValues)[i] / 2.0;
                sigmaValuesForNN[3 * i + 2] = (*rhoSigmaGradValues)[i] / 4.0;
              }
          }

        std::vector<double> excValuesFromNN(sizeInput, 0);
        std::vector<double> vxcValuesFromNN(5 * sizeInput, 0);
        d_NNGGAPtr->evaluatevxc(&(rhoValuesForNN[0]),
                                &(sigmaValuesForNN[0]),
                                sizeInput,
                                &excValuesFromNN[0],
                                &vxcValuesFromNN[0]);
        if (d_isSpinPolarized)
          {
            for (unsigned int i = 0; i < sizeInput; i++)
              {
                (*derExchangeEnergyWithDensity)[2 * i] +=
                  vxcValuesFromNN[5 * i];
                (*derExchangeEnergyWithDensity)[2 * i + 1] +=
                  vxcValuesFromNN[5 * i + 1];
                (*derExchangeEnergyWithSigmaGradDensity)[3 * i] +=
                  vxcValuesFromNN[5 * i + 2];
                (*derExchangeEnergyWithSigmaGradDensity)[3 * i + 1] +=
                  vxcValuesFromNN[5 * i + 3];
                (*derExchangeEnergyWithSigmaGradDensity)[3 * i + 2] +=
                  vxcValuesFromNN[5 * i + 4];
              }
          }
        else
          {
            for (unsigned int i = 0; i < sizeInput; i++)
              {
                (*derExchangeEnergyWithDensity)[i] +=
                  vxcValuesFromNN[3 * i] + vxcValuesFromNN[3 * i + 1];
                (*derExchangeEnergyWithSigmaGradDensity)[i] +=
                  vxcValuesFromNN[3 * i + 2];
              }
          }
      }
  }


  void
  excDensityGGAClass::computeDensityBasedFxc(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::map<fxcOutputDataAttributes, std::vector<double> *>
      &outputDer2ExchangeEnergy,
    std::map<fxcOutputDataAttributes, std::vector<double> *>
      &outputDer2CorrEnergy) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;
    auto rhoSigmaGradValues =
      rhoData.find(rhoDataAttributes::sigmaGradValue)->second;


    auto der2ExchangeEnergyWithDensity =
      outputDer2ExchangeEnergy
        .find(fxcOutputDataAttributes::der2EnergyWithDensity)
        ->second;
    auto der2ExchangeEnergyWithDensitySigma =
      outputDer2ExchangeEnergy
        .find(fxcOutputDataAttributes::der2EnergyWithDensitySigma)
        ->second;

    auto der2ExchangeEnergyWithSigmaGradDensity =
      outputDer2ExchangeEnergy
        .find(fxcOutputDataAttributes::der2EnergyWithSigma)
        ->second;

    auto der2CorrEnergyWithDensity =
      outputDer2CorrEnergy.find(fxcOutputDataAttributes::der2EnergyWithDensity)
        ->second;
    auto der2CorrEnergyWithDensitySigma =
      outputDer2CorrEnergy
        .find(fxcOutputDataAttributes::der2EnergyWithDensitySigma)
        ->second;

    auto der2CorrEnergyWithSigmaGradDensity =
      outputDer2CorrEnergy.find(fxcOutputDataAttributes::der2EnergyWithSigma)
        ->second;



    xc_gga_fxc(d_funcXPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*der2ExchangeEnergyWithDensity)[0],
               &(*der2ExchangeEnergyWithDensitySigma)[0],
               &(*der2ExchangeEnergyWithSigmaGradDensity)[0]);

    xc_gga_fxc(d_funcCPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*rhoSigmaGradValues)[0],
               &(*der2CorrEnergyWithDensity)[0],
               &(*der2CorrEnergyWithDensitySigma)[0],
               &(*der2CorrEnergyWithSigmaGradDensity)[0]);
  }
} // namespace dftfe
