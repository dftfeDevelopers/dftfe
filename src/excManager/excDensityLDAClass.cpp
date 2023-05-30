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
#include <NNLDA.h>

namespace dftfe
{
  excDensityLDAClass::excDensityLDAClass(xc_func_type *funcXPtr,
                                         xc_func_type *funcCPtr,
                                         bool          isSpinPolarized,
                                         bool          scaleExchange,
                                         bool          computeCorrelation,
                                         double        scaleExchangeFactor)
    : excDensityBaseClass(isSpinPolarized)
  {
    d_familyType = densityFamilyType::LDA;
    d_funcXPtr   = funcXPtr;
    d_funcCPtr   = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNLDAPtr = nullptr;
#endif
  }

  excDensityLDAClass::excDensityLDAClass(xc_func_type *funcXPtr,
                                         xc_func_type *funcCPtr,
                                         bool          isSpinPolarized,
                                         std::string   modelXCInputFile,
                                         bool          scaleExchange,
                                         bool          computeCorrelation,
                                         double        scaleExchangeFactor)
    : excDensityBaseClass(isSpinPolarized)
  {
    d_familyType = densityFamilyType::LDA;
    d_funcXPtr   = funcXPtr;
    d_funcCPtr   = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNLDAPtr = new NNLDA(modelXCInputFile, true);
#endif
  }

  excDensityLDAClass::~excDensityLDAClass()
  {
    if (d_NNLDAPtr != nullptr)
      delete d_NNLDAPtr;
  }

  void
  excDensityLDAClass::computeDensityBasedEnergyDensity(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::vector<double> &outputExchangeEnergyDensity,
    std::vector<double> &outputCorrEnergyDensity) const
  {
    auto rhoValues = rhoData.find(rhoDataAttributes::values)->second;

    xc_lda_exc(d_funcXPtr,
               sizeInput,
               &(*rhoValues)[0],
               &outputExchangeEnergyDensity[0]);
    xc_lda_exc(d_funcCPtr,
               sizeInput,
               &(*rhoValues)[0],
               &outputCorrEnergyDensity[0]);

#ifdef DFTFE_WITH_TORCH
    if (d_NNLDAPtr != nullptr)
      {
        std::vector<double> rhoValuesForNN(2 * sizeInput, 0);
        if (d_isSpinPolarized)
          {
            for (unsigned int i = 0; i < 2 * sizeInput; i++)
              {
                rhoValuesForNN[i] = (*rhoValues)[i];
              }
          }
        else
          {
            for (unsigned int i = 0; i < sizeInput; i++)
              {
                rhoValuesForNN[2 * i]     = 0.5 * (*rhoValues)[i];
                rhoValuesForNN[2 * i + 1] = 0.5 * (*rhoValues)[i];
              }
          }

        std::vector<double> excValuesFromNN(sizeInput, 0);
        d_NNLDAPtr->evaluateexc(&(rhoValuesForNN[0]),
                                sizeInput,
                                &excValuesFromNN[0]);
        for (unsigned int i = 0; i < sizeInput; i++)
          outputExchangeEnergyDensity[i] += excValuesFromNN[i];
      }
#endif
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

    xc_lda_vxc(d_funcXPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*exchangePotentialVal)[0]);
    xc_lda_vxc(d_funcCPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*corrPotentialVal)[0]);

#ifdef DFTFE_WITH_TORCH
    if (d_NNLDAPtr != nullptr)
      {
        std::vector<double> rhoValuesForNN(2 * sizeInput, 0);
        if (d_isSpinPolarized)
          {
            for (unsigned int i = 0; i < 2 * sizeInput; i++)
              {
                rhoValuesForNN[i] = (*rhoValues)[i];
              }
          }
        else
          {
            for (unsigned int i = 0; i < sizeInput; i++)
              {
                rhoValuesForNN[2 * i]     = 0.5 * (*rhoValues)[i];
                rhoValuesForNN[2 * i + 1] = 0.5 * (*rhoValues)[i];
              }
          }

        std::vector<double> excValuesFromNN(2 * sizeInput, 0);
        std::vector<double> vxcValuesFromNN(2 * sizeInput, 0);
        d_NNLDAPtr->evaluatevxc(&(rhoValuesForNN[0]),
                                sizeInput,
                                &excValuesFromNN[0],
                                &vxcValuesFromNN[0]);
        if (d_isSpinPolarized)
          {
            for (unsigned int i = 0; i < 2 * sizeInput; i++)
              (*exchangePotentialVal)[i] += vxcValuesFromNN[i];
          }
        else
          {
            for (unsigned int i = 0; i < sizeInput; i++)
              (*exchangePotentialVal)[i] += vxcValuesFromNN[2 * i];
          }
      }
#endif
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



    xc_lda_fxc(d_funcXPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*der2ExchangeEnergyWithDensity)[0]);

    xc_lda_fxc(d_funcCPtr,
               sizeInput,
               &(*rhoValues)[0],
               &(*der2CorrEnergyWithDensity)[0]);
  }



} // namespace dftfe
