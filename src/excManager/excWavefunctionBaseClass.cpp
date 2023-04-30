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

#include <excWavefunctionBaseClass.h>
#include <excDensityGGAClass.h>
#include <excDensityLDAClass.h>

namespace dftfe
{
  excWavefunctionBaseClass::excWavefunctionBaseClass(
    densityFamilyType densityFamilyTypeObj,
    xc_func_type *    funcXPtr,
    xc_func_type *    funcCPtr,
    bool              isSpinPolarized,
    double            factorForWavefunctionDependent,
    bool              scaleExchange,
    bool              computeCorrelation,
    double            scaleExchangeFactor)
  {
    switch (densityFamilyTypeObj)
      {
        case densityFamilyType::LDA:
          d_excDensityBaseClassPtr =
            new excDensityLDAClass(funcXPtr,
                                   funcCPtr,
                                   isSpinPolarized,
                                   scaleExchange,
                                   computeCorrelation,
                                   scaleExchangeFactor);
          break;
        case densityFamilyType::GGA:
          d_excDensityBaseClassPtr =
            new excDensityGGAClass(funcXPtr,
                                   funcCPtr,
                                   isSpinPolarized,
                                   scaleExchange,
                                   computeCorrelation,
                                   scaleExchangeFactor);
          break;
        default:
          std::cout << " Error in deciphering "
                       "family type of density based exc functional\n";
          break;
      }
  }


  excWavefunctionBaseClass::excWavefunctionBaseClass(
    densityFamilyType densityFamilyTypeObj,
    xc_func_type *    funcXPtr,
    xc_func_type *    funcCPtr,
    bool              isSpinPolarized,
    std::string       modelXCInputFile,
    double            factorForWavefunctionDependent,
    bool              scaleExchange,
    bool              computeCorrelation,
    double            scaleExchangeFactor)
  {
    switch (densityFamilyTypeObj)
      {
        case densityFamilyType::LDA:
          d_excDensityBaseClassPtr =
            new excDensityLDAClass(funcXPtr,
                                   funcCPtr,
                                   isSpinPolarized,
                                   modelXCInputFile,
                                   scaleExchange,
                                   computeCorrelation,
                                   scaleExchangeFactor);
          break;
        case densityFamilyType::GGA:
          d_excDensityBaseClassPtr =
            new excDensityGGAClass(funcXPtr,
                                   funcCPtr,
                                   isSpinPolarized,
                                   modelXCInputFile,
                                   scaleExchange,
                                   computeCorrelation,
                                   scaleExchangeFactor);
          break;
        default:
          std::cout << " Error in deciphering "
                       "family type of density based exc functional\n";
          break;
      }
  }

  excWavefunctionBaseClass::~excWavefunctionBaseClass()
  {
    delete d_excDensityBaseClassPtr;
  }

  void
  excWavefunctionBaseClass::computeDensityBasedEnergyDensity(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::vector<double> &outputExchangeEnergyDensity,
    std::vector<double> &outputCorrEnergyDensity) const
  {
    d_excDensityBaseClassPtr->computeDensityBasedEnergyDensity(
      sizeInput, rhoData, outputExchangeEnergyDensity, outputCorrEnergyDensity);
  }

  void
  excWavefunctionBaseClass::computeDensityBasedVxc(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::map<VeffOutputDataAttributes, std::vector<double> *>
      &outputDerExchangeEnergy,
    std::map<VeffOutputDataAttributes, std::vector<double> *>
      &outputDerCorrEnergy) const
  {
    d_excDensityBaseClassPtr->computeDensityBasedVxc(sizeInput,
                                                     rhoData,
                                                     outputDerExchangeEnergy,
                                                     outputDerCorrEnergy);
  }

  void
  excWavefunctionBaseClass::computeDensityBasedFxc(
    unsigned int                                                    sizeInput,
    const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
    std::map<fxcOutputDataAttributes, std::vector<double> *>
      &outputDer2ExchangeEnergy,
    std::map<fxcOutputDataAttributes, std::vector<double> *>
      &outputDer2CorrEnergy) const
  {
    d_excDensityBaseClassPtr->computeDensityBasedFxc(sizeInput,
                                                     rhoData,
                                                     outputDer2ExchangeEnergy,
                                                     outputDer2CorrEnergy);
  }

  densityFamilyType
  excWavefunctionBaseClass::getDensityBasedFamilyType() const
  {
    return d_excDensityBaseClassPtr->getDensityBasedFamilyType();
  }

  wavefunctionFamilyType
  excWavefunctionBaseClass::getWavefunctionBasedFamilyType() const
  {
    return d_wavefunctionFamilyType;
  }


} // namespace dftfe
