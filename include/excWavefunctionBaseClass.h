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

#ifndef DFTFE_EXCWAVEFUNCTIONBASECLASS_H
#define DFTFE_EXCWAVEFUNCTIONBASECLASS_H

#include <excDensityGGAClass.h>
#include <excDensityLDAClass.h>

namespace dftfe
{
  enum class wavefunctionFamilyType
  {
    NONE,
    TAU,
    SCALED_FOCK,
    HUBBARD
  };

  class excWavefunctionBaseClass
  {
  public:
    excWavefunctionBaseClass(
      densityFamilyType densityFamilyTypeObj,
      xc_func_type funcX,
                             xc_func_type funcC,
                             double factorForWavefunctionDependent,
			     bool scaleExchange,
                             bool computeCorrelation,
                             double scaleExchangeFactor);
    ~excWavefunctionBaseClass();
    void computeDensityBasedEnergyDensity(unsigned int sizeInput,
                                     const std::map<rhoDataAttributes,const std::vector<double>*> &rhoData,
                                     std::vector<double> &outputExchangeEnergyDensity,
                                     std::vector<double> &outputCorrEnergyDensity) const ;
    void computeDensityBasedVxc(unsigned int sizeInput,
                           const std::map<rhoDataAttributes,const std::vector<double>*> &rhoData,
                           std::map<VeffOutputDataAttributes,std::vector<double>*> &outputDerExchangeEnergy,
                           std::map<VeffOutputDataAttributes,std::vector<double>*> &outputDerCorrEnergy) const ;
   void computeDensityBasedFxc(unsigned int sizeInput,
                           const std::map<rhoDataAttributes,const std::vector<double>*> &rhoData,
                           std::map<fxcOutputDataAttributes,std::vector<double>*> &outputDer2ExchangeEnergy,
                           std::map<fxcOutputDataAttributes,std::vector<double>*> &outputDer2CorrEnergy) const ;

    virtual void applyWaveFunctionDependentVxc() const = 0;
    virtual void updateWaveFunctionDependentVxc() const = 0;
    virtual double computeWaveFunctionDependentExcEnergy() const = 0;

    densityFamilyType getDensityBasedFamilyType() const ;

    wavefunctionFamilyType getWavefunctionBasedFamilyType() const;

  protected:
    excDensityBaseClass *d_excDensityBaseClassPtr;
    wavefunctionFamilyType d_wavefunctionFamilyType;
  };
}

#endif // DFTFE_EXCWAVEFUNCTIONBASECLASS_H
