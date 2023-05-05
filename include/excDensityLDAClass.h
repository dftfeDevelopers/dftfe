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

#ifndef DFTFE_EXCDENSIYLDACLASS_H
#define DFTFE_EXCDENSIYLDACLASS_H

//#include <NNLDA.h>
#include <excDensityBaseClass.h>

namespace dftfe
{
  class NNLDA;
  class excDensityLDAClass : public excDensityBaseClass
  {
  public:
    excDensityLDAClass(xc_func_type *funcXPtr,
                       xc_func_type *funcCPtr,
                       bool          isSpinPolarized,
                       bool          scaleExchange,
                       bool          computeCorrelation,
                       double        scaleExchangeFactor);

    excDensityLDAClass(xc_func_type *funcXPtr,
                       xc_func_type *funcCPtr,
                       bool          isSpinPolarized,
                       std::string   modelXCInputFile,
                       bool          scaleExchange,
                       bool          computeCorrelation,
                       double        scaleExchangeFactor);

    ~excDensityLDAClass();

    void
    computeDensityBasedEnergyDensity(
      unsigned int                                                    sizeInput,
      const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
      std::vector<double> &outputExchangeEnergyDensity,
      std::vector<double> &outputCorrEnergyDensity) const override;

    void
    computeDensityBasedVxc(
      unsigned int                                                    sizeInput,
      const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
      std::map<VeffOutputDataAttributes, std::vector<double> *>
        &outputDerExchangeEnergy,
      std::map<VeffOutputDataAttributes, std::vector<double> *>
        &outputDerCorrEnergy) const override;

    void
    computeDensityBasedFxc(
      unsigned int                                                    sizeInput,
      const std::map<rhoDataAttributes, const std::vector<double> *> &rhoData,
      std::map<fxcOutputDataAttributes, std::vector<double> *>
        &outputDer2ExchangeEnergy,
      std::map<fxcOutputDataAttributes, std::vector<double> *>
        &outputDer2CorrEnergy) const override;


  private:
    NNLDA *d_NNLDAPtr;
  };
} // namespace dftfe

#endif // DFTFE_EXCDENSIYLDACLASS_H
