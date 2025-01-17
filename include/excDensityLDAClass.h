// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#include <xc.h>
#include <ExcSSDFunctionalBaseClass.h>
namespace dftfe
{
  class NNLDA;
  template <dftfe::utils::MemorySpace memorySpace>
  class excDensityLDAClass : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    excDensityLDAClass(std::shared_ptr<xc_func_type> funcXPtr,
                       std::shared_ptr<xc_func_type> funcCPtr);

    excDensityLDAClass(std::shared_ptr<xc_func_type> funcXPtr,
                       std::shared_ptr<xc_func_type> funcCPtr,
                       std::string                   modelXCInputFile);

    ~excDensityLDAClass();

    void
    computeRhoTauDependentXCData(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    quadPoints,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &xDataOut,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &cDataout) const override;

    void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const override;

    void
    applyWaveFunctionDependentFuncDerWrtPsi(
      const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        &                                                                src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const unsigned int inputVecSize,
      const unsigned int kPointIndex,
      const unsigned int spinIndex) override;

    void
    updateWaveFunctionDependentFuncDerWrtPsi(
      const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
      const std::vector<double> &kPointWeights) override;
    void
    computeWaveFunctionDependentExcEnergy(
      const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
      const std::vector<double> &kPointWeights) override;

    double
    getWaveFunctionDependentExcEnergy() override;

    double
    getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi() override;

    void
    reinitKPointDependentVariables(unsigned int kPointIndex) override;

  private:
    NNLDA *                       d_NNLDAPtr;
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
  };
} // namespace dftfe

#endif // DFTFE_EXCDENSIYLDACLASS_H
