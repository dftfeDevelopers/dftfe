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
#ifndef DFTFE_EXCDENSITYGGACLASS_H
#define DFTFE_EXCDENSITYGGACLASS_H

#include <xc.h>
#include <ExcSSDFunctionalBaseClass.h>
namespace dftfe
{
  class NNGGA;
  template <dftfe::utils::MemorySpace memorySpace>
  class excDensityGGAClass : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    excDensityGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                       std::shared_ptr<xc_func_type> funcCPtr);


    excDensityGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                       std::shared_ptr<xc_func_type> funcCPtr,
                       std::string                   modelXCInputFile);


    ~excDensityGGAClass();



    void
    computeRhoTauDependentXCData(
      AuxDensityMatrix<memorySpace>             &auxDensityMatrix,
      const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
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
                                                                        &src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const dftfe::uInt inputVecSize,
      const dftfe::uInt kPointIndex,
      const dftfe::uInt spinIndex) override;

    /*
     * @brief The apply function that will be called in HXCheby() with single precision.
     * The distribute() and updateGhostValues() for src
     * has to be called before this function.
     * Similarly for dst, accumulateLocallyOwned() should be called in HX()
     * after this function is called. param[in] src The input vector param[out]
     * dst The output vector param[in] inputVecSize The size of the input vector
     * param[in] kPointIndex the k point for which the HX() is called
     * param[in] spinIndex the spin index for which the HX() is called
     */
    void
    applyWaveFunctionDependentFuncDerWrtPsi(
      const dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                              memorySpace> &src,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                       &dst,
      const dftfe::uInt inputVecSize,
      const dftfe::uInt kPointIndex,
      const dftfe::uInt spinIndex) override;

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
    reinitKPointDependentVariables(dftfe::uInt kPointIndex) override;

  private:
    NNGGA                        *d_NNGGAPtr;
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
    std::vector<double>           d_spacingFDStencil;
    dftfe::uInt                   d_vxcDivergenceTermFDStencilSize;
  };
} // namespace dftfe
#endif // DFTFE_EXCDENSITYGGACLASS_H
