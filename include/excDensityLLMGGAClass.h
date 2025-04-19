//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_EXCDENSITYLLMGGACLASS_H
#define DFTFE_EXCDENSITYLLMGGACLASS_H

#include <xc.h>
#include <ExcSSDFunctionalBaseClass.h>
namespace dftfe
{
  class NNLLMGGA;
  template <dftfe::utils::MemorySpace memorySpace>
  class excDensityLLMGGAClass : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    excDensityLLMGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                          std::shared_ptr<xc_func_type> funcCPtr);

    excDensityLLMGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                          std::shared_ptr<xc_func_type> funcCPtr,
                          std::string                   modelXCInputFile);

    ~excDensityLLMGGAClass();


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
    NNLLMGGA                     *d_NNLLMGGAPtr;
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
    std::vector<double>           d_spacingFDStencil;
    dftfe::uInt                   d_vxcDivergenceTermFDStencilSize;
  };
} // namespace dftfe
#endif // DFTFE_EXCDENSITYLLMGGACLASS_H
