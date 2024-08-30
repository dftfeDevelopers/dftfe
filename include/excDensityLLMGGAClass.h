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
    computeOutputXCData(
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
    applyWaveFunctionDependentFuncDer(
      const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        &                                                                src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const unsigned int inputVecSize,
      const double       factor,
      const unsigned int kPointIndex,
      const unsigned int spinIndex) override;
    void
    updateWaveFunctionDependentFuncDer(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) override;
    double
    computeWaveFunctionDependentExcEnergy(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) override;


  private:
    NNLLMGGA *                    d_NNLLMGGAPtr;
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
    std::vector<double>           d_spacingFDStencil;
    unsigned int                  d_vxcDivergenceTermFDStencilSize;
  };
} // namespace dftfe
#endif // DFTFE_EXCDENSITYLLMGGACLASS_H
