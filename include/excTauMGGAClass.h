#ifndef DFTFE_excTauMGGAClass_H
#define DFTFE_excTauMGGAClass_H

#include <xc.h>
#include <ExcSSDFunctionalBaseClass.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class excTauMGGAClass : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    excTauMGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                    std::shared_ptr<xc_func_type> funcCPtr);

    excTauMGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                    std::shared_ptr<xc_func_type> funcCPtr,
                    std::string                   modelXCInputFile);

    ~excTauMGGAClass();

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
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
  };

} // namespace dftfe


#endif
