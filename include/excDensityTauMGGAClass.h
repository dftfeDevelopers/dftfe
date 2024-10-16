#ifndef DFTFE_EXCDENSITYMGGACLASS_H
#define DFTFE_EXCDENSITYMGGACLASS_H

#include <xc.h>
#include <ExcSSDFunctionalBaseClass.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class excDensityMGGAClass : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    excDensityMGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                        std::shared_ptr<xc_func_type> funcCPtr);

    excDensityMGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                        std::shared_ptr<xc_func_type> funcCPtr,
                        std::string                   modelXCInputFile);

    ~excDensityMGGAClass();

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

  private:
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
  };

} // namespace dftfe


#endif
