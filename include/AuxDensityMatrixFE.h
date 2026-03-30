//
// Created by Sambit Das.
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIXFE_H
#define DFTFE_AUXDM_AUXDENSITYMATRIXFE_H

#include <vector>
#include <utility>
#include <AuxDensityMatrix.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class AuxDensityMatrixFE : public AuxDensityMatrix<memorySpace>
  {
  public:
    // FIXME: to be implemented

    void
    setDensityMatrixComponents(
      const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                             &eigenVectorsFlattenedMemSpace,
      const std::vector<std::vector<double>> &fractionalOccupancies);


    void
    applyLocalOperations(
      const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
      std::unordered_map<
        DensityDescriptorDataAttributes,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityData) override;



    void
    applyLocalOperations(
      const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
      std::unordered_map<
        WfcDescriptorDataAttributes,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &wfcData) override;

    void
    evalOverlapMatrixStart(
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadpts,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadWt) override;

    void
    evalOverlapMatrixEnd(const MPI_Comm &mpiComm) override;

    virtual void
    projectDensityMatrixStart(
      const std::unordered_map<std::string, std::vector<dataTypes::number>>
        &projectionInputsDataType,
      const std::unordered_map<
        std::string,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                      &projectionInputsReal,
      const dftfe::Int iSpin) override;

    void
    projectDensityMatrixEnd(const MPI_Comm &mpiComm) override;

    /**
     * @brief Projects the quadrature density to aux basis (L2 projection).
     * This is actually a copy call. All the local partition quadrature points
     * must to be passed to this function in one go
     *
     * @param projectionInputs is a map from string to inputs needed
     *                          for projection.
     *      projectionInputs["quadpts"],
     *      projectionInputs["quadWt"],
     *      projectionInputs["densityFunc"]
     *      projectionInputs["gradDensityFunc"]
     *
     * densityFunc The density Values at quad points
     *                densityFunc(spin_index, quad_index),
     *                quad_index is fastest.
     *
     * gradDensityFunc The density Values at quad points
     *                gradDensityFunc(spin_index, quad_index,dim_index),
     *                dim_index is fastest.
     *
     */
    void
    projectDensityStart(
      const std::unordered_map<
        std::string,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &projectionInputs) override;

    void
    projectDensityEnd(const MPI_Comm &mpiComm) override;

    const std::vector<std::vector<double>> *
    getDensityMatrixComponents_occupancies() const;

    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> *
    getDensityMatrixComponents_wavefunctions() const;

  private:
    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      *d_eigenVectorsFlattenedMemSpacePtr;

    const std::vector<std::vector<double>> *d_fractionalOccupancies;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_densityValsTotalAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_densityValsSpinUpAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_densityValsSpinDownAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_magAxisAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_gradDensityValsSpinUpAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_gradDensityValsSpinDownAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_tauValsTotalAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_tauValsSpinUpAllQuads;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_tauValsSpinDownAllQuads;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_quadPointsAll;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_quadWeightsAll;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIXFE_H
