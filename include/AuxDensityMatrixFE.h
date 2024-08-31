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
        &                                     eigenVectorsFlattenedMemSpace,
      const std::vector<std::vector<double>> &fractionalOccupancies);



    // CAUTION: points have to be a contiguous subset of d_quadPointsSet
    void
    applyLocalOperations(
      const std::vector<double> &points,
      std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
        &densityData) override;

    void
    evalOverlapMatrixStart(const std::vector<double> &quadpts,
                           const std::vector<double> &quadWt) override;

    void
    evalOverlapMatrixEnd(const MPI_Comm &mpiComm) override;

    virtual void
    projectDensityMatrixStart(
      const std::unordered_map<std::string, std::vector<dataTypes::number>>
        &projectionInputsDataType,
      const std::unordered_map<std::string, std::vector<double>>
        &       projectionInputsReal,
      const int iSpin) override;

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
      const std::unordered_map<std::string, std::vector<double>>
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

    std::vector<double> d_densityValsTotalAllQuads;
    std::vector<double> d_densityValsSpinUpAllQuads;
    std::vector<double> d_densityValsSpinDownAllQuads;
    std::vector<double> d_gradDensityValsSpinUpAllQuads;
    std::vector<double> d_gradDensityValsSpinDownAllQuads;
    std::vector<double> d_quadPointsAll;
    std::vector<double> d_quadWeightsAll;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIXFE_H
