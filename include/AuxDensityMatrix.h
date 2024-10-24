//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIX_H
#define DFTFE_AUXDM_AUXDENSITYMATRIX_H

#include <vector>
#include <utility>
#include <map>
#include <string>
#include <unordered_map>
#include <mpi.h>
#include <dftUtils.h>

namespace dftfe
{
  enum class DensityDescriptorDataAttributes
  {
    valuesTotal,
    valuesSpinUp,
    valuesSpinDown,
    gradValuesSpinUp,
    gradValuesSpinDown,
    hessianSpinUp,
    hessianSpinDown,
    laplacianSpinUp,
    laplacianSpinDown
  };

  template <dftfe::utils::MemorySpace memorySpace>
  class AuxDensityMatrix
  {
  public:
    /**
     * @brief compute local descriptors of the aux basis electron-density
     * representation at the supplied set of points using
     */
    virtual void
    applyLocalOperations(
      const std::vector<double> &Points,
      std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
        &densityData) = 0;

    /**
     * @brief Compute aux basis overlap matrix batchwise contribution from
     * supplied set of quadrature points and their associated weights
     */
    virtual void
    evalOverlapMatrixStart(const std::vector<double> &quadpts,
                           const std::vector<double> &quadWt) = 0;

    /**
     * @brief for MPI accumulation
     */
    virtual void
    evalOverlapMatrixEnd(const MPI_Comm &mpiComm) = 0;

    // FIXME: to be extended to memoryspace
    /**
     * @brief Projects the KS density matrix to aux basis (L2 projection) batch wise
     */
    virtual void
    projectDensityMatrixStart(
      const std::unordered_map<std::string, std::vector<dataTypes::number>>
        &projectionInputsDataType,
      const std::unordered_map<std::string, std::vector<double>>
        &       projectionInputsReal,
      const int iSpin) = 0;

    /**
     * @brief for MPI accumulation
     */
    virtual void
    projectDensityMatrixEnd(const MPI_Comm &mpiComm) = 0;


    /**
     * @brief Projects the quadrature density to aux basis (L2 projection) batch wise
     */
    virtual void
    projectDensityStart(
      const std::unordered_map<std::string, std::vector<double>>
        &projectionInputs) = 0;

    /**
     * @brief for MPI accumulation
     */
    virtual void
    projectDensityEnd(const MPI_Comm &mpiComm) = 0;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIX_H
