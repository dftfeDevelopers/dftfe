#ifndef ATOMCENTEREDOORBTIALPOSTPROCESING_H
#define ATOMCENTEREDOORBTIALPOSTPROCESING_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionValenceDensitySpline.h"
#include "AtomCenteredSphericalFunctionCoreDensitySpline.h"
#include "AtomCenteredSphericalFunctionLocalPotentialSpline.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "AtomCenteredSphericalFunctionContainer.h"
#include "AtomicCenteredNonLocalOperator.h"
#include <memory>
#include <MemorySpaceType.h>
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>
#include "FEBasisOperations.h"
#include <BLASWrapper.h>
#include <xc.h>
#include <excManager.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class atomCenteredOrbitalsPostProcessing
  {
  public:
    atomCenteredOrbitalsPostProcessing(const MPI_Comm &   mpi_comm_parent,
                                       const MPI_Comm &   mpi_comm_domain,
                                       const std::string &scratchFolderName,
                                       const std::set<unsigned int> &atomTypes,
                                       const bool           reproducibleOutput,
                                       const int            verbosity,
                                       const bool           useDevice,
                                       const dftParameters *dftParamsPtr);
    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */

    void
    initialise(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        basisOperationsDevicePtr,
#endif
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        BLASWrapperPtrDevice,
#endif
      unsigned int                            sparsityPatternQuadratureId,
      unsigned int                            nlpspQuadratureId,
      const std::vector<std::vector<double>> &atomLocations,
      unsigned int                            numEigenValues);


    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */
    void
    initialiseNonLocalContribution(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double> &             kPointWeights,
      const std::vector<double> &             kPointCoordinates,
      const bool                              updateNonlocalSparsity);

    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

    double
    smearFunction(double x, const dftParameters *dftParamsPtr);

    std::unordered_map<unsigned int, std::string> LQnumToNameMap;

    void
    computeAtomCenteredEntries(
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
      const unsigned int                      totalNumWaveFunctions,
      const std::vector<std::vector<double>> &eigenValues,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
        &basisOperationsPtr,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        BLASWrapperPtrDevice,
#endif
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                 BLASWrapperPtrHost,
      const unsigned int         quadratureIndex,
      const std::vector<double> &kPointWeights,
      const MPI_Comm &           interBandGroupComm,
      const MPI_Comm &           interpoolComm,
      const dftParameters *      dftParamsPtr,
      double                     fermiEnergy,
      unsigned int               highestStateNscfSolve);

  private:
    const MPI_Comm             d_mpiCommParentPostProcessing;
    const MPI_Comm             d_mpiCommDomain;
    const unsigned int         d_this_mpi_process;
    std::string                d_dftfeScratchFolderName;
    std::set<unsigned int>     d_atomTypes;
    bool                       d_reproducible_output;
    unsigned int               d_verbosity;
    bool                       d_useDevice;
    dealii::ConditionalOStream pcout;
    unsigned int               d_sparsityPatternQuadratureId;
    unsigned int               d_nlpspQuadratureId;
    unsigned int               d_numEigenValues;

    void
    createAtomCenteredSphericalFunctionsForOrbitals();

    std::map<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>>
                        nlNumsMap;
    dealii::TimerOutput computing_timer;

    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicOrbitalFnsContainer;

    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicOrbitalFnsMap;

    std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
      d_nonLocalOperator;

    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorHostPtr;


    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperDevicePtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      d_BasisOperatorDevicePtr;
#endif
  };
} // namespace dftfe

#endif
