// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef chebyshevOrthogonalizedSubspaceIterationSolverDevice_h
#    define chebyshevOrthogonalizedSubspaceIterationSolverDevice_h


#    include "deviceDirectCCLWrapper.h"
#    include "headers.h"
#    include "operatorDevice.h"
#    include "elpaScalaManager.h"
#    include "dftParameters.h"
#    include <BLASWrapper.h>

namespace dftfe
{
  /**
   * @brief Concrete class implementing Chebyshev filtered orthogonalized subspace
   * iteration solver.
   * @author Sambit Das, Phani Motamarri
   */
  class chebyshevOrthogonalizedSubspaceIterationSolverDevice
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param mpi_comm_parent parent mpi communicator
     * @param mpi_comm_domain domain decomposition mpi communicator
     * @param lowerBoundWantedSpectrum Lower Bound of the Wanted Spectrum.
     * @param lowerBoundUnWantedSpectrum Lower Bound of the UnWanted Spectrum.
     */
    chebyshevOrthogonalizedSubspaceIterationSolverDevice(
      const MPI_Comm &     mpi_comm_parent,
      const MPI_Comm &     mpi_comm_domain,
      double               lowerBoundWantedSpectrum,
      double               lowerBoundUnWantedSpectrum,
      double               upperBoundUnWantedSpectrum,
      const dftParameters &dftParams);



    /**
     * @brief Solve a generalized eigen problem.
     */
    double
    solve(operatorDFTDeviceClass &               operatorMatrix,
          const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
            dftfe::utils::MemorySpace::DEVICE>> &BLASWrapperPtr,
          elpaScalaManager &                     elpaScala,
          dataTypes::number *                    eigenVectorsFlattenedDevice,
          dataTypes::number *      eigenVectorsRotFracDensityFlattenedDevice,
          const unsigned int       flattenedSize,
          const unsigned int       totalNumberWaveFunctions,
          std::vector<double> &    eigenValues,
          std::vector<double> &    residuals,
          utils::DeviceCCLWrapper &devicecclMpiCommDomain,
          const MPI_Comm &         interBandGroupComm,
          const bool               isFirstFilteringCall,
          const bool               computeResidual,
          const bool               useMixedPrecOverall = false,
          const bool               isFirstScf          = false);


    /**
     * @brief Used for XL-BOMD.
     */
    void
    solveNoRR(operatorDFTDeviceClass &               operatorMatrix,
              const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
                dftfe::utils::MemorySpace::DEVICE>> &BLASWrapperPtr,
              elpaScalaManager &                     elpaScala,
              dataTypes::number *      eigenVectorsFlattenedDevice,
              const unsigned int       flattenedSize,
              const unsigned int       totalNumberWaveFunctions,
              std::vector<double> &    eigenValues,
              utils::DeviceCCLWrapper &devicecclMpiCommDomain,
              const MPI_Comm &         interBandGroupComm,
              const unsigned int       numberPasses,
              const bool               useMixedPrecOverall);


    /**
     * @brief Used for LRD preconditioner, also required for XL-BOMD
     */
    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTDeviceClass &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                        BLASWrapperPtr,
      dataTypes::number *        eigenVectorsFlattenedDevice,
      const unsigned int         flattenedSize,
      const unsigned int         totalNumberWaveFunctions,
      const std::vector<double> &eigenValues,
      const double               fermiEnergy,
      std::vector<double> &      densityMatDerFermiEnergy,
      utils::DeviceCCLWrapper &  devicecclMpiCommDomain,
      const MPI_Comm &           interBandGroupComm,
      dftfe::elpaScalaManager &  elpaScala);


    /**
     * @brief reinit spectrum bounds
     */
    void
    reinitSpectrumBounds(double lowerBoundWantedSpectrum,
                         double lowerBoundUnWantedSpectrum,
                         double upperBoundUnWantedSpectrum);

  private:
    const MPI_Comm d_mpiCommParent;
    //
    // stores lower bound of wanted spectrum
    //
    double d_lowerBoundWantedSpectrum;

    //
    // stores lower bound of unwanted spectrum
    //
    double d_lowerBoundUnWantedSpectrum;

    //
    // stores upper bound of unwanted spectrum
    //
    double d_upperBoundUnWantedSpectrum;

    const dftParameters &d_dftParams;

    //
    // temporary parallel vectors needed for Chebyshev filtering
    //
    distributedDeviceVec<dataTypes::number> d_YArray;

    distributedDeviceVec<dataTypes::numberFP32>
      d_deviceFlattenedFloatArrayBlock;

    distributedDeviceVec<dataTypes::number> d_deviceFlattenedArrayBlock2;

    distributedDeviceVec<dataTypes::number> d_YArray2;

    distributedDeviceVec<dataTypes::number> d_projectorKetTimesVector2;

    bool d_isTemporaryParallelVectorsCreated;

    //
    // variables for printing out and timing
    //
    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;
  };
} // namespace dftfe
#  endif
#endif
