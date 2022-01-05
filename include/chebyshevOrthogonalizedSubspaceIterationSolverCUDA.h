// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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

#if defined(DFTFE_WITH_GPU)
#  ifndef chebyshevOrthogonalizedSubspaceIterationSolverCUDA_h
#    define chebyshevOrthogonalizedSubspaceIterationSolverCUDA_h


#    include "gpuDirectCCLWrapper.h"
#    include "headers.h"
#    include "operatorCUDA.h"
#    include "elpaScalaManager.h"

namespace dftfe
{
  /**
   * @brief Concrete class implementing Chebyshev filtered orthogonalized subspace
   * iteration solver.
   * @author Sambit Das, Phani Motamarri
   */
  class chebyshevOrthogonalizedSubspaceIterationSolverCUDA
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param mpi_comm domain decomposition mpi communicator
     * @param lowerBoundWantedSpectrum Lower Bound of the Wanted Spectrum.
     * @param lowerBoundUnWantedSpectrum Lower Bound of the UnWanted Spectrum.
     */
    chebyshevOrthogonalizedSubspaceIterationSolverCUDA(
      const MPI_Comm &mpi_comm_domain,
      double          lowerBoundWantedSpectrum,
      double          lowerBoundUnWantedSpectrum,
      double          upperBoundUnWantedSpectrum);



    /**
     * @brief Solve a generalized eigen problem.
     */
    double
    solve(operatorDFTCUDAClass &operatorMatrix,
          elpaScalaManager &    elpaScala,
          dataTypes::numberGPU *eigenVectorsFlattenedCUDA,
          dataTypes::numberGPU *eigenVectorsRotFracDensityFlattenedCUDA,
          const unsigned int    flattenedSize,
          const unsigned int    totalNumberWaveFunctions,
          std::vector<double> & eigenValues,
          std::vector<double> & residuals,
          GPUCCLWrapper &       gpucclMpiCommDomain,
          const MPI_Comm &      interBandGroupComm,
          const bool            isFirstFilteringCall,
          const bool            computeResidual,
          const bool            useMixedPrecOverall = false,
          const bool            isFirstScf          = false);


    /**
     * @brief Used for XL-BOMD.
     */
    void
    solveNoRR(operatorDFTCUDAClass &operatorMatrix,
              elpaScalaManager &    elpaScala,
              dataTypes::numberGPU *eigenVectorsFlattenedCUDA,
              const unsigned int    flattenedSize,
              const unsigned int    totalNumberWaveFunctions,
              std::vector<double> & eigenValues,
              GPUCCLWrapper &       gpucclMpiCommDomain,
              const MPI_Comm &      interBandGroupComm,
              const unsigned int    numberPasses,
              const bool            useMixedPrecOverall);


    /**
     * @brief Used for LRJI preconditioner, also required for XL-BOMD
     */
    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTCUDAClass &     operatorMatrix,
      dataTypes::numberGPU *     eigenVectorsFlattenedCUDA,
      const unsigned int         flattenedSize,
      const unsigned int         totalNumberWaveFunctions,
      const std::vector<double> &eigenValues,
      const double               fermiEnergy,
      std::vector<double> &      densityMatDerFermiEnergy,
      GPUCCLWrapper &            gpucclMpiCommDomain,
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

    //
    // temporary parallel vectors needed for Chebyshev filtering
    //
    distributedGPUVec<dataTypes::numberGPU> d_YArray;

    distributedGPUVec<dataTypes::numberFP32GPU> d_cudaFlattenedFloatArrayBlock;

    distributedGPUVec<dataTypes::numberGPU> d_cudaFlattenedArrayBlock2;

    distributedGPUVec<dataTypes::numberGPU> d_YArray2;

    distributedGPUVec<dataTypes::numberGPU> d_projectorKetTimesVector2;

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
