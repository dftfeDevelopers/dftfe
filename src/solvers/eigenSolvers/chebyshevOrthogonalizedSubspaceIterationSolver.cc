// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri, Sambit Das

#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <dftUtils.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsCPU.h>
#include <vectorUtilities.h>

static const unsigned int order_lookup[][2] = {
  {500, 24}, // <= 500 ~> chebyshevOrder = 24
  {750, 30},
  {1000, 39},
  {1500, 50},
  {2000, 53},
  {3000, 57},
  {4000, 62},
  {5000, 69},
  {9000, 77},
  {14000, 104},
  {20000, 119},
  {30000, 162},
  {50000, 300},
  {80000, 450},
  {100000, 550},
  {200000, 700},
  {500000, 1000}};

namespace dftfe
{
  namespace chebyshevOrthogonalizedSubspaceIterationSolverInternal
  {
    unsigned int
    setChebyshevOrder(const unsigned int upperBoundUnwantedSpectrum)
    {
      for (int i = 0; i < sizeof(order_lookup) / sizeof(order_lookup[0]); i++)
        {
          if (upperBoundUnwantedSpectrum <= order_lookup[i][0])
            return order_lookup[i][1];
        }
      return 1250;
    }
    void
    pointWiseScaleWithDiagonal(const double *     diagonal,
                               const unsigned int numberFields,
                               const unsigned int numberDofs,
                               dataTypes::number *fieldsArrayFlattened)
    {
      const unsigned int inc = 1;

      for (unsigned int i = 0; i < numberDofs; ++i)
        {
#ifdef USE_COMPLEX
          double scalingCoeff = diagonal[i];
          zdscal_(&numberFields,
                  &scalingCoeff,
                  &fieldsArrayFlattened[i * numberFields],
                  &inc);
#else
          double scalingCoeff = diagonal[i];
          dscal_(&numberFields,
                 &scalingCoeff,
                 &fieldsArrayFlattened[i * numberFields],
                 &inc);
#endif
        }
    }
  } // namespace chebyshevOrthogonalizedSubspaceIterationSolverInternal

  //
  // Constructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolver::
    chebyshevOrthogonalizedSubspaceIterationSolver(
      const MPI_Comm &     mpi_comm_parent,
      const MPI_Comm &     mpi_comm_domain,
      double               lowerBoundWantedSpectrum,
      double               lowerBoundUnWantedSpectrum,
      double               upperBoundUnWantedSpectrum,
      const dftParameters &dftParams)
    : d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum)
    , d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum)
    , d_upperBoundUnWantedSpectrum(upperBoundUnWantedSpectrum)
    , d_mpiCommParent(mpi_comm_parent)
    , d_dftParams(dftParams)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dftParams.reproducible_output || dftParams.verbosity < 4 ?
                        dealii::TimerOutput::never :
                        dealii::TimerOutput::summary,
                      dealii::TimerOutput::wall_times)
  {}

  //
  // Destructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolver::
    ~chebyshevOrthogonalizedSubspaceIterationSolver()
  {
    //
    //
    //
    return;
  }

  //
  // reinitialize spectrum bounds
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolver::reinitSpectrumBounds(
    double lowerBoundWantedSpectrum,
    double lowerBoundUnWantedSpectrum,
    double upperBoundUnWantedSpectrum)
  {
    d_lowerBoundWantedSpectrum   = lowerBoundWantedSpectrum;
    d_lowerBoundUnWantedSpectrum = lowerBoundUnWantedSpectrum;
    d_upperBoundUnWantedSpectrum = upperBoundUnWantedSpectrum;
  }


  //
  // solve
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolver::solve(
    operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &                  BLASWrapperPtr,
    elpaScalaManager &   elpaScala,
    dataTypes::number *  eigenVectorsFlattened,
    dataTypes::number *  eigenVectorsRotFracDensityFlattened,
    const unsigned int   totalNumberWaveFunctions,
    const unsigned int   localVectorSize,
    std::vector<double> &eigenValues,
    std::vector<double> &residualNorms,
    const MPI_Comm &     interBandGroupComm,
    const MPI_Comm &     mpiCommDomain,
    const bool           isFirstFilteringCall,
    const bool           computeResidual,
    const bool           useMixedPrec,
    const bool           isFirstScf)
  {
    dealii::TimerOutput computingTimerStandard(
      mpiCommDomain,
      pcout,
      d_dftParams.reproducible_output || d_dftParams.verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

    unsigned int chebyshevOrder = d_dftParams.chebyshevOrder;
    //
    // set Chebyshev order
    //
    if (chebyshevOrder == 0)
      {
        chebyshevOrder =
          chebyshevOrthogonalizedSubspaceIterationSolverInternal::
            setChebyshevOrder(d_upperBoundUnWantedSpectrum);

        if (d_dftParams.orthogType.compare("CGS") == 0 &&
            !d_dftParams.isPseudopotential)
          chebyshevOrder *= 0.5;
      }

    chebyshevOrder =
      (isFirstScf && d_dftParams.isPseudopotential) ?
        chebyshevOrder *
          d_dftParams.chebyshevFilterPolyDegreeFirstScfScalingFactor :
        chebyshevOrder;

    //
    // output statements
    //
    if (d_dftParams.verbosity >= 2)
      {
        char buffer[100];

        sprintf(buffer,
                "%s:%18.10e\n",
                "upper bound of unwanted spectrum",
                d_upperBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s:%18.10e\n",
                "lower bound of unwanted spectrum",
                d_lowerBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s: %u\n\n",
                "Chebyshev polynomial degree",
                chebyshevOrder);
        pcout << buffer;
      }

    computingTimerStandard.enter_subsection("Chebyshev filtering on CPU");


    if (d_dftParams.verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpiCommDomain,
                                        "Before starting chebyshev filtering");



    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumberWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);
    const unsigned int totalNumberBlocks = std::ceil(
      (double)totalNumberWaveFunctions / (double)d_dftParams.chebyWfcBlockSize);
    const unsigned int vectorsBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize,
               bandGroupLowHighPlusOneIndices[1]);


    //
    // allocate storage for eigenVectorsFlattenedArray for multiple blocks
    //
    distributedCPUMultiVec<dataTypes::number> *eigenVectorsFlattenedArrayBlock =
      &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 0);

    distributedCPUMultiVec<dataTypes::number>
      *eigenVectorsFlattenedArrayBlock2 =
        &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 1);
    distributedCPUMultiVec<dataTypes::numberFP32>
      *eigenVectorsFlattenedArrayBlockFP32 =
        d_dftParams.useSinglePrecCheby ?
          &operatorMatrix.getScratchFEMultivectorSinglePrec(vectorsBlockSize,
                                                            0) :
          NULL;
    distributedCPUMultiVec<dataTypes::numberFP32>
      *eigenVectorsFlattenedArrayBlock2FP32 =
        d_dftParams.useSinglePrecCheby ?
          &operatorMatrix.getScratchFEMultivectorSinglePrec(vectorsBlockSize,
                                                            1) :
          NULL;

    std::vector<double> eigenValuesBlock(vectorsBlockSize);
    /// storage for cell wavefunction matrix
    std::vector<dataTypes::number> cellWaveFunctionMatrix;

    int startIndexBandParal = totalNumberWaveFunctions;
    int numVectorsBandParal = 0;
    for (unsigned int jvec = 0; jvec < totalNumberWaveFunctions;
         jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int BVec =
          std::min(vectorsBlockSize, totalNumberWaveFunctions - jvec);

        if ((jvec + BVec) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + BVec) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            if (jvec < startIndexBandParal)
              startIndexBandParal = jvec;
            numVectorsBandParal = jvec + BVec - startIndexBandParal;

            // create custom partitioned dealii array
            if (BVec != vectorsBlockSize)
              {
                eigenVectorsFlattenedArrayBlock =
                  &operatorMatrix.getScratchFEMultivector(BVec, 0);
                eigenVectorsFlattenedArrayBlock2 =
                  &operatorMatrix.getScratchFEMultivector(BVec, 1);
                if (d_dftParams.useSinglePrecCheby)
                  {
                    eigenVectorsFlattenedArrayBlockFP32 =
                      &operatorMatrix.getScratchFEMultivectorSinglePrec(BVec,
                                                                        0);
                    eigenVectorsFlattenedArrayBlock2FP32 =
                      &operatorMatrix.getScratchFEMultivectorSinglePrec(BVec,
                                                                        1);
                  }
              }

            // fill the eigenVectorsFlattenedArrayBlock from
            // eigenVectorsFlattenedArray
            computing_timer.enter_subsection(
              "Copy from full to block flattened array");
            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              std::copy(eigenVectorsFlattened +
                          iNode * totalNumberWaveFunctions + jvec,
                        eigenVectorsFlattened +
                          iNode * totalNumberWaveFunctions + jvec + BVec,
                        eigenVectorsFlattenedArrayBlock->data() + iNode * BVec);
            computing_timer.leave_subsection(
              "Copy from full to block flattened array");



            //
            // call Chebyshev filtering function only for the current block to
            // be filtered and does in-place filtering
            computing_timer.enter_subsection("Chebyshev filtering");
            if (d_dftParams.useSinglePrecCheby && !isFirstFilteringCall)
              {
                eigenValuesBlock.resize(BVec);
                for (unsigned int i = 0; i < BVec; i++)
                  {
                    eigenValuesBlock[i] = eigenValues[jvec + i];
                  }

                linearAlgebraOperations::chebyshevFilterSinglePrec(
                  BLASWrapperPtr,
                  operatorMatrix,
                  (*eigenVectorsFlattenedArrayBlock),
                  (*eigenVectorsFlattenedArrayBlock2),
                  (*eigenVectorsFlattenedArrayBlockFP32),
                  (*eigenVectorsFlattenedArrayBlock2FP32),
                  eigenValuesBlock,
                  chebyshevOrder,
                  d_lowerBoundUnWantedSpectrum,
                  d_upperBoundUnWantedSpectrum,
                  d_lowerBoundWantedSpectrum);
              }
            else
              linearAlgebraOperations::chebyshevFilter(
                operatorMatrix,
                *eigenVectorsFlattenedArrayBlock,
                *eigenVectorsFlattenedArrayBlock2,
                chebyshevOrder,
                d_lowerBoundUnWantedSpectrum,
                d_upperBoundUnWantedSpectrum,
                d_lowerBoundWantedSpectrum);

            computing_timer.leave_subsection("Chebyshev filtering");



            if (d_dftParams.verbosity >= 4)
              dftUtils::printCurrentMemoryUsage(
                mpiCommDomain, "During blocked chebyshev filtering");

            // copy the eigenVectorsFlattenedArrayBlock into
            // eigenVectorsFlattenedArray after filtering
            computing_timer.enter_subsection(
              "Copy from block to full flattened array");
            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              std::copy(eigenVectorsFlattenedArrayBlock->data() + iNode * BVec,
                        eigenVectorsFlattenedArrayBlock->data() +
                          (iNode + 1) * BVec,
                        eigenVectorsFlattened +
                          iNode * totalNumberWaveFunctions + jvec);

            computing_timer.leave_subsection(
              "Copy from block to full flattened array");
          }
        else
          {
            // set to zero wavefunctions which wont go through chebyshev
            // filtering inside a given band group
            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              for (unsigned int iWave = 0; iWave < BVec; ++iWave)
                eigenVectorsFlattened[iNode * totalNumberWaveFunctions + jvec +
                                      iWave] = dataTypes::number(0.0);
          }
      } // block loop


    if (numberBandGroups > 1)
      {
        if (!d_dftParams.bandParalOpt)
          {
            computing_timer.enter_subsection(
              "MPI All Reduce wavefunctions across all band groups");
            MPI_Barrier(interBandGroupComm);
            const unsigned int blockSize =
              d_dftParams.mpiAllReduceMessageBlockSizeMB * 1e+6 /
              sizeof(dataTypes::number);
            for (unsigned int i = 0;
                 i < totalNumberWaveFunctions * localVectorSize;
                 i += blockSize)
              {
                const unsigned int currentBlockSize =
                  std::min(blockSize,
                           totalNumberWaveFunctions * localVectorSize - i);
                MPI_Allreduce(MPI_IN_PLACE,
                              eigenVectorsFlattened + i,
                              currentBlockSize,
                              dataTypes::mpi_type_id(eigenVectorsFlattened),
                              MPI_SUM,
                              interBandGroupComm);
              }
            computing_timer.leave_subsection(
              "MPI All Reduce wavefunctions across all band groups");
          }
        else
          {
            computing_timer.enter_subsection(
              "MPI_Allgatherv across band groups");
            MPI_Barrier(interBandGroupComm);
            std::vector<dataTypes::number> eigenVectorsBandGroup(
              numVectorsBandParal * localVectorSize, 0);
            std::vector<dataTypes::number> eigenVectorsBandGroupTransposed(
              numVectorsBandParal * localVectorSize, 0);
            std::vector<dataTypes::number> eigenVectorsTransposed(
              totalNumberWaveFunctions * localVectorSize, 0);

            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              for (unsigned int iWave = 0; iWave < numVectorsBandParal; ++iWave)
                eigenVectorsBandGroup[iNode * numVectorsBandParal + iWave] =
                  eigenVectorsFlattened[iNode * totalNumberWaveFunctions +
                                        startIndexBandParal + iWave];


            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              for (unsigned int iWave = 0; iWave < numVectorsBandParal; ++iWave)
                eigenVectorsBandGroupTransposed[iWave * localVectorSize +
                                                iNode] =
                  eigenVectorsBandGroup[iNode * numVectorsBandParal + iWave];

            std::vector<int> recvcounts(numberBandGroups, 0);
            std::vector<int> displs(numberBandGroups, 0);

            int recvcount = numVectorsBandParal * localVectorSize;
            MPI_Allgather(&recvcount,
                          1,
                          MPI_INT,
                          &recvcounts[0],
                          1,
                          MPI_INT,
                          interBandGroupComm);

            int displ = startIndexBandParal * localVectorSize;
            MPI_Allgather(
              &displ, 1, MPI_INT, &displs[0], 1, MPI_INT, interBandGroupComm);

            MPI_Allgatherv(&eigenVectorsBandGroupTransposed[0],
                           numVectorsBandParal * localVectorSize,
                           dataTypes::mpi_type_id(
                             &eigenVectorsBandGroupTransposed[0]),
                           &eigenVectorsTransposed[0],
                           &recvcounts[0],
                           &displs[0],
                           dataTypes::mpi_type_id(&eigenVectorsTransposed[0]),
                           interBandGroupComm);


            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              for (unsigned int iWave = 0; iWave < totalNumberWaveFunctions;
                   ++iWave)
                eigenVectorsFlattened[iNode * totalNumberWaveFunctions +
                                      iWave] =
                  eigenVectorsTransposed[iWave * localVectorSize + iNode];
            MPI_Barrier(interBandGroupComm);
            computing_timer.leave_subsection(
              "MPI_Allgatherv across band groups");
          }
      }

    computingTimerStandard.leave_subsection("Chebyshev filtering on CPU");
    if (d_dftParams.verbosity >= 4)
      pcout << "ChebyShev Filtering Done: " << std::endl;
    //
    // scale the eigenVectors (initial guess of single atom wavefunctions or
    // previous guess) to convert into Lowden Orthonormalized FE basis multiply
    // by M^{1/2}
    chebyshevOrthogonalizedSubspaceIterationSolverInternal::
      pointWiseScaleWithDiagonal(operatorMatrix.getSqrtMassVector().data(),
                                 totalNumberWaveFunctions,
                                 localVectorSize,
                                 eigenVectorsFlattened);


    if (d_dftParams.orthogType.compare("CGS") == 0)
      {
        computing_timer.enter_subsection("Rayleigh-Ritz GEP");
        if (eigenValues.size() != totalNumberWaveFunctions)
          {
            linearAlgebraOperations::rayleighRitzGEPSpectrumSplitDirect(
              operatorMatrix,
              elpaScala,
              eigenVectorsFlattened,
              eigenVectorsRotFracDensityFlattened,
              totalNumberWaveFunctions,
              localVectorSize,
              totalNumberWaveFunctions - eigenValues.size(),
              d_mpiCommParent,
              interBandGroupComm,
              mpiCommDomain,
              useMixedPrec,
              eigenValues,
              d_dftParams);
          }
        else
          {
            linearAlgebraOperations::rayleighRitzGEP(operatorMatrix,
                                                     elpaScala,
                                                     eigenVectorsFlattened,
                                                     totalNumberWaveFunctions,
                                                     localVectorSize,
                                                     d_mpiCommParent,
                                                     interBandGroupComm,
                                                     mpiCommDomain,
                                                     eigenValues,
                                                     useMixedPrec,
                                                     d_dftParams);
          }
        computing_timer.leave_subsection("Rayleigh-Ritz GEP");

        computing_timer.enter_subsection("eigen vectors residuals opt");
        if (eigenValues.size() != totalNumberWaveFunctions)
          {
            linearAlgebraOperations::computeEigenResidualNorm(
              operatorMatrix,
              eigenVectorsRotFracDensityFlattened,
              eigenValues,
              eigenValues.size(),
              localVectorSize,
              d_mpiCommParent,
              mpiCommDomain,
              interBandGroupComm,
              residualNorms,
              d_dftParams);
          }
        else
          {
            linearAlgebraOperations::computeEigenResidualNorm(
              operatorMatrix,
              eigenVectorsFlattened,
              eigenValues,
              totalNumberWaveFunctions,
              localVectorSize,
              d_mpiCommParent,
              mpiCommDomain,
              interBandGroupComm,
              residualNorms,
              d_dftParams);
          }
        computing_timer.leave_subsection("eigen vectors residuals opt");
      }
    else if (d_dftParams.orthogType.compare("GS") == 0)
      {
        computing_timer.enter_subsection("Gram-Schmidt Orthogn Opt");
        linearAlgebraOperations::gramSchmidtOrthogonalization(
          eigenVectorsFlattened,
          totalNumberWaveFunctions,
          localVectorSize,
          mpiCommDomain);
        computing_timer.leave_subsection("Gram-Schmidt Orthogn Opt");

        if (d_dftParams.verbosity >= 4)
          pcout << "Orthogonalization Done: " << std::endl;

        computing_timer.enter_subsection("Rayleigh-Ritz proj Opt");

        if (eigenValues.size() != totalNumberWaveFunctions)
          {
            linearAlgebraOperations::rayleighRitzSpectrumSplitDirect(
              operatorMatrix,
              elpaScala,
              eigenVectorsFlattened,
              eigenVectorsRotFracDensityFlattened,
              totalNumberWaveFunctions,
              localVectorSize,
              totalNumberWaveFunctions - eigenValues.size(),
              d_mpiCommParent,
              interBandGroupComm,
              mpiCommDomain,
              useMixedPrec,
              eigenValues,
              d_dftParams);
          }
        else
          {
            linearAlgebraOperations::rayleighRitz(operatorMatrix,
                                                  elpaScala,
                                                  eigenVectorsFlattened,
                                                  totalNumberWaveFunctions,
                                                  localVectorSize,
                                                  d_mpiCommParent,
                                                  interBandGroupComm,
                                                  mpiCommDomain,
                                                  eigenValues,
                                                  d_dftParams,
                                                  false);
          }


        computing_timer.leave_subsection("Rayleigh-Ritz proj Opt");

        if (d_dftParams.verbosity >= 4)
          {
            pcout << "Rayleigh-Ritz Done: " << std::endl;
            pcout << std::endl;
          }

        if (computeResidual)
          {
            computing_timer.enter_subsection("eigen vectors residuals opt");

            if (eigenValues.size() != totalNumberWaveFunctions)
              {
                linearAlgebraOperations::computeEigenResidualNorm(
                  operatorMatrix,
                  eigenVectorsRotFracDensityFlattened,
                  eigenValues,
                  eigenValues.size(),
                  localVectorSize,
                  d_mpiCommParent,
                  mpiCommDomain,
                  interBandGroupComm,
                  residualNorms,
                  d_dftParams);
              }
            else
              linearAlgebraOperations::computeEigenResidualNorm(
                operatorMatrix,
                eigenVectorsFlattened,
                eigenValues,
                totalNumberWaveFunctions,
                localVectorSize,
                d_mpiCommParent,
                mpiCommDomain,
                interBandGroupComm,
                residualNorms,
                d_dftParams);
            computing_timer.leave_subsection("eigen vectors residuals opt");
          }
      }


    if (d_dftParams.verbosity >= 4)
      {
        pcout << "EigenVector Residual Computation Done: " << std::endl;
        pcout << std::endl;
      }

    //
    // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // the usual FE basis
    //
    chebyshevOrthogonalizedSubspaceIterationSolverInternal::
      pointWiseScaleWithDiagonal(
        operatorMatrix.getInverseSqrtMassVector().data(),
        totalNumberWaveFunctions,
        localVectorSize,
        eigenVectorsFlattened);

    if (eigenValues.size() != totalNumberWaveFunctions)
      {
        chebyshevOrthogonalizedSubspaceIterationSolverInternal::
          pointWiseScaleWithDiagonal(
            operatorMatrix.getInverseSqrtMassVector().data(),
            eigenValues.size(),
            localVectorSize,
            eigenVectorsRotFracDensityFlattened);
      }


    if (d_dftParams.verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(
        mpiCommDomain, "After all steps of subspace iteration");
  }


  //
  // solve
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolver::solve(
    operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
    std::vector<distributedCPUVec<double>> &           eigenVectors,
    std::vector<double> &                              eigenValues,
    std::vector<double> &                              residualNorms)
  {}

} // namespace dftfe
