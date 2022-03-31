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
// @author Sambit Das
//

// source file for electron density related computations

#include <dftParameters.h>
#include <dftUtils.h>
#include <vectorUtilities.h>

namespace dftfe
{
  namespace
  {
    void
    sumRhoData(
      const dealii::DoFHandler<3> &                  dofHandler,
      std::map<dealii::CellId, std::vector<double>> *rhoValues,
      std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
      std::map<dealii::CellId, std::vector<double>> *rhoValuesSpinPolarized,
      std::map<dealii::CellId, std::vector<double>> *gradRhoValuesSpinPolarized,
      const bool                                     isGradRhoDataPresent,
      const MPI_Comm &                               interComm)
    {
      typename dealii::DoFHandler<3>::active_cell_iterator
        cell = dofHandler.begin_active(),
        endc = dofHandler.end();

      // gather density from inter communicator
      if (dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              const dealii::CellId cellId = cell->id();

              dealii::Utilities::MPI::sum((*rhoValues)[cellId],
                                          interComm,
                                          (*rhoValues)[cellId]);
              if (isGradRhoDataPresent)
                dealii::Utilities::MPI::sum((*gradRhoValues)[cellId],
                                            interComm,
                                            (*gradRhoValues)[cellId]);

              if (dftParameters::spinPolarized == 1)
                {
                  dealii::Utilities::MPI::sum(
                    (*rhoValuesSpinPolarized)[cellId],
                    interComm,
                    (*rhoValuesSpinPolarized)[cellId]);
                  if (isGradRhoDataPresent)
                    dealii::Utilities::MPI::sum(
                      (*gradRhoValuesSpinPolarized)[cellId],
                      interComm,
                      (*gradRhoValuesSpinPolarized)[cellId]);
                }
            }
    }

  } // namespace

  // constructor
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  DensityCalculator<FEOrder, FEOrderElectro>::DensityCalculator()
  {}

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  DensityCalculator<FEOrder, FEOrderElectro>::computeRhoFromPSI(
    const std::vector<std::vector<dataTypes::number>> &eigenVectorsInput,
    const std::vector<std::vector<dataTypes::number>> &eigenVectorsFracInput,
    const unsigned int                                 totalNumWaveFunctions,
    const unsigned int                                 Nfr,
    const std::vector<std::vector<double>> &           eigenValues,
    const double                                       fermiEnergy,
    const double                                       fermiEnergyUp,
    const double                                       fermiEnergyDown,
    const dealii::DoFHandler<3> &                      dofHandler,
    const dealii::AffineConstraints<double> &          constraints,
    const dealii::MatrixFree<3, double> &              mfData,
    const unsigned int                                 mfDofIndex,
    const unsigned int                                 mfQuadIndex,
    const std::vector<dealii::types::global_dof_index>
      &localProc_dof_indicesReal,
    const std::vector<dealii::types::global_dof_index>
      &                                            localProc_dof_indicesImag,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> *_rhoValues,
    std::map<dealii::CellId, std::vector<double>> *_gradRhoValues,
    std::map<dealii::CellId, std::vector<double>> *_rhoValuesSpinPolarized,
    std::map<dealii::CellId, std::vector<double>> *_gradRhoValuesSpinPolarized,
    const bool                                     isEvaluateGradRho,
    const MPI_Comm &                               interpoolcomm,
    const MPI_Comm &                               interBandGroupComm,
    const bool                                     isConsiderSpectrumSplitting,
    const bool                                     lobattoNodesFlag)
  {
    int this_process;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_process);
    MPI_Barrier(MPI_COMM_WORLD);
    double cpu_time = MPI_Wtime();

#ifdef USE_COMPLEX
    dealii::FEEvaluation<
      3,
      FEOrder,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      2>
      psiEval(mfData, mfDofIndex, mfQuadIndex);
    dealii::FEEvaluation<3,
                         FEOrder,
                         C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1,
                         2>
      psiEvalGL(mfData, mfDofIndex, mfQuadIndex);
#else
    dealii::FEEvaluation<
      3,
      FEOrder,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1>
      psiEval(mfData, mfDofIndex, mfQuadIndex);
    dealii::FEEvaluation<3,
                         FEOrder,
                         C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1,
                         1>
      psiEvalGL(mfData, mfDofIndex, mfQuadIndex);
#endif

    dftUtils::constraintMatrixInfo constraintsNoneDataInfo;
    dftUtils::constraintMatrixInfo constraintsNoneDataInfo2;

    constraintsNoneDataInfo.initialize(mfData.get_vector_partitioner(0),
                                       constraints);

    constraintsNoneDataInfo2.initialize(mfData.get_vector_partitioner(0),
                                        constraints);

    distributedCPUVec<double> tempEigenVec;
    mfData.initialize_dof_vector(tempEigenVec, mfDofIndex);

    const unsigned int numEigenVectorsTotal = totalNumWaveFunctions;
    const unsigned int numEigenVectorsFrac  = Nfr;
    const unsigned int numEigenVectorsCore  = numEigenVectorsTotal - Nfr;
    const unsigned int numKPoints           = kPointWeights.size();

    const unsigned int numQuadPoints =
      lobattoNodesFlag ? psiEvalGL.n_q_points : psiEval.n_q_points;

    // initialization to zero
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellId = cell->id();
          (*_rhoValues)[cellId]       = std::vector<double>(numQuadPoints, 0.0);
          if (dftParameters::xcFamilyType == "GGA")
            (*_gradRhoValues)[cellId] =
              std::vector<double>(3 * numQuadPoints, 0.0);

          if (dftParameters::spinPolarized == 1)
            {
              (*_rhoValuesSpinPolarized)[cellId] =
                std::vector<double>(2 * numQuadPoints, 0.0);
              if (dftParameters::xcFamilyType == "GGA")
                (*_gradRhoValuesSpinPolarized)[cellId] =
                  std::vector<double>(6 * numQuadPoints, 0.0);
            }
        }

    dealii::Tensor<1, 2, dealii::VectorizedArray<double>> zeroTensor1;
    zeroTensor1[0] = dealii::make_vectorized_array(0.0);
    zeroTensor1[1] = dealii::make_vectorized_array(0.0);
    dealii::Tensor<1, 2, dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
                                                          zeroTensor2;
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        zeroTensor2[0][idim] = dealii::make_vectorized_array(0.0);
        zeroTensor2[1][idim] = dealii::make_vectorized_array(0.0);
        zeroTensor3[idim]    = dealii::make_vectorized_array(0.0);
      }

    // temp arrays
    std::vector<double> rhoTemp(numQuadPoints),
      rhoTempSpinPolarized(2 * numQuadPoints), rho(numQuadPoints),
      rhoSpinPolarized(2 * numQuadPoints);
    std::vector<double> gradRhoTemp(3 * numQuadPoints),
      gradRhoTempSpinPolarized(6 * numQuadPoints), gradRho(3 * numQuadPoints),
      gradRhoSpinPolarized(6 * numQuadPoints);


    std::vector<std::vector<double>> partialOccupancies(
      kPointWeights.size(),
      std::vector<double>((1 + dftParameters::spinPolarized) *
                            totalNumWaveFunctions,
                          0.0));
    for (unsigned int spinIndex = 0;
         spinIndex < (1 + dftParameters::spinPolarized);
         ++spinIndex)
      for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
        for (unsigned int iWave = 0; iWave < totalNumWaveFunctions; ++iWave)
          {
            const double eigenValue =
              eigenValues[kPoint][totalNumWaveFunctions * spinIndex + iWave];
            partialOccupancies[kPoint][totalNumWaveFunctions * spinIndex +
                                       iWave] =
              dftUtils::getPartialOccupancy(eigenValue,
                                            fermiEnergy,
                                            C_kb,
                                            dftParameters::TVal);

            if (dftParameters::constraintMagnetization)
              {
                partialOccupancies[kPoint][totalNumWaveFunctions * spinIndex +
                                           iWave] = 1.0;
                if (spinIndex == 0)
                  {
                    if (eigenValue > fermiEnergyUp)
                      partialOccupancies[kPoint]
                                        [totalNumWaveFunctions * spinIndex +
                                         iWave] = 0.0;
                  }
                else if (spinIndex == 1)
                  {
                    if (eigenValue > fermiEnergyDown)
                      partialOccupancies[kPoint]
                                        [totalNumWaveFunctions * spinIndex +
                                         iWave] = 0.0;
                  }
              }
          }

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               numEigenVectorsTotal,
                                               bandGroupLowHighPlusOneIndices);

    const unsigned int eigenVectorsBlockSize =
      std::min(dftParameters::wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

    const unsigned int localVectorSize =
      eigenVectorsInput[0].size() / numEigenVectorsTotal;

    std::vector<std::vector<distributedCPUVec<double>>> eigenVectors(
      (1 + dftParameters::spinPolarized) * kPointWeights.size());
    std::vector<distributedCPUVec<dataTypes::number>>
      eigenVectorsFlattenedBlock((1 + dftParameters::spinPolarized) *
                                 kPointWeights.size());


    std::vector<std::vector<distributedCPUVec<double>>> eigenVectorsRotFrac(
      (1 + dftParameters::spinPolarized) * kPointWeights.size());
    std::vector<distributedCPUVec<dataTypes::number>>
      eigenVectorsRotFracFlattenedBlock((1 + dftParameters::spinPolarized) *
                                        kPointWeights.size());

    for (unsigned int ivec = 0; ivec < numEigenVectorsTotal;
         ivec += eigenVectorsBlockSize)
      {
        const unsigned int currentBlockSize =
          std::min(eigenVectorsBlockSize, numEigenVectorsTotal - ivec);

        if (currentBlockSize != eigenVectorsBlockSize || ivec == 0)
          {
            for (unsigned int kPoint = 0;
                 kPoint <
                 (1 + dftParameters::spinPolarized) * kPointWeights.size();
                 ++kPoint)
              {
                eigenVectors[kPoint].resize(currentBlockSize);
                for (unsigned int i = 0; i < currentBlockSize; ++i)
                  eigenVectors[kPoint][i].reinit(tempEigenVec);


                vectorTools::createDealiiVector<dataTypes::number>(
                  mfData.get_vector_partitioner(0),
                  currentBlockSize,
                  eigenVectorsFlattenedBlock[kPoint]);
                eigenVectorsFlattenedBlock[kPoint] = dataTypes::number(0.0);
              }

            constraintsNoneDataInfo.precomputeMaps(
              mfData.get_vector_partitioner(0),
              eigenVectorsFlattenedBlock[0].get_partitioner(),
              currentBlockSize);
          }

        const bool isRotFracEigenVectorsInBlock =
          ((numEigenVectorsFrac != numEigenVectorsTotal) &&
           (ivec + currentBlockSize) > numEigenVectorsCore &&
           isConsiderSpectrumSplitting) ?
            true :
            false;

        unsigned int currentBlockSizeFrac    = eigenVectorsBlockSize;
        unsigned int startingIndexFracGlobal = 0;
        unsigned int startingIndexFrac       = 0;
        if (isRotFracEigenVectorsInBlock)
          {
            if (ivec < numEigenVectorsCore)
              {
                currentBlockSizeFrac =
                  ivec + currentBlockSize - numEigenVectorsCore;
                startingIndexFracGlobal = 0;
                startingIndexFrac       = numEigenVectorsCore - ivec;
              }
            else
              {
                currentBlockSizeFrac    = currentBlockSize;
                startingIndexFracGlobal = ivec - numEigenVectorsCore;
                startingIndexFrac       = 0;
              }

            if (currentBlockSizeFrac != eigenVectorsRotFrac[0].size() ||
                eigenVectorsRotFrac[0].size() == 0)
              {
                for (unsigned int kPoint = 0;
                     kPoint <
                     (1 + dftParameters::spinPolarized) * kPointWeights.size();
                     ++kPoint)
                  {
                    eigenVectorsRotFrac[kPoint].resize(currentBlockSizeFrac);
                    for (unsigned int i = 0; i < currentBlockSizeFrac; ++i)
                      eigenVectorsRotFrac[kPoint][i].reinit(tempEigenVec);


                    vectorTools::createDealiiVector<dataTypes::number>(
                      mfData.get_vector_partitioner(0),
                      currentBlockSizeFrac,
                      eigenVectorsRotFracFlattenedBlock[kPoint]);
                    eigenVectorsRotFracFlattenedBlock[kPoint] =
                      dataTypes::number(0.0);
                  }

                constraintsNoneDataInfo2.precomputeMaps(
                  mfData.get_vector_partitioner(0),
                  eigenVectorsRotFracFlattenedBlock[0].get_partitioner(),
                  currentBlockSizeFrac);
              }
          }

        if ((ivec + currentBlockSize) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (ivec + currentBlockSize) >
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            for (unsigned int kPoint = 0;
                 kPoint <
                 (1 + dftParameters::spinPolarized) * kPointWeights.size();
                 ++kPoint)
              {
                for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
                  for (unsigned int iWave = 0; iWave < currentBlockSize;
                       ++iWave)
                    eigenVectorsFlattenedBlock[kPoint].local_element(
                      iNode * currentBlockSize + iWave) =
                      eigenVectorsInput[kPoint][iNode * numEigenVectorsTotal +
                                                ivec + iWave];

                constraintsNoneDataInfo.distribute(
                  eigenVectorsFlattenedBlock[kPoint], currentBlockSize);
                eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
                vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                  eigenVectorsFlattenedBlock[kPoint],
                  currentBlockSize,
                  std::make_pair(0, currentBlockSize),
                  localProc_dof_indicesReal,
                  localProc_dof_indicesImag,
                  eigenVectors[kPoint],
                  false);

                // FIXME: The underlying call to update_ghost_values
                // is required because currently localProc_dof_indicesReal
                // and localProc_dof_indicesImag are only available for
                // locally owned nodes. Once they are also made available
                // for ghost nodes- use true for the last argument in
                // copyFlattenedDealiiVecToSingleCompVec(..) above and supress
                // underlying call.
                for (unsigned int i = 0; i < currentBlockSize; ++i)
                  eigenVectors[kPoint][i].update_ghost_values();
#else
                vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                  eigenVectorsFlattenedBlock[kPoint],
                  currentBlockSize,
                  std::make_pair(0, currentBlockSize),
                  eigenVectors[kPoint],
                  true);

#endif

                if (isRotFracEigenVectorsInBlock)
                  {
                    for (unsigned int iNode = 0; iNode < localVectorSize;
                         ++iNode)
                      for (unsigned int iWave = 0; iWave < currentBlockSizeFrac;
                           ++iWave)
                        eigenVectorsRotFracFlattenedBlock[kPoint].local_element(
                          iNode * currentBlockSizeFrac + iWave) =
                          eigenVectorsFracInput[kPoint]
                                               [iNode * numEigenVectorsFrac +
                                                startingIndexFracGlobal +
                                                iWave];

                    constraintsNoneDataInfo2.distribute(
                      eigenVectorsRotFracFlattenedBlock[kPoint],
                      currentBlockSizeFrac);
                    eigenVectorsRotFracFlattenedBlock[kPoint]
                      .update_ghost_values();

#ifdef USE_COMPLEX
                    vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                      eigenVectorsRotFracFlattenedBlock[kPoint],
                      currentBlockSizeFrac,
                      std::make_pair(0, currentBlockSizeFrac),
                      localProc_dof_indicesReal,
                      localProc_dof_indicesImag,
                      eigenVectorsRotFrac[kPoint],
                      false);

                    // FIXME: The underlying call to update_ghost_values
                    // is required because currently localProc_dof_indicesReal
                    // and localProc_dof_indicesImag are only available for
                    // locally owned nodes. Once they are also made available
                    // for ghost nodes- use true for the last argument in
                    // copyFlattenedDealiiVecToSingleCompVec(..) above and
                    // supress underlying call.
                    for (unsigned int i = 0; i < currentBlockSizeFrac; ++i)
                      eigenVectorsRotFrac[kPoint][i].update_ghost_values();
#else
                    vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                      eigenVectorsRotFracFlattenedBlock[kPoint],
                      currentBlockSizeFrac,
                      std::make_pair(0, currentBlockSizeFrac),
                      eigenVectorsRotFrac[kPoint],
                      true);

#endif
                  }
              }

#ifdef USE_COMPLEX
            dealii::AlignedVector<
              dealii::Tensor<1, 2, dealii::VectorizedArray<double>>>
              psiQuads(numQuadPoints * currentBlockSize * numKPoints,
                       zeroTensor1);
            dealii::AlignedVector<
              dealii::Tensor<1, 2, dealii::VectorizedArray<double>>>
              psiQuads2(numQuadPoints * currentBlockSize * numKPoints,
                        zeroTensor1);
            dealii::AlignedVector<dealii::Tensor<
              1,
              2,
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>>
              gradPsiQuads(numQuadPoints * currentBlockSize * numKPoints,
                           zeroTensor2);
            dealii::AlignedVector<dealii::Tensor<
              1,
              2,
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>>
              gradPsiQuads2(numQuadPoints * currentBlockSize * numKPoints,
                            zeroTensor2);

            dealii::AlignedVector<
              dealii::Tensor<1, 2, dealii::VectorizedArray<double>>>
              psiRotFracQuads(numQuadPoints * currentBlockSizeFrac * numKPoints,
                              zeroTensor1);
            dealii::AlignedVector<
              dealii::Tensor<1, 2, dealii::VectorizedArray<double>>>
              psiRotFracQuads2(numQuadPoints * currentBlockSizeFrac *
                                 numKPoints,
                               zeroTensor1);
            dealii::AlignedVector<dealii::Tensor<
              1,
              2,
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>>
              gradPsiRotFracQuads(numQuadPoints * currentBlockSizeFrac *
                                    numKPoints,
                                  zeroTensor2);
            dealii::AlignedVector<dealii::Tensor<
              1,
              2,
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>>
              gradPsiRotFracQuads2(numQuadPoints * currentBlockSizeFrac *
                                     numKPoints,
                                   zeroTensor2);
#else
            dealii::AlignedVector<dealii::VectorizedArray<double>> psiQuads(
              numQuadPoints * currentBlockSize,
              dealii::make_vectorized_array(0.0));
            dealii::AlignedVector<dealii::VectorizedArray<double>> psiQuads2(
              numQuadPoints * currentBlockSize,
              dealii::make_vectorized_array(0.0));
            dealii::AlignedVector<
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
              gradPsiQuads(numQuadPoints * currentBlockSize, zeroTensor3);
            dealii::AlignedVector<
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
              gradPsiQuads2(numQuadPoints * currentBlockSize, zeroTensor3);

            dealii::AlignedVector<dealii::VectorizedArray<double>>
              psiRotFracQuads(numQuadPoints * currentBlockSizeFrac,
                              dealii::make_vectorized_array(0.0));
            dealii::AlignedVector<dealii::VectorizedArray<double>>
              psiRotFracQuads2(numQuadPoints * currentBlockSizeFrac,
                               dealii::make_vectorized_array(0.0));
            dealii::AlignedVector<
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
              gradPsiRotFracQuads(numQuadPoints * currentBlockSizeFrac,
                                  zeroTensor3);
            dealii::AlignedVector<
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
              gradPsiRotFracQuads2(numQuadPoints * currentBlockSizeFrac,
                                   zeroTensor3);
#endif

            for (unsigned int cell = 0; cell < mfData.n_macro_cells(); ++cell)
              {
                lobattoNodesFlag ? psiEvalGL.reinit(cell) :
                                   psiEval.reinit(cell);

                const unsigned int numSubCells =
                  mfData.n_components_filled(cell);

                for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
                  for (unsigned int iEigenVec = 0; iEigenVec < currentBlockSize;
                       ++iEigenVec)
                    {
                      lobattoNodesFlag ?
                        psiEvalGL.read_dof_values_plain(
                          eigenVectors[(1 + dftParameters::spinPolarized) *
                                       kPoint][iEigenVec]) :
                        psiEval.read_dof_values_plain(
                          eigenVectors[(1 + dftParameters::spinPolarized) *
                                       kPoint][iEigenVec]);

                      if (isEvaluateGradRho)
                        lobattoNodesFlag ? psiEvalGL.evaluate(true, true) :
                                           psiEval.evaluate(true, true);
                      else
                        lobattoNodesFlag ? psiEvalGL.evaluate(true, false) :
                                           psiEval.evaluate(true, false);

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          psiQuads[q * currentBlockSize * numKPoints +
                                   currentBlockSize * kPoint + iEigenVec] =
                            lobattoNodesFlag ? psiEvalGL.get_value(q) :
                                               psiEval.get_value(q);
                          if (isEvaluateGradRho)
                            gradPsiQuads[q * currentBlockSize * numKPoints +
                                         currentBlockSize * kPoint +
                                         iEigenVec] =
                              lobattoNodesFlag ? psiEvalGL.get_gradient(q) :
                                                 psiEval.get_gradient(q);
                        }

                      if (dftParameters::spinPolarized == 1)
                        {
                          lobattoNodesFlag ?
                            psiEvalGL.read_dof_values_plain(
                              eigenVectors[(1 + dftParameters::spinPolarized) *
                                             kPoint +
                                           1][iEigenVec]) :
                            psiEval.read_dof_values_plain(
                              eigenVectors[(1 + dftParameters::spinPolarized) *
                                             kPoint +
                                           1][iEigenVec]);

                          if (isEvaluateGradRho)
                            lobattoNodesFlag ? psiEvalGL.evaluate(true, true) :
                                               psiEval.evaluate(true, true);
                          else
                            lobattoNodesFlag ? psiEvalGL.evaluate(true, false) :
                                               psiEval.evaluate(true, false);


                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              psiQuads2[q * currentBlockSize * numKPoints +
                                        currentBlockSize * kPoint + iEigenVec] =
                                lobattoNodesFlag ? psiEvalGL.get_value(q) :
                                                   psiEval.get_value(q);
                              if (isEvaluateGradRho)
                                gradPsiQuads2[q * currentBlockSize *
                                                numKPoints +
                                              currentBlockSize * kPoint +
                                              iEigenVec] =
                                  lobattoNodesFlag ? psiEvalGL.get_gradient(q) :
                                                     psiEval.get_gradient(q);
                            }
                        }

                      if (isRotFracEigenVectorsInBlock &&
                          iEigenVec >= startingIndexFrac)
                        {
                          const unsigned int vectorIndex =
                            iEigenVec - startingIndexFrac;

                          lobattoNodesFlag ?
                            psiEvalGL.read_dof_values_plain(
                              eigenVectorsRotFrac
                                [(1 + dftParameters::spinPolarized) * kPoint]
                                [vectorIndex]) :
                            psiEval.read_dof_values_plain(
                              eigenVectorsRotFrac
                                [(1 + dftParameters::spinPolarized) * kPoint]
                                [vectorIndex]);


                          if (isEvaluateGradRho)
                            lobattoNodesFlag ? psiEvalGL.evaluate(true, true) :
                                               psiEval.evaluate(true, true);
                          else
                            lobattoNodesFlag ? psiEvalGL.evaluate(true, false) :
                                               psiEval.evaluate(true, false);

                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              psiRotFracQuads[q * currentBlockSizeFrac *
                                                numKPoints +
                                              currentBlockSizeFrac * kPoint +
                                              vectorIndex] =
                                lobattoNodesFlag ? psiEvalGL.get_value(q) :
                                                   psiEval.get_value(q);
                              if (isEvaluateGradRho)
                                gradPsiRotFracQuads[q * currentBlockSizeFrac *
                                                      numKPoints +
                                                    currentBlockSizeFrac *
                                                      kPoint +
                                                    vectorIndex] =
                                  lobattoNodesFlag ? psiEvalGL.get_gradient(q) :
                                                     psiEval.get_gradient(q);
                            }

                          if (dftParameters::spinPolarized == 1)
                            {
                              lobattoNodesFlag ?
                                psiEvalGL.read_dof_values_plain(
                                  eigenVectorsRotFrac
                                    [(1 + dftParameters::spinPolarized) *
                                       kPoint +
                                     1][vectorIndex]) :
                                psiEval.read_dof_values_plain(
                                  eigenVectorsRotFrac
                                    [(1 + dftParameters::spinPolarized) *
                                       kPoint +
                                     1][vectorIndex]);

                              if (isEvaluateGradRho)
                                lobattoNodesFlag ?
                                  psiEvalGL.evaluate(true, true) :
                                  psiEval.evaluate(true, true);
                              else
                                lobattoNodesFlag ?
                                  psiEvalGL.evaluate(true, false) :
                                  psiEval.evaluate(true, false);

                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                {
                                  psiRotFracQuads2[q * currentBlockSizeFrac *
                                                     numKPoints +
                                                   currentBlockSizeFrac *
                                                     kPoint +
                                                   vectorIndex] =
                                    lobattoNodesFlag ? psiEvalGL.get_value(q) :
                                                       psiEval.get_value(q);
                                  if (isEvaluateGradRho)
                                    gradPsiRotFracQuads2
                                      [q * currentBlockSizeFrac * numKPoints +
                                       currentBlockSizeFrac * kPoint +
                                       vectorIndex] =
                                        lobattoNodesFlag ?
                                          psiEvalGL.get_gradient(q) :
                                          psiEval.get_gradient(q);
                                }
                            }
                        }

                    } // eigenvector per k point

                for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                     ++iSubCell)
                  {
                    const dealii::CellId subCellId =
                      mfData.get_cell_iterator(cell, iSubCell, mfDofIndex)
                        ->id();

                    std::fill(rhoTemp.begin(), rhoTemp.end(), 0.0);
                    std::fill(rho.begin(), rho.end(), 0.0);

                    if (dftParameters::spinPolarized == 1)
                      std::fill(rhoTempSpinPolarized.begin(),
                                rhoTempSpinPolarized.end(),
                                0.0);

                    if (isEvaluateGradRho)
                      {
                        std::fill(gradRhoTemp.begin(), gradRhoTemp.end(), 0.0);
                        if (dftParameters::spinPolarized == 1)
                          std::fill(gradRhoTempSpinPolarized.begin(),
                                    gradRhoTempSpinPolarized.end(),
                                    0.0);
                      }

                    for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
                      {
                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            double partialOccupancy =
                              partialOccupancies[kPoint][ivec + iEigenVec];
                            double partialOccupancy2 =
                              partialOccupancies[kPoint]
                                                [dftParameters::spinPolarized *
                                                   numEigenVectorsTotal +
                                                 ivec + iEigenVec];

                            for (unsigned int q = 0; q < numQuadPoints; ++q)
                              {
                                const unsigned int id =
                                  q * currentBlockSize * numKPoints +
                                  currentBlockSize * kPoint + iEigenVec;
#ifdef USE_COMPLEX
                                dealii::Vector<double> psi, psi2;
                                psi.reinit(2);
                                psi2.reinit(2);

                                psi(0) = psiQuads[id][0][iSubCell];
                                psi(1) = psiQuads[id][1][iSubCell];

                                if (dftParameters::spinPolarized == 1)
                                  {
                                    psi2(0) = psiQuads2[id][0][iSubCell];
                                    psi2(1) = psiQuads2[id][1][iSubCell];
                                  }

                                std::vector<dealii::Tensor<1, 3, double>>
                                  gradPsi(2), gradPsi2(2);

                                if (isEvaluateGradRho)
                                  for (unsigned int idim = 0; idim < 3; ++idim)
                                    {
                                      gradPsi[0][idim] =
                                        gradPsiQuads[id][0][idim][iSubCell];
                                      gradPsi[1][idim] =
                                        gradPsiQuads[id][1][idim][iSubCell];

                                      if (dftParameters::spinPolarized == 1)
                                        {
                                          gradPsi2[0][idim] =
                                            gradPsiQuads2[id][0][idim]
                                                         [iSubCell];
                                          gradPsi2[1][idim] =
                                            gradPsiQuads2[id][1][idim]
                                                         [iSubCell];
                                        }
                                    }
#else
                                double psi, psi2;
                                psi = psiQuads[id][iSubCell];
                                if (dftParameters::spinPolarized == 1)
                                  psi2 = psiQuads2[id][iSubCell];

                                dealii::Tensor<1, 3, double> gradPsi, gradPsi2;
                                if (isEvaluateGradRho)
                                  for (unsigned int idim = 0; idim < 3; ++idim)
                                    {
                                      gradPsi[idim] =
                                        gradPsiQuads[id][idim][iSubCell];
                                      if (dftParameters::spinPolarized == 1)
                                        gradPsi2[idim] =
                                          gradPsiQuads2[id][idim][iSubCell];
                                    }

#endif

                                if (isRotFracEigenVectorsInBlock &&
                                    iEigenVec >= startingIndexFrac)
                                  {
                                    const unsigned int idFrac =
                                      q * currentBlockSizeFrac * numKPoints +
                                      currentBlockSizeFrac * kPoint +
                                      iEigenVec - startingIndexFrac;
#ifdef USE_COMPLEX
                                    dealii::Vector<double> psiRotFrac,
                                      psiRotFrac2;
                                    psiRotFrac.reinit(2);
                                    psiRotFrac2.reinit(2);

                                    psiRotFrac(0) =
                                      psiRotFracQuads[idFrac][0][iSubCell];
                                    psiRotFrac(1) =
                                      psiRotFracQuads[idFrac][1][iSubCell];

                                    if (dftParameters::spinPolarized == 1)
                                      {
                                        psiRotFrac2(0) =
                                          psiRotFracQuads2[idFrac][0][iSubCell];
                                        psiRotFrac2(1) =
                                          psiRotFracQuads2[idFrac][1][iSubCell];
                                      }

                                    std::vector<dealii::Tensor<1, 3, double>>
                                      gradPsiRotFrac(2), gradPsiRotFrac2(2);

                                    if (isEvaluateGradRho)
                                      for (unsigned int idim = 0; idim < 3;
                                           ++idim)
                                        {
                                          gradPsiRotFrac[0][idim] =
                                            gradPsiRotFracQuads[idFrac][0][idim]
                                                               [iSubCell];
                                          gradPsiRotFrac[1][idim] =
                                            gradPsiRotFracQuads[idFrac][1][idim]
                                                               [iSubCell];

                                          if (dftParameters::spinPolarized == 1)
                                            {
                                              gradPsiRotFrac2[0][idim] =
                                                gradPsiRotFracQuads2[idFrac][0]
                                                                    [idim]
                                                                    [iSubCell];
                                              gradPsiRotFrac2[1][idim] =
                                                gradPsiRotFracQuads2[idFrac][1]
                                                                    [idim]
                                                                    [iSubCell];
                                            }
                                        }
#else
                                    double psiRotFrac, psiRotFrac2;
                                    psiRotFrac =
                                      psiRotFracQuads[idFrac][iSubCell];
                                    if (dftParameters::spinPolarized == 1)
                                      psiRotFrac2 =
                                        psiRotFracQuads2[idFrac][iSubCell];

                                    dealii::Tensor<1, 3, double> gradPsiRotFrac,
                                      gradPsiRotFrac2;
                                    if (isEvaluateGradRho)
                                      for (unsigned int idim = 0; idim < 3;
                                           ++idim)
                                        {
                                          gradPsiRotFrac[idim] =
                                            gradPsiRotFracQuads[idFrac][idim]
                                                               [iSubCell];
                                          if (dftParameters::spinPolarized == 1)
                                            gradPsiRotFrac2[idim] =
                                              gradPsiRotFracQuads2[idFrac][idim]
                                                                  [iSubCell];
                                        }

#endif

#ifdef USE_COMPLEX
                                    if (dftParameters::spinPolarized == 1)
                                      {
                                        rhoTempSpinPolarized[2 * q] +=
                                          kPointWeights[kPoint] *
                                          (partialOccupancy *
                                             (psiRotFrac(0) * psiRotFrac(0) +
                                              psiRotFrac(1) * psiRotFrac(1)) -
                                           (psiRotFrac(0) * psiRotFrac(0) +
                                            psiRotFrac(1) * psiRotFrac(1)) +
                                           (psi(0) * psi(0) + psi(1) * psi(1)));

                                        rhoTempSpinPolarized[2 * q + 1] +=
                                          kPointWeights[kPoint] *
                                          (partialOccupancy2 *
                                             (psiRotFrac2(0) * psiRotFrac2(0) +
                                              psiRotFrac2(1) * psiRotFrac2(1)) -
                                           (psiRotFrac2(0) * psiRotFrac2(0) +
                                            psiRotFrac2(1) * psiRotFrac2(1)) +
                                           (psi2(0) * psi2(0) +
                                            psi2(1) * psi2(1)));
                                        //
                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            {
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       idim] +=
                                                2.0 * kPointWeights[kPoint] *
                                                (partialOccupancy *
                                                   (psiRotFrac(0) *
                                                      gradPsiRotFrac[0][idim] +
                                                    psiRotFrac(1) *
                                                      gradPsiRotFrac[1][idim]) -
                                                 (psiRotFrac(0) *
                                                    gradPsiRotFrac[0][idim] +
                                                  psiRotFrac(1) *
                                                    gradPsiRotFrac[1][idim]) +
                                                 (psi(0) * gradPsi[0][idim] +
                                                  psi(1) * gradPsi[1][idim]));
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       3 +
                                                                       idim] +=
                                                2.0 * kPointWeights[kPoint] *
                                                (partialOccupancy2 *
                                                   (psiRotFrac2(0) *
                                                      gradPsiRotFrac2[0][idim] +
                                                    psiRotFrac2(1) *
                                                      gradPsiRotFrac2[1]
                                                                     [idim]) -
                                                 (psiRotFrac2(0) *
                                                    gradPsiRotFrac2[0][idim] +
                                                  psiRotFrac2(1) *
                                                    gradPsiRotFrac2[1][idim]) +
                                                 (psi2(0) * gradPsi2[0][idim] +
                                                  psi2(1) * gradPsi2[1][idim]));
                                            }
                                      }
                                    else
                                      {
                                        rhoTemp[q] +=
                                          2.0 * kPointWeights[kPoint] *
                                          (partialOccupancy *
                                             (psiRotFrac(0) * psiRotFrac(0) +
                                              psiRotFrac(1) * psiRotFrac(1)) -
                                           (psiRotFrac(0) * psiRotFrac(0) +
                                            psiRotFrac(1) * psiRotFrac(1)) +
                                           (psi(0) * psi(0) + psi(1) * psi(1)));
                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            gradRhoTemp[3 * q + idim] +=
                                              2.0 * 2.0 *
                                              kPointWeights[kPoint] *
                                              (partialOccupancy *
                                                 (psiRotFrac(0) *
                                                    gradPsiRotFrac[0][idim] +
                                                  psiRotFrac(1) *
                                                    gradPsiRotFrac[1][idim]) -
                                               (psiRotFrac(0) *
                                                  gradPsiRotFrac[0][idim] +
                                                psiRotFrac(1) *
                                                  gradPsiRotFrac[1][idim]) +
                                               (psi(0) * gradPsi[0][idim] +
                                                psi(1) * gradPsi[1][idim]));
                                      }
#else
                                    if (dftParameters::spinPolarized == 1)
                                      {
                                        rhoTempSpinPolarized[2 * q] +=
                                          (partialOccupancy * psiRotFrac *
                                             psiRotFrac -
                                           psiRotFrac * psiRotFrac + psi * psi);
                                        rhoTempSpinPolarized[2 * q + 1] +=
                                          (partialOccupancy2 * psiRotFrac2 *
                                             psiRotFrac2 -
                                           psiRotFrac2 * psiRotFrac2 +
                                           psi2 * psi2);

                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            {
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       idim] +=
                                                2.0 *
                                                (partialOccupancy * psiRotFrac *
                                                   gradPsiRotFrac[idim] -
                                                 psiRotFrac *
                                                   gradPsiRotFrac[idim] +
                                                 psi * gradPsi[idim]);
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       3 +
                                                                       idim] +=
                                                2.0 * (partialOccupancy2 *
                                                         psiRotFrac2 *
                                                         gradPsiRotFrac2[idim] -
                                                       psiRotFrac2 *
                                                         gradPsiRotFrac2[idim] +
                                                       psi2 * gradPsi2[idim]);
                                            }
                                      }
                                    else
                                      {
                                        rhoTemp[q] +=
                                          2.0 *
                                          (partialOccupancy * psiRotFrac *
                                             psiRotFrac -
                                           psiRotFrac * psiRotFrac + psi * psi);

                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            gradRhoTemp[3 * q + idim] +=
                                              2.0 * 2.0 *
                                              (partialOccupancy * psiRotFrac *
                                                 gradPsiRotFrac[idim] -
                                               psiRotFrac *
                                                 gradPsiRotFrac[idim] +
                                               psi * gradPsi[idim]);
                                      }

#endif
                                  }
                                else
                                  {
#ifdef USE_COMPLEX
                                    if (dftParameters::spinPolarized == 1)
                                      {
                                        rhoTempSpinPolarized[2 * q] +=
                                          partialOccupancy *
                                          kPointWeights[kPoint] *
                                          (psi(0) * psi(0) + psi(1) * psi(1));
                                        rhoTempSpinPolarized[2 * q + 1] +=
                                          partialOccupancy2 *
                                          kPointWeights[kPoint] *
                                          (psi2(0) * psi2(0) +
                                           psi2(1) * psi2(1));
                                        //
                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            {
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       idim] +=
                                                2.0 * partialOccupancy *
                                                kPointWeights[kPoint] *
                                                (psi(0) * gradPsi[0][idim] +
                                                 psi(1) * gradPsi[1][idim]);
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       3 +
                                                                       idim] +=
                                                2.0 * partialOccupancy2 *
                                                kPointWeights[kPoint] *
                                                (psi2(0) * gradPsi2[0][idim] +
                                                 psi2(1) * gradPsi2[1][idim]);
                                            }
                                      }
                                    else
                                      {
                                        rhoTemp[q] +=
                                          2.0 * partialOccupancy *
                                          kPointWeights[kPoint] *
                                          (psi(0) * psi(0) + psi(1) * psi(1));
                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            gradRhoTemp[3 * q + idim] +=
                                              2.0 * 2.0 * partialOccupancy *
                                              kPointWeights[kPoint] *
                                              (psi(0) * gradPsi[0][idim] +
                                               psi(1) * gradPsi[1][idim]);
                                      }
#else
                                    if (dftParameters::spinPolarized == 1)
                                      {
                                        rhoTempSpinPolarized[2 * q] +=
                                          partialOccupancy * psi * psi;
                                        rhoTempSpinPolarized[2 * q + 1] +=
                                          partialOccupancy2 * psi2 * psi2;

                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            {
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       idim] +=
                                                2.0 * partialOccupancy *
                                                (psi * gradPsi[idim]);
                                              gradRhoTempSpinPolarized[6 * q +
                                                                       3 +
                                                                       idim] +=
                                                2.0 * partialOccupancy2 *
                                                (psi2 * gradPsi2[idim]);
                                            }
                                      }
                                    else
                                      {
                                        rhoTemp[q] +=
                                          2.0 * partialOccupancy * psi * psi;

                                        if (isEvaluateGradRho)
                                          for (unsigned int idim = 0; idim < 3;
                                               ++idim)
                                            gradRhoTemp[3 * q + idim] +=
                                              2.0 * 2.0 * partialOccupancy *
                                              psi * gradPsi[idim];
                                      }

#endif
                                  }

                              } // quad point loop
                          }     // block eigenvectors per k point
                      }

                    std::vector<double>  dummy(1);
                    std::vector<double> &tempRhoQuadsCell =
                      (*_rhoValues)[subCellId];
                    std::vector<double> &tempGradRhoQuadsCell =
                      isEvaluateGradRho ? (*_gradRhoValues)[subCellId] : dummy;

                    std::vector<double> &tempRhoQuadsCellSP =
                      (dftParameters::spinPolarized == 1) ?
                        (*_rhoValuesSpinPolarized)[subCellId] :
                        dummy;
                    std::vector<double> &tempGradRhoQuadsCellSP =
                      ((dftParameters::spinPolarized == 1) &&
                       isEvaluateGradRho) ?
                        (*_gradRhoValuesSpinPolarized)[subCellId] :
                        dummy;

                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        if (dftParameters::spinPolarized == 1)
                          {
                            tempRhoQuadsCellSP[2 * q] +=
                              rhoTempSpinPolarized[2 * q];
                            tempRhoQuadsCellSP[2 * q + 1] +=
                              rhoTempSpinPolarized[2 * q + 1];

                            if (isEvaluateGradRho)
                              for (unsigned int idim = 0; idim < 3; ++idim)
                                {
                                  tempGradRhoQuadsCellSP[6 * q + idim] +=
                                    gradRhoTempSpinPolarized[6 * q + idim];
                                  tempGradRhoQuadsCellSP[6 * q + 3 + idim] +=
                                    gradRhoTempSpinPolarized[6 * q + 3 + idim];
                                }

                            tempRhoQuadsCell[q] +=
                              rhoTempSpinPolarized[2 * q] +
                              rhoTempSpinPolarized[2 * q + 1];

                            if (isEvaluateGradRho)
                              for (unsigned int idim = 0; idim < 3; ++idim)
                                tempGradRhoQuadsCell[3 * q + idim] +=
                                  gradRhoTempSpinPolarized[6 * q + idim] +
                                  gradRhoTempSpinPolarized[6 * q + 3 + idim];
                          }
                        else
                          {
                            tempRhoQuadsCell[q] += rhoTemp[q];

                            if (isEvaluateGradRho)
                              for (unsigned int idim = 0; idim < 3; ++idim)
                                tempGradRhoQuadsCell[3 * q + idim] +=
                                  gradRhoTemp[3 * q + idim];
                          }
                      }
                  } // subcell loop
              }     // macro cell loop
          }         // band parallelization
      }             // eigenvectors block loop

    // gather density from all inter communicators
    sumRhoData(dofHandler,
               _rhoValues,
               _gradRhoValues,
               _rhoValuesSpinPolarized,
               _gradRhoValuesSpinPolarized,
               isEvaluateGradRho,
               interBandGroupComm);

    sumRhoData(dofHandler,
               _rhoValues,
               _gradRhoValues,
               _rhoValuesSpinPolarized,
               _gradRhoValuesSpinPolarized,
               isEvaluateGradRho,
               interpoolcomm);

    MPI_Barrier(MPI_COMM_WORLD);
    cpu_time = MPI_Wtime() - cpu_time;

    if (this_process == 0 && dftParameters::verbosity >= 2)
      std::cout << "Time for compute rho on CPU: " << cpu_time << std::endl;
  }

#include "densityCalculator.inst.cc"
} // namespace dftfe
